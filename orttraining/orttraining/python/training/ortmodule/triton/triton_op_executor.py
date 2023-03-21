import functools
import hashlib
import itertools
import json
import os
import sys
import types
from typing import List, Tuple

import onnx
from torch._C import _from_dlpack
from torch.utils.dlpack import to_dlpack

from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from ._codecache import PyCodeCache
from ._codegen import codegen
from ._node_sets import get_supported_ops


def _process_onnx_model(onnx_str: bytes, input_shapes: List[List[int]]) -> bytes:
    model = onnx.load_model_from_string(onnx_str)
    graph = model.graph
    name_map = {}
    node_idx = 0
    arg_idx = 0
    for node in graph.node:
        node.name = "node_" + str(node_idx)
        node_idx += 1
        for idx, name in enumerate(node.input):
            if name not in name_map:
                name_map[name] = "arg_" + str(arg_idx)
                arg_idx += 1
            node.input[idx] = name_map[name]
        for idx, name in enumerate(node.output):
            if name not in name_map:
                name_map[name] = "arg_" + str(arg_idx)
                arg_idx += 1
            node.output[idx] = name_map[name]
    for node in itertools.chain(graph.input, graph.output, graph.initializer):
        node.name = name_map[node.name]

    assert len(graph.input) == len(input_shapes)
    new_inputs = []
    for i, g_input in enumerate(graph.input):
        dtype = g_input.type.tensor_type.elem_type
        shape = input_shapes[i]
        tensor_type = onnx.helper.make_tensor_type_proto(elem_type=dtype, shape=shape)
        input_value = onnx.helper.make_value_info(g_input.name, tensor_type)
        new_inputs.append(input_value)
    graph.ClearField("input")
    graph.input.extend(new_inputs)
    new_model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
    return onnx._serialize(new_model)


@functools.lru_cache(None)
def _gen_module(onnx_str: bytes, debug_mode: bool) -> Tuple[str, types.ModuleType]:
    model = onnx.load_model_from_string(onnx_str)
    func_name = "triton_kernel_" + str(int(hashlib.sha1(onnx_str).hexdigest(), 16) % (10**8))
    src_code = codegen({func_name: model}, debug_mode)
    if debug_mode:
        py_file_path = "triton_debug/" + func_name + ".py"
        os.makedirs(os.path.dirname(py_file_path), exist_ok=True)
        with open(py_file_path, "w") as f:
            f.write(src_code)
        onnx.save(model, "triton_debug/" + func_name + ".onnx")
    return func_name, PyCodeCache().load(src_code)


def get_config():
    config = {"ops": get_supported_ops(), "initializer": "scalar"}
    return json.dumps(config)


def execute_triton_op(func_name: str, onnx_str: bytes, *tensors):
    torch_tensors = [_from_dlpack(tensor) for tensor in tensors]
    if not onnx_str:
        assert func_name
        func = getattr(sys.modules[".".join(__name__.split(".")[:-1])], func_name)
        output = func(*torch_tensors)
        if type(output) is tuple:
            return tuple([to_dlpack(tensor) for tensor in output])
        return to_dlpack(output)

    try:
        concrete_shapes = [list(tensor.size()) for tensor in torch_tensors]
        new_onnx_str = _process_onnx_model(onnx_str, concrete_shapes)
        func_name, mod = _gen_module(new_onnx_str, True)
        func = getattr(mod, f"launch_{func_name}")
        outputs = func(torch_tensors)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e
    if len(outputs) == 1:
        return to_dlpack(outputs[0])
    return tuple([to_dlpack(output) for output in outputs])
