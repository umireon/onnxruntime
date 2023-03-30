# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import copy
import enum
from abc import abstractmethod

import numpy as np
import onnx
from onnx import TensorProto, ValueInfoProto, helper, numpy_helper

from onnxruntime.tools import symbolic_shape_infer


class DataType(enum.Enum):
    UNDEFINED = 0
    # Basic types.
    FLOAT = 1  # float
    UINT8 = 2  # uint8_t
    INT8 = 3  # int8_t
    UINT16 = 4  # uint16_t
    INT16 = 5  # int16_t
    INT32 = 6  # int32_t
    INT64 = 7  # int64_t
    STRING = 8  # string
    BOOL = 9  # bool
    # Advanced types
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14  # complex with float32 real and imaginary components
    COMPLEX128 = 15  # complex with float64 real and imaginary components
    # Future extensions go here.


class AttributeType(enum.Enum):
    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    SPARSE_TENSOR = 11
    TYPE_PROTO = 13

    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSORS = 12
    TYPE_PROTOS = 14


# This map is used for converting TensorProto values into Numpy arrays
TENSOR_TYPE_TO_NP_TYPE = {
    int(TensorProto.FLOAT): np.dtype("float32"),
    int(TensorProto.UINT8): np.dtype("uint8"),
    int(TensorProto.INT8): np.dtype("int8"),
    int(TensorProto.UINT16): np.dtype("uint16"),
    int(TensorProto.INT16): np.dtype("int16"),
    int(TensorProto.INT32): np.dtype("int32"),
    int(TensorProto.INT64): np.dtype("int64"),
    int(TensorProto.BOOL): np.dtype("bool"),
    int(TensorProto.FLOAT16): np.dtype("float16"),
    # Native numpy does not support bfloat16 so now use float32 for bf16 values
    int(TensorProto.BFLOAT16): np.dtype("float32"),
    int(TensorProto.DOUBLE): np.dtype("float64"),
    int(TensorProto.COMPLEX64): np.dtype("complex64"),
    int(TensorProto.COMPLEX128): np.dtype("complex128"),
    int(TensorProto.UINT32): np.dtype("uint32"),
    int(TensorProto.UINT64): np.dtype("uint64"),
    int(TensorProto.STRING): np.dtype("object"),
}

# Currently native numpy does not support bfloat16 so TensorProto.BFLOAT16 is ignored for now
# Numpy float32 array is only reversed to TensorProto.FLOAT
NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items() if k != TensorProto.BFLOAT16}

NP_TYPE_C_TYPE = {
    np.bool_: "bool",
    np.byte: "int8_t",
    np.ubyte: "uint8_t",
    np.short: "short",
    np.ushort: "unsigned short",
    np.intc: "int",
    np.uintc: "unsigned int",
    np.int_: "long",
    np.int64: "int64_t",
    np.uint: "unsigned long",
    np.longlong: "long long",
    np.ulonglong: "unsigned long long",
    np.single: "float",
    np.float32: "float",
    np.double: "double",
    np.longdouble: "long double",
    np.float16: "float16",
}


class NodeVisitor(object):
    def __init__(self):
        pass

    # interface for codegen/lowering
    @abstractmethod
    def visit(self, node, context, indent: int):
        pass


class CodeGenContext(object):
    def __init__(self, var_map: dict):
        self.var_map = var_map
        self.vectorized_var_set = set()


class SpecialVar(object):
    def __init__(self):
        self.input_args = "input_args"
        self.output_args = "output_args"
        self.input_args_size = "input_args_size"
        self.parallel_loop_start = "p_loop_start"
        self.parallel_loop_end = "p_loop_end"
        self.dynamic_shape_args = "dynamic_shape_args"
        self.rbase = "rbase"
        self.rblock = "RBLOCK"


def parse_onnx_attributes(attributes) -> dict:
    """
    Parse ONNX attributes into a dictionary
    """
    attributes_map = {}
    if not attributes:
        return attributes_map
    for attribute in attributes:
        attributes_map[attribute.name] = helper.get_attribute_value(attribute)
    return attributes_map


def add_all_intermidiate_values(model):
    model_proto = copy.deepcopy(model)
    # model_proto, check = simplify(model_proto)
    org_outputs = [x.name for x in model_proto.graph.output]
    for node in model_proto.graph.node:
        for output in node.output:
            if output not in org_outputs:
                model_proto.graph.output.extend([ValueInfoProto(name=output)])
    return model_proto


def get_symbol_shape(model_path):
    if isinstance(model_path, str):
        model = onnx.load(model_path)
    else:
        model = model_path
    symbolic_shape_infer.logger.setLevel(symbolic_shape_infer.logging.ERROR)
    symbol_shape = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(
        model, 2**31 - 1, True, guess_output_rank=True, verbose=1
    )

    return symbol_shape


def parse_onnx_to_numpyarray(value):
    data = None
    if isinstance(value, onnx.onnx_ml_pb2.NodeProto):
        assert len(value.attribute) == 1
        Attr_type = AttributeType(value.attribute[0].type)
        if Attr_type == AttributeType.INT:
            data = np.array([value.attribute[0].i])
        elif Attr_type == AttributeType.TENSOR:
            value = value.attribute[0].t
        elif Attr_type == AttributeType.FLOAT:
            data = np.array([value.attribute[0].f])
        else:
            raise Exception("not support")
    # handle      AttributeType.TENSOR
    if isinstance(value, onnx.onnx_ml_pb2.TensorProto):
        assert data is None
        data = numpy_helper.to_array(value)
    elif data is not None:
        pass
    else:
        assert RuntimeError("not support proto type")
    return data


class GraphIOBuffer(object):
    def __init__(self):
        self.var_buffer_in = []
        self.var_buffer_out = []
        self.const_buffer = []


class HardwareContext(object):
    def __init__(self, vec_lanes):
        self.vec_lanes = vec_lanes
