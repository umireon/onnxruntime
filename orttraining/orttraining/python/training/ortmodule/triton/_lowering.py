# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from collections import defaultdict
from functools import wraps
from typing import Dict, List

from ._common import GraphIOBuffer, HardwareContext, NodeVisitor, parse_onnx_to_numpyarray
from ._execution_planner import ConnectionGraph, ExecutionPrepare
from ._ir import (
    ComputeBuffer,
    ExecutionBlock,
    FunctionNode,
    IRNode,
    MaskLoadNode,
    MaskStoreNode,
    ModuleNode,
    ReduceNode,
)
from ._op_config import is_reduction_node
from ._scheduling import GPUSchedule


def Singleton(cls):
    instances = {}

    @wraps(cls)
    def getinstance(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return getinstance


@Singleton
class UniqueNameGenerator(object):
    def __init__(self):
        self.count = 0

    def get_unique_var_name(self, prefix):
        self.count += 1
        return prefix + str(self.count)


def create_load_or_store(buf: ComputeBuffer, is_load: bool):
    mask_id_name = UniqueNameGenerator().get_unique_var_name("triton_mask_")
    mask_buf = ComputeBuffer(mask_id_name, shape=buf.shape[-1:])

    if is_load:
        return [MaskLoadNode(buf, mask_buf)]
    else:
        return [MaskStoreNode(buf, mask_buf)]


def insert_load_and_store(block: ExecutionBlock, global_buffer: GraphIOBuffer, c_graph: ConnectionGraph):
    input_name_map = {inp.name: inp for inp in block.input}
    output_name_map = {inp.name: inp for inp in block.output}
    new_group = []
    # avoid duplicate load
    load_cache = set()
    for g in block.group:
        for inp in g.input:
            producer_op = c_graph.egraph.produced_by[inp][0] if inp in c_graph.egraph.produced_by else None

            if (inp in block.load or inp in input_name_map) and not is_reduction_node(producer_op):
                load_buf = block.load[inp] if inp in block.load else input_name_map[inp]
                # we just skip unused load for constant scalar
                if (load_buf.data is not None and load_buf.data.size == 1) or load_buf.name in load_cache:
                    continue
                load_cache.add(load_buf.name)
                new_group.extend(create_load_or_store(load_buf, True))
        new_group.append(g)
        for out in g.output:
            if out in output_name_map and not isinstance(g, ReduceNode):
                load_cache.add(out)
                new_group.extend(create_load_or_store(output_name_map[out], False))

    block.group = new_group


def analyze_io(
    block: ExecutionBlock,
    global_buffer: GraphIOBuffer,
    c_graph: ConnectionGraph,
    cached_buffer: Dict[str, ComputeBuffer],
):
    # should we keep buffer here?
    # self.cached_buffer = cached_buffer
    inputs = defaultdict(lambda: 0)
    outputs = defaultdict(lambda: 0)
    loads = set()
    const_buffer_set = set([i.name for i in global_buffer.const_buffer])
    external_buffer_out_set = set([i.name for i in global_buffer.var_buffer_out])

    def is_const_input(inp):
        return inp in const_buffer_set or (
            inp in c_graph.egraph.produced_by and c_graph.egraph.produced_by[inp][0].op_type == "Constant"
        )

    for g in block.group:
        for inp_bf in g.input:
            assert isinstance(inp_bf, ComputeBuffer)
            inp = inp_bf.name
            if not is_const_input(inp):
                inputs[inp] += 1
            else:
                loads.add(inp_bf)
        for out_b in g.output:
            assert isinstance(out_b, ComputeBuffer)
            out = out_b.name
            if out not in const_buffer_set:
                outputs[out] = 1
            else:
                raise Exception("const buffer can not be output")
    # self.intermediate_var  = inputs.intersection(outputs)

    for out_name in list(outputs.keys()):
        if out_name in list(inputs.keys()) and out_name not in external_buffer_out_set:
            outputs[out_name] = len(c_graph.egraph.consumed_by[out_name]) - inputs[out_name]
            block.intermediate_var[out_name] = 0
            inputs.pop(out_name)
            assert outputs[out_name] >= 0, "output buffer can not be input"
            if outputs[out_name] == 0:
                outputs.pop(out_name)

    for v in outputs:
        assert v in cached_buffer, "found unhandled output buffer!!!"
        buffer = cached_buffer[v]
        if v not in c_graph.egraph.graph_output_names:
            buffer.attr_cross_loop = True
        block.output.append(buffer)

    for v in inputs:
        assert v in cached_buffer, "found unhandled output buffer!!!"
        buffer = cached_buffer[v]
        if v not in c_graph.egraph.graph_input_names and v not in c_graph.egraph.graph_output_names:
            buffer.attr_cross_loop = True
        block.input.append(buffer)

    for ov in loads:
        v = ov

        if v.name in c_graph.egraph.produced_by:
            pv = c_graph.egraph.produced_by[v][0].name
        else:
            pv = v.name
        tv = c_graph.constant_nodes[pv]
        data = parse_onnx_to_numpyarray(tv)
        assert (data == v.data).all(), "const buffer not matched"
        buffer = v
        cached_buffer[v] = buffer
        block.load[ov] = buffer

    insert_load_and_store(block, global_buffer, c_graph)
    pass


class GraphLowering(NodeVisitor):
    def __init__(self):
        super().__init__()

    def visit(self, node: IRNode, context: HardwareContext, indent: int = 0):
        fn = getattr(self, node.__class__.__name__)
        assert fn is not None, "unimplemented node: %s" % node.__class__.__name__
        return fn(node, context)

    def FunctionNode(self, node: FunctionNode, context: HardwareContext):
        assert len(node.body) == 1, "multiple body not supported in function node"
        shape_var = [i for i in node.body[0].shape if i.is_symbol]
        node.shape_var = list(set(shape_var))
        node.shape_var.sort(key=shape_var.index)

        node.body[0].gen_var(node.const_var)
        node.body[0].analyze_io_connections()

    def ModuleNode(self, node: ModuleNode, context: HardwareContext):
        allow_vectorize = True

        def lower_to_functionNode(
            blocks: List[ExecutionBlock], global_buffer: GraphIOBuffer, func_name: str, allow_vectorize: bool
        ):
            for block in blocks:
                block.lower(self, context)
            schedule = GPUSchedule()
            blocks = schedule.fusion_op(
                blocks,
                set(i.name for i in global_buffer.var_buffer_in),
                set(i.name for i in global_buffer.var_buffer_out),
            )
            blocks = schedule.enable_recompute_for_reduction(blocks)
            blocks = schedule.tile_inner_loop(blocks, context.vec_lanes)
            blocks = schedule.vectoring_inner_loop(blocks, context.vec_lanes)
            blocks = schedule.parallelize_outer_loop(blocks)
            func = FunctionNode(global_buffer.var_buffer_in, global_buffer.var_buffer_out)
            func.body = blocks
            func.const_var = global_buffer.const_buffer
            func.name = func_name
            func.hw_context = context
            func.lower(self, context)
            return func

        for idx, (func_name, model) in enumerate(node.modules.items()):
            plan = ExecutionPrepare(model)
            plan.prepare()
            node_group = plan.create_execution_plan(analyze_io)
            function: FunctionNode = lower_to_functionNode(node_group, plan.external_buffer, func_name, allow_vectorize)
            node.body.append(function)
        node.has_vectorization = allow_vectorize

    def ExecutionBlock(self, node: ExecutionBlock, context: HardwareContext):
        # add Loop()
        node.body = node.build_loop()
