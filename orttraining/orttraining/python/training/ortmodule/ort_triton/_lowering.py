# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from collections import OrderedDict, defaultdict, deque
from typing import Dict, Iterable, List, Tuple

from onnx import NodeProto

from ._common import TENSOR_TYPE_TO_NP_TYPE, GraphIOBuffer, HardwareContext, NodeVisitor, parse_onnx_to_numpyarray
from ._ir import (
    ComputeBuffer,
    ComputeNode,
    ExecutionBlock,
    FunctionNode,
    IRNode,
    MaskLoadNode,
    MaskStoreNode,
    ModuleNode,
    ReduceNode,
)
from ._op_config import is_elementwise_node, is_reduction_node
from ._scheduling import GPUSchedule
from ._sorted_graph import SortedGraph
from ._sympy_utils import TypeAndShape
from ._utils import gen_unique_name


def _group_nodes(sorted_graph: SortedGraph, io_buffer: GraphIOBuffer) -> Iterable[List[NodeProto]]:
    loop_range = 0
    for out in io_buffer.var_buffer_out:
        loop_range = max(loop_range, out.shape[-1])
    if loop_range.is_number and loop_range > 1024:
        before_fusion_groups = deque()
        after_fusion_groups = deque()
        for node in sorted_graph.sorted_nodes:
            before_fusion_groups.append([node])
        while len(before_fusion_groups) > 1:
            group1 = before_fusion_groups.popleft()
            group2 = before_fusion_groups.popleft()
            if is_elementwise_node(group1[-1]):
                group1.extend(group2)
                before_fusion_groups.appendleft(group1)
            else:
                after_fusion_groups.append(group1)
                before_fusion_groups.appendleft(group2)
        grouped_nodes = after_fusion_groups
    else:
        grouped_nodes = [sorted_graph.sorted_nodes]
    return grouped_nodes


def _extract_io(sorted_graph: SortedGraph) -> GraphIOBuffer:
    io_buffer = GraphIOBuffer()
    for input in sorted_graph.graph.input:
        input_buffer = ComputeBuffer(
            input.name,
            TENSOR_TYPE_TO_NP_TYPE[sorted_graph.node_arg_infos[input.name].dtype],
            sorted_graph.node_arg_infos[input.name].shape,
        )
        io_buffer.var_buffer_in.append(input_buffer)
    for output in sorted_graph.graph.output:
        output_buffer = ComputeBuffer(
            output.name,
            TENSOR_TYPE_TO_NP_TYPE[sorted_graph.node_arg_infos[output.name].dtype],
            sorted_graph.node_arg_infos[output.name].shape,
        )
        io_buffer.var_buffer_out.append(output_buffer)
    for initializer in sorted_graph.graph.initializer:
        data = parse_onnx_to_numpyarray(initializer)
        initializer_buffer = ComputeBuffer(initializer.name, data=data)
        io_buffer.const_buffer.append(initializer_buffer)
    for const_node in sorted_graph.const_nodes:
        data = parse_onnx_to_numpyarray(const_node)
        const_buffer = ComputeBuffer(const_node.output[0], data=data)
        io_buffer.const_buffer.append(const_buffer)
    return io_buffer


def _process_io(
    node: NodeProto, cache_buffers: Dict[str, ComputeBuffer], node_arg_infos: Dict[str, TypeAndShape]
) -> Tuple[List[ComputeBuffer], List[ComputeBuffer]]:
    input_buffer = []
    for input in node.input:
        if input in cache_buffers:
            input_buffer.append(cache_buffers[input])
        else:
            input_buffer.append(
                ComputeBuffer(
                    name=input,
                    dtype=TENSOR_TYPE_TO_NP_TYPE[node_arg_infos[input].dtype],
                    shape=node_arg_infos[input].shape,
                )
            )
            cache_buffers[input] = input_buffer[-1]
    output_buffer = []
    for output in node.output:
        output_buffer.append(
            ComputeBuffer(
                name=output,
                dtype=TENSOR_TYPE_TO_NP_TYPE[node_arg_infos[output].dtype],
                shape=node_arg_infos[output].shape,
            )
        )
        cache_buffers[output] = output_buffer[-1]
    return input_buffer, output_buffer


def _create_load_or_store(buffer: ComputeBuffer, is_load: bool):
    mask_id_name = gen_unique_name("triton_mask")
    mask_buffer = ComputeBuffer(mask_id_name, shape=buffer.shape[-1:])
    return [MaskLoadNode(buffer, mask_buffer)] if is_load else [MaskStoreNode(buffer, mask_buffer)]


def _insert_load_and_store(block: ExecutionBlock, reduction_outputs: set):
    input_name_map = [input.name for input in block.input]
    output_name_map = [output.name for output in block.output]
    new_group = []
    # avoid duplicate load
    load_cache = set()
    for node in block.group:
        for input in node.input:
            if (input.name in block.load or input.name in input_name_map) and input.name not in reduction_outputs:
                if (input.data is not None and input.data.size == 1) or input.name in load_cache:
                    continue
                load_cache.add(input.name)
                new_group.extend(_create_load_or_store(input, True))
        new_group.append(node)
        for output in node.output:
            if output.name in output_name_map and output.name not in reduction_outputs:
                load_cache.add(output.name)
                new_group.extend(_create_load_or_store(output, False))

    block.group = new_group


def _analyze_io(
    block: ExecutionBlock,
    io_buffer: GraphIOBuffer,
    consumer_counts: Dict[str, int],
    reduction_outputs: set,
    cached_buffer: Dict[str, ComputeBuffer],
):
    # should we keep buffer here?
    # self.cached_buffer = cached_buffer
    inputs = defaultdict(lambda: 0)
    outputs = set()
    loads = set()
    const_names = set([buffer.name for buffer in io_buffer.const_buffer])
    graph_output_names = set([buffer.name for buffer in io_buffer.var_buffer_out])
    graph_input_names = set(buffer.name for buffer in io_buffer.var_buffer_in)

    for node in block.group:
        for input_buffer in node.input:
            assert isinstance(input_buffer, ComputeBuffer)
            if input_buffer.name not in const_names:
                inputs[input_buffer.name] += 1
            else:
                loads.add(input_buffer)

    outputs = set()
    for node in block.group:
        for output_buffer in node.output:
            assert isinstance(output_buffer, ComputeBuffer)
            name = output_buffer.name
            outputs.add(name)
            if name not in graph_output_names:
                block.intermediate_var[name] = 0
                if consumer_counts[name] > inputs[name]:
                    output_buffer.attr_cross_loop = True
                    block.output.append(output_buffer)
            else:
                block.output.append(output_buffer)
            if name in inputs:
                inputs[name] = 0

    for input in inputs:
        assert inputs[input] >= 0 and input in cached_buffer
        if inputs[input] > 0:
            buffer = cached_buffer[input]
            if input not in graph_input_names and input not in outputs:
                buffer.attr_cross_loop = True
            block.input.append(buffer)

    for const_buffer in loads:
        block.load[const_buffer.name] = const_buffer

    _insert_load_and_store(block, reduction_outputs)


def _lower(sorted_graph: SortedGraph) -> Tuple[GraphIOBuffer, List[ExecutionBlock]]:
    io_buffer = _extract_io(sorted_graph)
    grouped_nodes = _group_nodes(sorted_graph, io_buffer)
    cache_buffers: Dict[str, ComputeBuffer] = OrderedDict()
    for input_buffer in io_buffer.var_buffer_in:
        cache_buffers[input_buffer.name] = input_buffer
    for output_buffer in io_buffer.var_buffer_out:
        cache_buffers[output_buffer.name] = output_buffer
    for const_buffer in io_buffer.const_buffer:
        cache_buffers[const_buffer.name] = const_buffer
    fusion_blocks = []
    for group in grouped_nodes:
        exe_block = ExecutionBlock(sorted_graph.node_arg_infos[group[-1].output[0]])
        fusion_blocks.append(exe_block)
        new_group = []
        for node in group:
            in_buffers, out_buffers = _process_io(node, cache_buffers, sorted_graph.node_arg_infos)
            ir_node = ComputeNode(node.op_type, in_buffers, out_buffers, node.name, node.attribute)
            for input_buffer in in_buffers:
                input_buffer.successor.append(ir_node)
            for output_buffer in out_buffers:
                output_buffer.predecessor = ir_node
            if is_reduction_node(node):
                exe_block.has_reduce = True
                for output in node.output:
                    output_buffer = out_buffers[out_buffers.index(output)]
                    exe_block.forward_var_map_list[0][output] = output_buffer
                new_group.append(ReduceNode(ir_node))
            else:
                new_group.append(ir_node)
        exe_block.fused_groups.append(new_group)
        exe_block.group = new_group

    consumer_counts = {}
    reduction_outputs = set()
    for node in sorted_graph.sorted_nodes:
        for input in node.input:
            if input not in consumer_counts:
                consumer_counts[input] = 0
            consumer_counts[input] += 1
        if is_reduction_node(node):
            reduction_outputs.add(node.output[0])
    for block in fusion_blocks:
        _analyze_io(block, io_buffer, consumer_counts, reduction_outputs, cache_buffers)

    return io_buffer, fusion_blocks


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
        def _lower_to_function_node(
            blocks: List[ExecutionBlock], global_buffer: GraphIOBuffer, func_name: str
        ) -> FunctionNode:
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

        for func_name, sorted_graph in node.modules.items():
            io_buffer, blocks = _lower(sorted_graph)
            function: FunctionNode = _lower_to_function_node(blocks, io_buffer, func_name)
            node.body.append(function)
        node.has_vectorization = True

    def ExecutionBlock(self, node: ExecutionBlock, context: HardwareContext):
        node.body = node.build_loop()
