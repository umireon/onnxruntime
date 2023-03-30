# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List, Union

from onnx import NodeProto, ValueInfoProto

from onnxruntime.tools.symbolic_shape_infer import get_elem_type_from_type_proto, get_shape_from_value_info

from ._common import TENSOR_TYPE_TO_NP_TYPE
from ._ir import ComputeBuffer
from ._sympy_utils import sympy_symbol


class DirContext(object):
    def __init__(self, build_dir: Path):
        self.cur_dir = None
        self.build_dir = build_dir

    def __enter__(self):
        self.cur_dir = Path(".").resolve()
        os.chdir(path=self.build_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(path=self.cur_dir)


def convert_onnx_value_to_computebuffer(tensors: Union[ValueInfoProto, List[ValueInfoProto]], prefix=""):
    not_list = False
    if not isinstance(tensors, list):
        not_list = True
        tensors = [tensors]
    bufs = []
    for tensor in tensors:
        dtype = TENSOR_TYPE_TO_NP_TYPE[get_elem_type_from_type_proto(tensor.type)]
        shape = get_shape_from_value_info(tensor)
        shape = sympy_symbol(shape)
        bufs.append(ComputeBuffer(prefix + tensor.name, dtype, shape))
    return bufs if not_list == False else bufs[0]


def gen_unique_name(prefix: str) -> str:
    return prefix + "_" + uuid.uuid4().hex[:8]


def _topological_soft_internal(node, visited, output_consumers, sorted_nodes):
    visited.add(node.name)
    for next_node in output_consumers[node.name]:
        if next_node.name not in visited:
            _topological_soft_internal(next_node, visited, output_consumers, sorted_nodes)

    sorted_nodes.insert(0, node)


def topological_sort(inputs: List[str], nodes: List[NodeProto]) -> List[NodeProto]:
    # inputs constains the graph inputs and initializers. need to add constant nodes.
    const_nodes = []
    non_const_nodes = []
    for node in nodes:
        if node.name == "":
            node.name = gen_unique_name(node.op_type)
        if node.op_type == "Constant":
            inputs.append(node.output[0])
            const_nodes.append(node)
        else:
            non_const_nodes.append(node)
    graph_input_consumers = defaultdict(list)
    output_consumers = defaultdict(list)
    input_set = set(inputs)
    for node in non_const_nodes:
        for input in node.input:
            if input in input_set:
                graph_input_consumers[input].append(node)
        for output in node.output:
            if not output:
                continue
            for consumer in non_const_nodes:
                if output in consumer.input:
                    output_consumers[node.name].append(consumer)

    visited = set()
    sorted_nodes = []
    for input in inputs:
        for node in graph_input_consumers[input]:
            if node.name not in visited:
                _topological_soft_internal(node, visited, output_consumers, sorted_nodes)

    return const_nodes + sorted_nodes
