# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import copy
from typing import List

import numpy as np
import onnx
from onnx import ModelProto, numpy_helper
from sympy import Symbol

from ._de_compose import DecomposeDispatch
from ._sympy_utils import TypeAndShape, TypeAndShapeInfer, sympy_symbol
from ._utils import topological_sort


class SortedGraph(object):
    def __init__(self, model: ModelProto, input_shapes: List[List[Symbol]]):
        self.model = model
        self.graph = model.graph
        self.input_shapes = input_shapes
        self.sorted_nodes = topological_sort(
            [input.name for input in self.graph.input] + [initializer.name for initializer in self.graph.initializer],
            self.graph.node,
        )
        self.node_arg_infos = {}

        for idx, input in enumerate(self.graph.input):
            self.node_arg_infos[input.name] = TypeAndShape(input.type.tensor_type.elem_type, self.input_shapes[idx])
        for initializer in self.graph.initializer:
            self.node_arg_infos[initializer.name] = TypeAndShape(
                initializer.data_type,
                sympy_symbol(list(numpy_helper.to_array(initializer).shape)),
            )

        self._decompose()
        self._recompose()

        initializers = {}
        for initializer in self.graph.initializer:
            initializers[initializer.name] = initializer
        self.sorted_initializers = []
        for node in self.sorted_nodes:
            for input in node.input:
                if input in initializers:
                    self.sorted_initializers.append(initializers[input])
                    initializers.pop(input)

        self.const_nodes = [node for node in self.sorted_nodes if node.op_type == "Constant"]
        self.sorted_nodes = [node for node in self.sorted_nodes if node.op_type != "Constant"]

    def __str__(self):
        graph_inputs = []
        name_map = {}
        for idx, input in enumerate(self.graph.input):
            shape_str = str(self.input_shapes[idx]).replace(" ", "")
            graph_inputs.append(f"({str(input.type.tensor_type.elem_type)},{shape_str})")
            name_map[input.name] = f"i{idx}"
        graph_inputs_str = ",".join(graph_inputs)
        constants = []
        for idx, initializer in enumerate(self.sorted_initializers):
            data_str = (
                np.array2string(numpy_helper.to_array(initializer), separator=",").replace("\n", "").replace(" ", "")
            )
            constants.append(f"({initializer.data_type},{data_str})")
            name_map[initializer.name] = f"c{idx}"
        for idx, node in enumerate(self.const_nodes):
            data_str = (
                np.array2string(numpy_helper.to_array(node.attribute[0].t), separator=",")
                .replace("\n", "")
                .replace(" ", "")
            )
            constants.append(f"({node.attribute[0].t.data_type},{data_str})")
            name_map[node.output[0]] = f"c{idx + len(self.sorted_initializers)}"
        constants_str = ",".join(constants)
        for idx, output in enumerate(self.graph.output):
            name_map[output.name] = f"o{idx}"
        nodes = []
        for node_idx, node in enumerate(self.sorted_nodes):
            inputs = []
            for input in node.input:
                inputs.append(name_map[input] if input in name_map else input)
            inputs_str = ",".join(inputs)
            outputs = []
            for idx, output in enumerate(node.output):
                if output in name_map:
                    outputs.append(name_map[output])
                else:
                    name_map[output] = f"t{node_idx}_{idx}"
                    outputs.append(name_map[output])
            outputs_str = ",".join(outputs)
            attributes = []
            for attr in node.attribute:
                fields = [str(f[1]) for f in attr.ListFields()]
                attributes.append(f"{fields[0]}:{fields[2]}={fields[1]}")
            attributes_str = ",".join(attributes)
            nodes.append(f"{node.op_type}[{attributes_str}]({inputs_str})->({outputs_str})")
        nodes_str = ",".join(nodes)
        return f"{graph_inputs_str}|{str(len(self.graph.output))}|{constants_str}|{nodes_str}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def _decompose(self):
        dispatch = DecomposeDispatch()
        pos = 0
        while pos < len(self.sorted_nodes):
            node = self.sorted_nodes[pos]
            if node.op_type in dispatch:
                new_nodes = dispatch(node, node_arg_infos=self.node_arg_infos)
                new_nodes = topological_sort(node.input, new_nodes)
                self.sorted_nodes[pos : pos + 1] = new_nodes
                continue
            if node.op_type == "Constant":
                self.node_arg_infos[node.output[0]] = TypeAndShape(
                    node.attribute[0].t.data_type,
                    sympy_symbol(list(numpy_helper.to_array(node.attribute[0].t).shape)),
                )
            else:
                input_infos = []
                for input in node.input:
                    input_infos.append(self.node_arg_infos[input])
                output_infos = TypeAndShapeInfer.infer(node, input_infos)
                if len(node.output) == 1:
                    self.node_arg_infos[node.output[0]] = output_infos
                else:
                    for idx, output in enumerate(node.output):
                        self.node_arg_infos[output] = output_infos[idx]
            pos += 1

    def _recompose(self):
        pass

    def save_onnx(self, file_path_prefix):
        onnx.save(self.model, file_path_prefix + "_original.onnx")
        processed_model = copy.deepcopy(self.model)
        processed_model.graph.ClearField("node")
        processed_model.graph.node.extend(self.const_nodes)
        processed_model.graph.node.extend(self.sorted_nodes)
        onnx.save(processed_model, file_path_prefix + "_processed.onnx")
