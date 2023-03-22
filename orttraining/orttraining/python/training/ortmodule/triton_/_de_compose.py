# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
from onnx import AttributeProto, NodeProto, TensorProto, helper


def attr_parse(attr, ind=None):
    if attr.type == AttributeProto.INTS:
        v = attr.ints
        if ind is not None:
            return v[ind]
        return v
    elif attr.type == AttributeProto.INT:
        return attr.i
    elif attr.type == AttributeProto.FLOATS:
        v = attr.floats
        if ind is not None:
            return v[ind]
    elif attr.type == AttributeProto.FLOAT:
        return attr.f
    else:
        raise NotImplementedError("Not implemented for attr type: {}".format(attr.type))


'''
decompose a complicated op into a series of simple ops
'''
class DecomposeDispatch(object):
    def __init__(self):
        super().__init__()
        self.count = 0

    def get_unique_var_name(self, prefix):
        self.count += 1
        return prefix + str(self.count)

    def new_node(self, op_type, inputs, name, output=None, **kwargs):
        name = self.get_unique_var_name(name)
        if output is None:
            output = [self.get_unique_var_name(f"{op_type}_out")]
        return output[0], helper.make_node(op_type, inputs, output, name, **kwargs)

    def __call__(self, node: NodeProto, **kwargs):
        if not hasattr(self, node.op_type):
            raise NotImplementedError("Not implemented for op type: {}".format(node.op_type))
        return getattr(self, node.op_type)(node, **kwargs)

    def __contains__(self, node_or_op_type):
        op_type = node_or_op_type.op_type if isinstance(node_or_op_type, NodeProto) else node_or_op_type
        return isinstance(op_type, str) and hasattr(self, op_type)

    def LayerNormalization(self, node: NodeProto, **kwargs):
        shape_info_map = kwargs["shape_info_map"]
        input_type = shape_info_map[node.input[0]][0]
        is_half = input_type == TensorProto.FLOAT16 or input_type == TensorProto.BFLOAT16
        output = node.output[0]
        mean_output = node.output[1] if len(node.output) > 1 else None
        inv_std_dev_output = node.output[2] if len(node.output) > 2 else None
        attr = {i.name: i.i or i.f for i in node.attribute}

        e1_12_v = helper.make_tensor(
            name="f",
            data_type=TensorProto.FLOAT if is_half else input_type,
            dims=(1,),
            vals=np.array([attr["epsilon"]]),
        )

        e1_12_out, e1_12_node = self.new_node("Constant", [], f"e1_12_{node.name}", value=e1_12_v)
        input = node.input[0]
        cast_node = None
        if is_half:
            input, cast_node = self.new_node("Cast", [input], f"{node.name}_cast", to=TensorProto.FLOAT)
        reducemean_out, reducemean_node = self.new_node(
            "ReduceMean", [input], f"{node.name}_reducemean", output=[mean_output], axes=[-1]
        )
        sub_out, sub_node = self.new_node("Sub", [input, reducemean_out], f"{node.name}_sub")
        mul_out0, mul_node0 = self.new_node("Mul", [sub_out, sub_out], f"{node.name}_exp")
        last_dim = shape_info_map[node.input[0]][1][-1]
        reducemean_out1, reducemean_node1 = self.new_node(
            "ReduceMean", [mul_out0], f"{node.name}_reducemean1", axes=[-1], last_dim=[last_dim]
        )
        add_out, add_node = self.new_node("Add", [reducemean_out1, e1_12_out], f"{node.name}_add")
        rsqrt_out, rsqrt_node = self.new_node("Rsqrt", [add_out], f"{node.name}_rsqrt", output=[inv_std_dev_output])
        mul_out, mul_node = self.new_node("Mul", [sub_out, rsqrt_out], f"{node.name}_mul")
        cast_node1 = None
        if is_half:
            mul_out, cast_node1 = self.new_node("Cast", [mul_out], f"{node.name}_cast1", to=input_type)
        mul_out1, mul_node1 = self.new_node("Mul", [node.input[1], mul_out], f"{node.name}_mul1")
        _, add_node1 = self.new_node("Add", [node.input[2], mul_out1], f"{node.name}_add1", output=[output])

        cast_nodes = [cast_node, cast_node1] if is_half else []
        return cast_nodes + [
            e1_12_node,
            reducemean_node,
            sub_node,
            mul_node0,
            reducemean_node1,
            add_node,
            rsqrt_node,
            mul_node,
            mul_node1,
            add_node1,
        ]

    def Softmax(self, node: NodeProto, **kwargs):
        axis = node.attribute[0].i
        name = node.name
        axes_out = self.get_unique_var_name("axes_in_softmax")
        axes_v = helper.make_tensor(
            name="axes",
            data_type=TensorProto.INT64,
            dims=(1,),
            vals=np.array([axis]),
        )

        axes_out, axes_node = self.new_node("Constant", [], f"axes_in_reduceMean_{node.name}", value=axes_v)

        max_out, max_node = self.new_node("ReduceMax", [node.input[0]], f"{name}_max", axes=[axis])
        sub_out, sub_node = self.new_node(
            "Sub",
            [node.input[0], max_out],
            f"{name}_sub",
        )
        exp_out, exp_node = self.new_node(
            "Exp",
            [sub_out],
            f"{name}_exp",
        )
        sum_out, sum_node = self.new_node("ReduceSum", [exp_out, axes_out], f"{name}_sum")

        _, div_node = self.new_node("Div", [exp_out, sum_out], f"sum/decomposed_from_{node.name}", output=node.output)

        return [axes_node, max_node, sub_node, exp_node, sum_node, div_node]

    def ReduceMean(self, node: NodeProto, **kwargs):
        shape_info_map = kwargs["shape_info_map"]
        axes_out = self.get_unique_var_name("axes_in_reduceMean")

        attr = {i.name: attr_parse(i, 0) for i in node.attribute}
        axes_v = helper.make_tensor(
            name="axes",
            data_type=TensorProto.INT64,
            dims=(1,),
            vals=np.array([attr["axes"]]),
        )

        axes_out, axes_node = self.new_node("Constant", [], f"axes_in_reduceMean_{node.name}", value=axes_v)
        sum_out, sum_node = self.new_node("ReduceSum", [node.input[0], axes_out], f"sum/decomposed_from_{node.name}")
        last_dim_v = attr["last_dim"] if "last_dim" in attr else shape_info_map[node.input[0]][1][-1]
        v = helper.make_tensor(
            name="last_dim", dims=(), data_type=TensorProto.FLOAT, vals=np.array([last_dim_v], dtype=np.float32)
        )
        const_v_neg1, n_elem_node = self.new_node("Constant", [], f"shape[-1]/decomposed_from_{node.name}", value=v)
        _, div_node = self.new_node(
            "Div", [sum_out, const_v_neg1], f"Div/decomposed_from_{node.name}", output=node.output
        )
        return [axes_node, sum_node, n_elem_node, div_node]
