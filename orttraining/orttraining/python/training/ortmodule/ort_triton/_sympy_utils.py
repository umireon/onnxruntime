# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import re
from typing import List

import sympy
from onnx import NodeProto, TensorProto
from sympy.core.logic import fuzzy_and, fuzzy_or  # type: ignore[import]


def sympy_symbol(name):
    if isinstance(name, int):
        return sympy.Integer(name)
    if isinstance(name, list):
        return [sympy_symbol(x) for x in name]
    if isinstance(name, str):
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    return sympy.Symbol(name, integer=True, positive=True)


def sympy_dot(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


class FloorDiv(sympy.Function):
    """
    We maintain this so that:
    1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
    2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)
    """

    nargs = (2,)
    precedence = 50  # precedence of mul  # noqa: F811

    # Default return type for SymPy assumptions.
    # https://docs.sympy.org/latest/guides/assumptions.html#implementing-assumptions-handlers
    is_real = True

    @property
    def base(self):
        return self.args[0]

    @property
    def divisor(self):
        return self.args[1]

    def _sympystr(self, printer):
        base = printer.parenthesize(self.base, self.precedence)
        divisor = printer.parenthesize(self.divisor, self.precedence)
        return f"{base}//{divisor}"

    # SymPy assumptions based on argument types.
    def _eval_is_real(self):
        return fuzzy_or([self.base.is_real, self.divisor.is_real])

    def _eval_is_integer(self):
        return fuzzy_and([self.base.is_integer, self.divisor.is_integer])

    # Automatic evaluation.
    # https://docs.sympy.org/latest/guides/custom-functions.html#best-practices-for-eval
    @classmethod
    def eval(cls, base, divisor):
        def check_supported_type(x):
            if (x.is_integer is False and x.is_real is False and x.is_complex) or x.is_Boolean:
                raise TypeError(
                    f"unsupported operand type(s) for //: "
                    f"'{type(base).__name__}' and '{type(divisor).__name__}'"
                    f", expected integer or real"
                )

        check_supported_type(base)
        check_supported_type(divisor)

        # We don't provide the same error message as in Python because SymPy
        # makes it difficult to check the types.
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")

        if base.is_zero:
            return sympy.S.Zero
        if base.is_integer and divisor == 1:
            return base
        if base.is_real and divisor == 1:
            return sympy.floor(base)
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return base // divisor
        if isinstance(base, (sympy.Integer, sympy.Float)) and isinstance(divisor, (sympy.Integer, sympy.Float)):
            return sympy.floor(base / divisor)
        if isinstance(base, FloorDiv):
            return FloorDiv(base.args[0], base.args[1] * divisor)

        if isinstance(base, sympy.Add):
            for a in base.args:
                gcd = sympy.gcd(a, divisor)
                if gcd == divisor:
                    return FloorDiv(base - a, divisor) + a / gcd

        gcd = sympy.gcd(base, divisor)
        if gcd != 1:
            return FloorDiv(sympy.simplify(base / gcd), sympy.simplify(divisor / gcd))


class TypeAndShape(object):
    def __init__(self, dtype, shape):
        self._dtype = dtype
        self._shape = shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape


def _infer_elementwise_shape(input_infos: List[TypeAndShape]):
    max_len = max([len(input_info.shape) for input_info in input_infos])
    output_shape = [sympy_symbol(1)] * max_len
    for input_info in input_infos:
        offset = max_len - len(input_info.shape)
        for i in range(len(input_info.shape)):
            if not input_info.shape[i].is_number or input_info.shape[i] != 1:
                output_shape[i + offset] = input_info.shape[i]
    return output_shape


def _infer_elementwise(node: NodeProto, input_infos: List[TypeAndShape]):
    return TypeAndShape(input_infos[0].dtype, _infer_elementwise_shape(input_infos))


def _infer_where(node: NodeProto, input_infos: List[TypeAndShape]):
    return TypeAndShape(input_infos[1].dtype, _infer_elementwise_shape(input_infos))


def _infer_reduction(node: NodeProto, input_infos: List[TypeAndShape]):
    # Support reduction on the last axis only for now.
    return TypeAndShape(input_infos[0].dtype, input_infos[0].shape[:-1] + [sympy_symbol(1)])


def _infer_unary(node: NodeProto, input_infos: List[TypeAndShape]):
    return input_infos[0]


def _infer_cast(node: NodeProto, input_infos: List[TypeAndShape]):
    dtype = TensorProto.UNDEFINED
    for attr in node.attribute:
        if attr.name == "to":
            dtype = attr.i
            break
    assert dtype != TensorProto.UNDEFINED
    return TypeAndShape(dtype, input_infos[0].shape)


def _infer_dropout(node: NodeProto, input_infos: List[TypeAndShape]):
    return [input_infos[0], TypeAndShape(TensorProto.BOOL, input_infos[0].shape)]


class TypeAndShapeInfer(object):
    _INFER_FUNC_MAP = {
        "Add": _infer_elementwise,
        "Sub": _infer_elementwise,
        "Mul": _infer_elementwise,
        "Div": _infer_elementwise,
        "Pow": _infer_elementwise,
        "Sqrt": _infer_elementwise,
        "Exp": _infer_elementwise,
        "Where": _infer_where,
        "Rsqrt": _infer_elementwise,
        "Cast": _infer_cast,
        "Dropout": _infer_dropout,
        "DropoutGrad": _infer_unary,
        "Identity": _infer_unary,
        "ReduceSum": _infer_reduction,
        "ReduceMax": _infer_reduction,
        "ReduceMin": _infer_reduction,
    }

    @classmethod
    def infer(cls, node: NodeProto, input_infos: List[TypeAndShape]):
        if node.op_type not in cls._INFER_FUNC_MAP:
            raise NotImplementedError(f"Unsupported op type: {node.op_type}")
        return cls._INFER_FUNC_MAP[node.op_type](node, input_infos)
