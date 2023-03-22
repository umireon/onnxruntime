# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from pathlib import Path
from typing import List, Union

import onnx

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


def convert_onnx_value_to_computebuffer(tensors: Union[onnx.ValueInfoProto, List[onnx.ValueInfoProto]], prefix=""):
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
