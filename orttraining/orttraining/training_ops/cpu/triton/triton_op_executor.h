// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/core/framework/torch/torch_proxy.h"

namespace onnxruntime {
namespace contrib {

using PythonObjectPtr = language_interop_ops::torch::PythonObjectPtr;
constexpr auto PythonObjectDeleter = language_interop_ops::torch::PythonObjectDeleter;

class TritonOpExecutor final {
 public:
  static TritonOpExecutor& Instance() {
    static TritonOpExecutor instance;
    return instance;
  }

  void RegisterTritonOpExecutor(PyObject* obj);

  void Initialize(PyObject* obj) {
    ORT_ENFORCE(executor_.get() == nullptr && obj != nullptr);
    Py_INCREF(obj);
    PythonObjectPtr ptr(obj, PythonObjectDeleter);
    executor_ = std::move(ptr);
  }

  bool IsInitialized() { return executor_.get() != nullptr; }

  PyObject* GetExecutor() {
    ORT_ENFORCE(executor_.get() != nullptr);
    return executor_.get();
  }

 private:
  PythonObjectPtr executor_;
};

}  // namespace contrib
}  // namespace onnxruntime
