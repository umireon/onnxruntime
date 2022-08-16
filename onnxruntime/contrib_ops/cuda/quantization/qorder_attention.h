// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/tensor/quantize_linear.cuh"
#include "contrib_ops/cpu/bert/attention_base.h"


namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class QOrderedAttention final : public CudaKernel, public AttentionBase {
 public:
  QOrderedAttention(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

 private:
  int64_t unidirectional_;
  int64_t qkv_hidden_sizes_;
  int order_input_;
  int order_weight_;
  int order_bias_;
  int order_output_;
  int qkv_weight_const_count_, scale_qkv_weight_const_count_, qkv_bias_const_cout_;
  BufferUniquePtr merged_qkv_weight_;
  TensorShape single_weight_shape_;
  float const_sacle_input_;
  float const_scale_qkv_gemm_[3];
  BufferUniquePtr merged_qkv_alpha_;
  BufferUniquePtr merged_qkv_bias_;
  BufferUniquePtr softmax_lookup_;

 private:
  Status PutIntoMergedWeight(const Tensor& tensor, AllocatorPtr alloc, int qkv_index);
  Status PutIntoMergedWeightScale(const Tensor& tensor, AllocatorPtr alloc, int qkv_index);
  Status PutIntoMergedBias(const Tensor& tensor, AllocatorPtr alloc, int qkv_index);
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime