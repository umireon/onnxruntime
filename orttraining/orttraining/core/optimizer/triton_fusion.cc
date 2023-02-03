// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CUDA

#include <fstream>

#include "orttraining/core/optimizer/triton_fusion.h"

#include "core/framework/compute_capability.h"
#include "core/graph/model.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/providers/partitioning_utils.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

namespace {

using OpSetVersionList = std::initializer_list<OperatorSetVersion>;
using SizeTypeVec = InlinedVector<size_t>;
using NodeVec = InlinedVector<Node*>;
using NodeArgVec = InlinedVector<NodeArg*>;
using ConstNodeArgVec = InlinedVector<const NodeArg*>;
using NodeArgSet = InlinedHashSet<NodeArg*>;
using IsSupportedFunc = std::function<bool(const Graph&, const Node&)>;

struct OpInfo {
  OpInfo(const char* op_type, const OpSetVersionList& supported_versions, const char* domain, const bool is_no_op,
         IsSupportedFunc is_supported_func)
      : op_type_(op_type),
        supported_versions_(supported_versions),
        domain_(domain),
        is_no_op_(is_no_op),
        is_supported_func_(is_supported_func){};

  std::string op_type_;
  OpSetVersionList supported_versions_;
  std::string domain_;
  bool is_no_op_;
  IsSupportedFunc is_supported_func_;
};

const OpSetVersionList OpSetV1 = {1};
const OpSetVersionList OpSetV13_14 = {13, 14};
const OpSetVersionList OpSetV9 = {9};
const OpSetVersionList OpSetV13 = {13};
IsSupportedFunc default_is_supported = [](const Graph&, const Node&) { return true; };
const InlinedHashMap<std::string, OpInfo> kSupportedOps{
    {"Add", OpInfo("Add", OpSetV13_14, kOnnxDomain, false, default_is_supported)},
    {"Sub", OpInfo("Sub", OpSetV13_14, kOnnxDomain, false, default_is_supported)},
    {"Mul", OpInfo("Mul", OpSetV13_14, kOnnxDomain, false, default_is_supported)},
    {"Div", OpInfo("Div", OpSetV13_14, kOnnxDomain, false, default_is_supported)},
    {"Where", OpInfo("Where", OpSetV9, kOnnxDomain, false, default_is_supported)},
    // {"Cast", OpInfo("Cast", OpSetV13, kOnnxDomain, false, default_is_supported)},
    // {"Reshape", OpInfo("Reshape", OpSetV13_14, kOnnxDomain, true, default_is_supported)},
    // {"Squeeze", OpInfo("Squeeze", OpSetV13, kOnnxDomain, true, default_is_supported)},
    // {"Unsqueeze", OpInfo("Unsqueeze", OpSetV13, kOnnxDomain, true, default_is_supported)},
    {"Softmax", OpInfo("Softmax", OpSetV13, kOnnxDomain, false, default_is_supported)},
    {"SoftmaxGrad_13", OpInfo("SoftmaxGrad_13", OpSetV1, kMSDomain, false, default_is_supported)},
};

struct Partition {
  NodeVec nodes;
  NodeArgSet outputs;
  NodeArgSet dependencies;
  size_t output_ref_count;

  void MergeFrom(const Partition& other) {
    nodes.insert(nodes.end(), other.nodes.begin(), other.nodes.end());
    outputs.insert(other.outputs.begin(), other.outputs.end());
    dependencies.insert(other.dependencies.begin(), other.dependencies.end());
    output_ref_count += other.output_ref_count;
  }

  bool IsValid() const {
    size_t count = 0;
    for (const auto& node : nodes) {
      if (!kSupportedOps.at(node->OpType()).is_no_op_) {
        ++count;
        if (count >= 2) return true;
      }
    }
    return false;
  }
};

}  // namespace

bool TritonFusion::IsSupportedNode(const Graph& graph, const Node& node) const {
  const auto& op_type = node.OpType();
  if (!graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
      kSupportedOps.find(op_type) == kSupportedOps.end()) {
    return false;
  }

  const auto& op_info = kSupportedOps.at(op_type);
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, op_info.op_type_, op_info.supported_versions_,
                                                        op_info.domain_) &&
         op_info.is_supported_func_(graph, node);
}

Status TritonFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  size_t global_id = 0;
  InlinedHashMap<size_t, Partition> partitions;
  InlinedHashMap<size_t, Partition> partitions_to_fuse;
  InlinedHashMap<NodeArg*, size_t> active_outputs;
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (!p_node) continue;
    auto& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    bool is_supported = IsSupportedNode(graph, node);
    SizeTypeVec partitions_to_merge;
    for (auto& pair : partitions) {
      auto& partition = pair.second;
      bool connect_to_output = false;
      bool connect_to_dependency = false;
      for (auto& input : node.MutableInputDefs()) {
        if (partition.outputs.find(input) != partition.outputs.end()) {
          partition.output_ref_count--;
          connect_to_output = true;
        }
        if (partition.dependencies.find(input) != partition.dependencies.end()) {
          connect_to_dependency = true;
        }
      }
      if (is_supported && connect_to_output && !connect_to_dependency) {
        partitions_to_merge.emplace_back(pair.first);
      } else if (connect_to_output || connect_to_dependency) {
        for (auto& output : node.MutableOutputDefs()) {
          partition.dependencies.emplace(output);
        }
      }
    }

    if (!partitions_to_merge.empty()) {
      std::sort(partitions_to_merge.begin(), partitions_to_merge.end());
      Partition& dst = partitions.at(partitions_to_merge[0]);
      for (size_t i = partitions_to_merge.size() - 1; i > 0; --i) {
        dst.MergeFrom(partitions.at(partitions_to_merge[i]));
        partitions.erase(partitions_to_merge[i]);
      }

      dst.nodes.emplace_back(&node);
      for (auto& output : node.MutableOutputDefs()) {
        dst.outputs.emplace(output);
      }
      dst.output_ref_count += node.GetOutputEdgesCount();
    } else if (is_supported) {
      Partition partition;
      partition.nodes.emplace_back(&node);
      for (auto& node_def : node.MutableOutputDefs()) {
        partition.outputs.emplace(node_def);
      }
      partition.output_ref_count = node.GetOutputEdgesCount();
      partitions.emplace(global_id++, partition);
    }

    SizeTypeVec partitions_to_erase;
    for (auto& pair : partitions) {
      if (pair.second.output_ref_count == 0) {
        if (pair.second.IsValid()) {
          pair.second.outputs.clear();
          pair.second.dependencies.clear();
          partitions_to_fuse.emplace(pair);
        }
        partitions_to_erase.emplace_back(pair.first);
      }
    }

    for (auto& id : partitions_to_erase) {
      partitions.erase(id);
    }

    for (auto& input : node.MutableInputDefs()) {
      if (active_outputs.find(input) != active_outputs.end()) {
        active_outputs.at(input)--;
        if (active_outputs.at(input) == 0) {
          active_outputs.erase(input);
          for (auto& pair : partitions) {
            pair.second.dependencies.erase(input);
          }
        }
      }
    }

    for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
      NodeArg* output = node.MutableOutputDefs()[it->GetSrcArgIndex()];
      if (active_outputs.find(output) == active_outputs.end()) {
        active_outputs.emplace(output, 1);
      } else {
        active_outputs.at(output)++;
      }
    }
  }

  SizeTypeVec partition_ids;
  for (auto& pair : partitions_to_fuse) {
    partition_ids.emplace_back(pair.first);
  }
  std::sort(partition_ids.begin(), partition_ids.end());

  for (auto& id : partition_ids) {
    auto& partition = partitions_to_fuse.at(id);

    Model sub_model("test", false, logger);
    Graph& sub_graph = sub_model.MainGraph();

    NodeArgVec graph_inputs;
    NodeArgSet initializers;
    ConstNodeArgVec graph_const_inputs;
    InlinedHashMap<NodeArg*, size_t> output_ref_counts;
    for (auto& p_node : partition.nodes) {
      auto& node = *p_node;
      sub_graph.AddNode(node);
      for (auto& input : node.MutableInputDefs()) {
        if (graph_utils::IsInitializer(graph, input->Name(), true)) {
          if (initializers.find(input) == initializers.end()) {
            const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
            if (graph.GetInitializedTensor(input->Name(), tensor) && tensor) {
              initializers.emplace(input);
              sub_graph.AddInitializedTensor(*tensor);
              continue;
            }
          }
        }

        if (output_ref_counts.find(input) != output_ref_counts.end()) {
          output_ref_counts.at(input)--;
          if (output_ref_counts.at(input) == 0) {
            output_ref_counts.erase(input);
          }
        } else {
          graph_inputs.emplace_back(input);
          graph_const_inputs.emplace_back(input);
        }
      }

      for (auto it = p_node->OutputEdgesBegin(), end = p_node->OutputEdgesEnd(); it != end; ++it) {
        NodeArg* output = p_node->MutableOutputDefs()[it->GetSrcArgIndex()];
        if (output_ref_counts.find(output) == output_ref_counts.end()) {
          output_ref_counts.emplace(output, 1);
        } else {
          output_ref_counts.at(output)++;
        }
      }
    }

    sub_graph.SetInputs(graph_const_inputs);
    NodeArgVec graph_outputs;
    ConstNodeArgVec graph_const_outputs;
    for (auto& pair : output_ref_counts) {
      graph_outputs.emplace_back(pair.first);
      graph_const_outputs.emplace_back(pair.first);
    }
    sub_graph.SetOutputs(graph_const_outputs);

    auto model_proto = sub_model.ToProto();
    std::string model_str;
    model_proto.SerializeToString(&model_str);

    Node& fused_node = graph.AddNode(graph.GenerateNodeName("TritonOp"), "TritonOp", "Fused nodes for TritonOp",
                                     graph_inputs, graph_outputs, {}, kMSDomain);
    fused_node.AddAttribute("onnx_string", model_str);
    fused_node.SetExecutionProviderType(partition.nodes[0]->GetExecutionProviderType());

    for (auto& p_node : partition.nodes) {
      graph_utils::RemoveNodeOutputEdges(graph, *p_node);
      graph.RemoveNode(p_node->Index());
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // USE_CUDA
