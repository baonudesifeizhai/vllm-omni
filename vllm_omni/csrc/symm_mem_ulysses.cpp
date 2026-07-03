// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from NVIDIA TensorRT-LLM's Async Ulysses implementation.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace vllm_omni::symm_mem_ulysses {

cudaError_t launch_permute_scatter(
    const void* input,
    void* send,
    void* recv,
    int rank,
    int batch,
    int shard_seq_len,
    int heads,
    int head_dim,
    int world_size,
    cudaStream_t stream);

cudaError_t launch_post_unscatter_qkv(
    const void* query_in,
    const void* key_in,
    const void* value_in,
    void* query_out,
    void* key_out,
    void* value_out,
    int world_size,
    int batch,
    int shard_seq_len,
    int query_heads,
    int key_heads,
    int value_heads,
    int head_dim,
    cudaStream_t stream);

inline void cuda_check(cudaError_t error) {
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

struct SendHandle : torch::CustomClassHolder {
  torch::Tensor send;
  std::vector<int64_t> peer_recv_ptrs;
  int64_t slot_bytes = 0;
  std::string group_name;
};

namespace {

class AsyncUlyssesOp {
 public:
  explicit AsyncUlyssesOp(c10::intrusive_ptr<c10d::ProcessGroup> group)
      : group_(std::move(group)) {
    register_group();
  }

  std::tuple<torch::Tensor, c10::intrusive_ptr<SendHandle>> prepare(
      torch::Tensor input) {
    TORCH_CHECK(input.is_cuda());
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(input.dim() == 4);
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16);

    c10::cuda::CUDAGuard guard(input.device());
    const int batch = input.size(0);
    const int shard_seq_len = input.size(1);
    const int heads = input.size(2);
    const int head_dim = input.size(3);
    const int world_size = group_->getSize();
    const int rank = group_->getRank();
    TORCH_CHECK(heads % world_size == 0);
    TORCH_CHECK(head_dim % 8 == 0);

    const int local_heads = heads / world_size;
    const std::vector<int64_t> shape = {
        world_size, batch, shard_seq_len, local_heads, head_dim};
    const int64_t bytes = input.numel() * input.element_size();
    Slot& slot = get_slot(bytes);
    auto options = input.options();
    auto send = torch::from_blob(slot.send, shape, [](void*) {}, options);
    auto recv = torch::from_blob(slot.recv, shape, [](void*) {}, options);

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
    cuda_check(launch_permute_scatter(
        input.data_ptr(),
        send.data_ptr(),
        recv.data_ptr(),
        rank,
        batch,
        shard_seq_len,
        heads,
        head_dim,
        world_size,
        stream));

    auto handle = c10::make_intrusive<SendHandle>();
    handle->send = std::move(send);
    handle->slot_bytes = bytes / world_size;
    handle->group_name = group_->getGroupName();
    handle->peer_recv_ptrs.reserve(world_size);
    for (void* ptr : slot.peer_ptrs) {
      handle->peer_recv_ptrs.push_back(reinterpret_cast<int64_t>(ptr));
    }
    return {std::move(recv), std::move(handle)};
  }

  void push(const c10::intrusive_ptr<SendHandle>& handle) {
    const int world_size = group_->getSize();
    const int rank = group_->getRank();
    const int peers = world_size - 1;
    if (peers == 0) {
      return;
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaStreamCaptureStatus capture_status;
    cuda_check(cudaStreamIsCapturing(stream, &capture_status));
    const auto* send = static_cast<const char*>(handle->send.data_ptr());

    if (capture_status == cudaStreamCaptureStatusNone) {
      std::vector<void*> destinations;
      std::vector<const void*> sources;
      std::vector<size_t> sizes;
      destinations.reserve(peers);
      sources.reserve(peers);
      sizes.reserve(peers);
      for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) {
          continue;
        }
        auto* peer_recv = reinterpret_cast<char*>(handle->peer_recv_ptrs[peer]);
        destinations.push_back(peer_recv + rank * handle->slot_bytes);
        sources.push_back(send + peer * handle->slot_bytes);
        sizes.push_back(handle->slot_bytes);
      }

      cudaMemcpyAttributes attributes{};
      attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
      attributes.flags = 1u;
      size_t attribute_indices[] = {0};
      cuda_check(cudaMemcpyBatchAsync(
          destinations.data(),
          sources.data(),
          sizes.data(),
          peers,
          &attributes,
          attribute_indices,
          1,
          stream));
      return;
    }

    for (int peer = 0; peer < world_size; ++peer) {
      if (peer == rank) {
        continue;
      }
      auto* peer_recv = reinterpret_cast<char*>(handle->peer_recv_ptrs[peer]);
      cuda_check(cudaMemcpyAsync(
          peer_recv + rank * handle->slot_bytes,
          send + peer * handle->slot_bytes,
          handle->slot_bytes,
          cudaMemcpyDeviceToDevice,
          stream));
    }
  }

  void barrier() {
    canonical_handle_->barrier(0, 10000);
  }

 private:
  struct Slot {
    at::Tensor symmetric_tensor;
    c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> handle;
    void* recv = nullptr;
    void* send = nullptr;
    size_t bytes = 0;
    std::vector<void*> peer_ptrs;
  };

  void register_group() {
    static std::set<std::string> registered;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    const std::string& name = group_->getGroupName();
    if (registered.insert(name).second) {
      c10d::symmetric_memory::set_group_info(
          name, group_->getRank(), group_->getSize(), group_->getStore());
    }
  }

  Slot& get_slot(size_t required_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot& slot = slots_[next_slot_];
    next_slot_ = (next_slot_ + 1) % slots_.size();
    if (slot.recv != nullptr && slot.bytes >= required_bytes) {
      return slot;
    }

    cudaStreamCaptureStatus capture_status;
    cuda_check(cudaStreamIsCapturing(
        at::cuda::getCurrentCUDAStream().stream(), &capture_status));
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone);

    int device_index;
    cuda_check(cudaGetDevice(&device_index));
    const c10::Device device(c10::DeviceType::CUDA, device_index);
    auto symmetric_tensor = c10d::symmetric_memory::empty_strided_p2p(
        {static_cast<int64_t>(required_bytes)},
        {1},
        at::kByte,
        device,
        group_->getGroupName(),
        std::nullopt);
    auto handle = c10d::symmetric_memory::rendezvous(
        symmetric_tensor, group_->getGroupName());
    auto peer_ptrs = handle->get_buffer_ptrs();
    void* send = nullptr;
    cuda_check(cudaMalloc(&send, required_bytes));
    if (slot.send != nullptr) {
      cuda_check(cudaFree(slot.send));
    }

    slot.symmetric_tensor = std::move(symmetric_tensor);
    slot.handle = std::move(handle);
    slot.recv = slot.symmetric_tensor.data_ptr();
    slot.send = send;
    slot.bytes = required_bytes;
    slot.peer_ptrs.assign(peer_ptrs.begin(), peer_ptrs.end());
    if (!canonical_handle_) {
      canonical_handle_ = slot.handle;
    }
    return slot;
  }

  c10::intrusive_ptr<c10d::ProcessGroup> group_;
  std::array<Slot, 3> slots_{};
  size_t next_slot_ = 0;
  std::mutex mutex_;
  c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory>
      canonical_handle_;
};

std::shared_ptr<AsyncUlyssesOp> get_op(
    const c10::intrusive_ptr<c10d::ProcessGroup>& group) {
  static std::map<std::string, std::shared_ptr<AsyncUlyssesOp>> cache;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  const std::string& name = group->getGroupName();
  auto [it, inserted] = cache.emplace(name, nullptr);
  if (inserted) {
    it->second = std::make_shared<AsyncUlyssesOp>(group);
  }
  return it->second;
}

std::tuple<torch::Tensor, c10::intrusive_ptr<SendHandle>> prepare(
    torch::Tensor input,
    const c10::intrusive_ptr<c10d::ProcessGroup>& group) {
  return get_op(group)->prepare(std::move(input));
}

void push(
    const c10::intrusive_ptr<SendHandle>& handle,
    const c10::intrusive_ptr<c10d::ProcessGroup>& group) {
  TORCH_CHECK(handle->group_name == group->getGroupName());
  get_op(group)->push(handle);
}

void barrier(const c10::intrusive_ptr<c10d::ProcessGroup>& group) {
  get_op(group)->barrier();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
post_unscatter_qkv(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda());
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::BFloat16 &&
      key.scalar_type() == at::ScalarType::BFloat16 &&
      value.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(query.dim() == 5 && key.dim() == 5 && value.dim() == 5);
  TORCH_CHECK(
      query.size(0) == key.size(0) && query.size(0) == value.size(0));
  TORCH_CHECK(
      query.size(1) == key.size(1) && query.size(1) == value.size(1));
  TORCH_CHECK(
      query.size(2) == key.size(2) && query.size(2) == value.size(2));
  TORCH_CHECK(
      query.size(4) == key.size(4) && query.size(4) == value.size(4));

  c10::cuda::CUDAGuard guard(query.device());
  const int world_size = query.size(0);
  const int batch = query.size(1);
  const int shard_seq_len = query.size(2);
  const int query_heads = query.size(3);
  const int key_heads = key.size(3);
  const int value_heads = value.size(3);
  const int head_dim = query.size(4);
  TORCH_CHECK(head_dim % 8 == 0);
  TORCH_CHECK(
      std::max({query_heads, key_heads, value_heads}) * (head_dim / 8) <=
      1024);

  auto query_out = torch::empty(
      {batch, world_size * shard_seq_len, query_heads, head_dim},
      query.options());
  auto key_out = torch::empty(
      {batch, world_size * shard_seq_len, key_heads, head_dim}, key.options());
  auto value_out = torch::empty(
      {batch, world_size * shard_seq_len, value_heads, head_dim},
      value.options());
  auto stream = at::cuda::getCurrentCUDAStream(query.get_device()).stream();
  cuda_check(launch_post_unscatter_qkv(
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      query_out.data_ptr(),
      key_out.data_ptr(),
      value_out.data_ptr(),
      world_size,
      batch,
      shard_seq_len,
      query_heads,
      key_heads,
      value_heads,
      head_dim,
      stream));
  return {std::move(query_out), std::move(key_out), std::move(value_out)};
}

}  // namespace

}  // namespace vllm_omni::symm_mem_ulysses

TORCH_LIBRARY_FRAGMENT(vllm_omni, library) {
  library.class_<vllm_omni::symm_mem_ulysses::SendHandle>(
      "SymmMemUlyssesSendHandle");
  library.def(
      "symm_mem_ulysses_prepare(Tensor input, "
      "__torch__.torch.classes.c10d.ProcessGroup group) -> "
      "(Tensor, __torch__.torch.classes.vllm_omni.SymmMemUlyssesSendHandle)");
  library.def(
      "symm_mem_ulysses_push("
      "__torch__.torch.classes.vllm_omni.SymmMemUlyssesSendHandle handle, "
      "__torch__.torch.classes.c10d.ProcessGroup group) -> ()");
  library.def(
      "symm_mem_ulysses_barrier("
      "__torch__.torch.classes.c10d.ProcessGroup group) -> ()");
  library.def(
      "symm_mem_ulysses_post_unscatter_qkv("
      "Tensor query, Tensor key, Tensor value) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(vllm_omni, CompositeExplicitAutograd, library) {
  library.impl(
      "symm_mem_ulysses_prepare",
      &vllm_omni::symm_mem_ulysses::prepare);
  library.impl(
      "symm_mem_ulysses_push", &vllm_omni::symm_mem_ulysses::push);
  library.impl(
      "symm_mem_ulysses_barrier", &vllm_omni::symm_mem_ulysses::barrier);
}

TORCH_LIBRARY_IMPL(vllm_omni, CUDA, library) {
  library.impl(
      "symm_mem_ulysses_post_unscatter_qkv",
      &vllm_omni::symm_mem_ulysses::post_unscatter_qkv);
}
