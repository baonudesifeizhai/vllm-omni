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

inline void cuda_check(cudaError_t error) {
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

cudaError_t launch_pack_qkv(
    const void* query,
    const void* key,
    const void* value,
    void* send,
    void* recv,
    int rank,
    int batch,
    int shard_seq_len,
    int query_heads,
    int key_heads,
    int value_heads,
    int head_dim,
    int world_size,
    cudaStream_t stream);

cudaError_t launch_unpack_qkv(
    const void* recv,
    void* query_out,
    void* key_out,
    void* value_out,
    int batch,
    int shard_seq_len,
    int query_shard_heads,
    int key_shard_heads,
    int value_shard_heads,
    int head_dim,
    int world_size,
    cudaStream_t stream);

namespace {

class AsyncUlyssesOp {
 public:
  explicit AsyncUlyssesOp(c10::intrusive_ptr<c10d::ProcessGroup> group)
      : group_(std::move(group)) {
    register_group();
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> exchange_qkv(
      torch::Tensor query,
      torch::Tensor key,
      torch::Tensor value) {
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda());
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous());
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4);
    TORCH_CHECK(query.scalar_type() == key.scalar_type());
    TORCH_CHECK(query.scalar_type() == value.scalar_type());
    TORCH_CHECK(query.device() == key.device());
    TORCH_CHECK(query.device() == value.device());

    c10::cuda::CUDAGuard guard(query.device());
    const int64_t batch = query.size(0);
    const int64_t shard_seq_len = query.size(1);
    const int64_t query_heads = query.size(2);
    const int64_t head_dim = query.size(3);
    TORCH_CHECK(key.size(0) == batch && value.size(0) == batch);
    TORCH_CHECK(key.size(1) == shard_seq_len && value.size(1) == shard_seq_len);
    TORCH_CHECK(key.size(3) == head_dim && value.size(3) == head_dim);

    const int world_size = group_->getSize();
    const int rank = group_->getRank();
    const int64_t key_heads = key.size(2);
    const int64_t value_heads = value.size(2);
    TORCH_CHECK(world_size > 1);
    TORCH_CHECK(query_heads % world_size == 0);
    TORCH_CHECK(key_heads % world_size == 0);
    TORCH_CHECK(value_heads % world_size == 0);

    const int64_t query_shard_heads = query_heads / world_size;
    const int64_t key_shard_heads = key_heads / world_size;
    const int64_t value_shard_heads = value_heads / world_size;
    const int64_t packed_shard_heads =
        query_shard_heads + key_shard_heads + value_shard_heads;
    TORCH_CHECK(query.element_size() == 2);
    TORCH_CHECK(head_dim % 8 == 0);
    const int64_t chunk_bytes =
        shard_seq_len * batch * packed_shard_heads * head_dim *
        query.element_size();
    const int64_t total_bytes = chunk_bytes * world_size;

    Slot& slot = get_slot(total_bytes);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cuda_check(launch_pack_qkv(
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        slot.send,
        slot.recv,
        rank,
        static_cast<int>(batch),
        static_cast<int>(shard_seq_len),
        static_cast<int>(query_heads),
        static_cast<int>(key_heads),
        static_cast<int>(value_heads),
        static_cast<int>(head_dim),
        world_size,
        stream));

    copy_to_peers(slot, static_cast<size_t>(chunk_bytes));
    barrier();

    auto query_out = torch::empty(
        {batch, world_size * shard_seq_len, query_shard_heads, head_dim},
        query.options());
    auto key_out = torch::empty(
        {batch, world_size * shard_seq_len, key_shard_heads, head_dim},
        query.options());
    auto value_out = torch::empty(
        {batch, world_size * shard_seq_len, value_shard_heads, head_dim},
        query.options());
    cuda_check(launch_unpack_qkv(
        slot.recv,
        query_out.data_ptr(),
        key_out.data_ptr(),
        value_out.data_ptr(),
        static_cast<int>(batch),
        static_cast<int>(shard_seq_len),
        static_cast<int>(query_shard_heads),
        static_cast<int>(key_shard_heads),
        static_cast<int>(value_shard_heads),
        static_cast<int>(head_dim),
        world_size,
        stream));
    return {std::move(query_out), std::move(key_out), std::move(value_out)};
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

  void copy_to_peers(const Slot& slot, size_t chunk_bytes) {
    const int world_size = group_->getSize();
    const int rank = group_->getRank();
    const int peers = world_size - 1;
    if (peers == 0) {
      return;
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaStreamCaptureStatus capture_status;
    cuda_check(cudaStreamIsCapturing(stream, &capture_status));
    const auto* send_ptr = static_cast<const char*>(slot.send);

    for (int peer = 0; peer < world_size; ++peer) {
      if (peer == rank) {
        continue;
      }
      auto* peer_recv = reinterpret_cast<char*>(slot.peer_ptrs[peer]);
      cuda_check(cudaMemcpyAsync(
          peer_recv + rank * chunk_bytes,
          send_ptr + peer * chunk_bytes,
          chunk_bytes,
          cudaMemcpyDeviceToDevice,
          stream));
    }
  }

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> exchange_qkv(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    const c10::intrusive_ptr<c10d::ProcessGroup>& group) {
  return get_op(group)->exchange_qkv(
      std::move(query), std::move(key), std::move(value));
}

}  // namespace

}  // namespace vllm_omni::symm_mem_ulysses

TORCH_LIBRARY_FRAGMENT(vllm_omni, library) {
  library.def(
      "symm_mem_ulysses_exchange_qkv("
      "Tensor query, Tensor key, Tensor value, "
      "__torch__.torch.classes.c10d.ProcessGroup group) -> "
      "(Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(vllm_omni, CUDA, library) {
  library.impl(
      "symm_mem_ulysses_exchange_qkv",
      &vllm_omni::symm_mem_ulysses::exchange_qkv);
}
