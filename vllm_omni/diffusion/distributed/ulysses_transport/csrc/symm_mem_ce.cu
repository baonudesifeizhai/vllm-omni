// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Copy-Engine Ulysses all-to-all over PyTorch symmetric memory. The transfer
// and device-epoch barrier design is based on Apache-2.0 Fast-Ulysses. Layout
// conversion is encoded in pitched peer copies, so only the small barrier uses
// an SM; the payload itself runs on CUDA Copy Engines.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/library.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace symm = c10d::symmetric_memory;

constexpr int64_t kScatterHeads = 0;
constexpr int64_t kGatherHeads = 1;
constexpr int kMaxWorldSize = 8;

class Event {
 public:
  Event() {
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  ~Event() {
    cudaEventDestroy(event_);
  }

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  operator cudaEvent_t() const {
    return event_;
  }

 private:
  cudaEvent_t event_ = nullptr;
};

int64_t protocol_slots(int world_size) {
  // [barrier arrivals x P][epoch]
  return world_size + 1;
}

int64_t barrier_offset_slots(int world_size) {
  const int64_t pad_slots = static_cast<int64_t>(symm::get_signal_pad_size()) / sizeof(uint64_t);
  // PyTorch owns channels at the beginning of the signal pad. Reserve the
  // final slots for this transport so the two protocols cannot alias.
  TORCH_CHECK(
      pad_slots >= protocol_slots(world_size) + kMaxWorldSize,
      "PyTorch's symmetric-memory signal pad is too small for the Ulysses "
      "barrier: ",
      pad_slots,
      " uint64 slots");
  return pad_slots - protocol_slots(world_size);
}

cudaStream_t get_transfer_stream(int device_index) {
  static std::mutex mutex;
  static std::unordered_map<int, c10::cuda::CUDAStream> streams;
  std::lock_guard<std::mutex> guard(mutex);
  auto it = streams.find(device_index);
  if (it == streams.end()) {
    it = streams
             .emplace(
                 device_index,
                 c10::cuda::getStreamFromPool(
                     /*isHighPriority=*/true,
                     device_index))
             .first;
  }
  return it->second.stream();
}

struct BarrierPeers {
  uint64_t ptrs[kMaxWorldSize];
};

__device__ __forceinline__ void release_max_u64(uint64_t* address, uint64_t value) {
#if __CUDA_ARCH__ >= 700
  asm volatile(
      "red.release.sys.global.max.u64 [%0], %1;" :: "l"(address), "l"(value) : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ uint64_t acquire_load_u64(const uint64_t* address) {
#if __CUDA_ARCH__ >= 700
  uint64_t value;
  asm volatile(
      "ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
#else
  asm volatile("trap;");
  return 0;
#endif
}

__global__ void barrier_kernel(
    uint64_t* local,
    BarrierPeers peers,
    int world_size,
    int rank) {
  __shared__ uint64_t epoch;
  if (threadIdx.x == 0) {
    epoch = atomicAdd(
                reinterpret_cast<unsigned long long*>(local + world_size),
                1ULL) +
        1;
  }
  __syncthreads();

  const int peer = threadIdx.x;
  if (peer >= world_size) {
    return;
  }
  release_max_u64(
      reinterpret_cast<uint64_t*>(peers.ptrs[peer]) + rank, epoch);
  while (acquire_load_u64(local + peer) < epoch) {
  }
}

void launch_barrier(
    cudaStream_t stream,
    const std::vector<void*>& signal_pad_ptrs,
    int rank,
    int world_size) {
  if (world_size == 1) {
    return;
  }
  TORCH_CHECK(
      static_cast<int>(signal_pad_ptrs.size()) == world_size,
      "symmetric-memory signal-pad peer count does not match process group");

  const auto byte_offset =
      static_cast<uint64_t>(barrier_offset_slots(world_size) * sizeof(uint64_t));
  BarrierPeers peers{};
  for (int peer = 0; peer < world_size; ++peer) {
    peers.ptrs[peer] =
        reinterpret_cast<uint64_t>(signal_pad_ptrs[peer]) + byte_offset;
  }
  auto* local = reinterpret_cast<uint64_t*>(peers.ptrs[rank]);
  barrier_kernel<<<1, 32, 0, stream>>>(local, peers, world_size, rank);
  C10_CUDA_CHECK(cudaGetLastError());
}

void init_ulysses_window_(
    at::Tensor& window,
    const std::string& group_name) {
  TORCH_CHECK(window.is_cuda(), "symm_mem Ulysses window must be CUDA");
  auto handle = symm::rendezvous(window, group_name);
  TORCH_CHECK(handle != nullptr, "window must be allocated by PyTorch symmetric memory");

  const int rank = handle->get_rank();
  const int world_size = handle->get_world_size();
  TORCH_CHECK(
      world_size >= 1 && world_size <= kMaxWorldSize,
      "symm_mem Ulysses world_size must be in [1, 8]");

  // A recycled signal pad can contain a later epoch from its previous owner.
  // Clear this rank's private tail, then use a PyTorch channel as the one-time
  // allocation barrier before the custom protocol starts using the region.
  const int64_t offset = barrier_offset_slots(world_size);
  handle
      ->get_signal_pad(
          rank,
          {protocol_slots(world_size)},
          at::kLong,
          offset)
      .zero_();
  handle->barrier(/*channel=*/0, /*timeout_ms=*/0);
}

void copy_2d(
    void* dst,
    size_t dst_pitch,
    const void* src,
    size_t src_pitch,
    size_t width,
    size_t rows,
    cudaStream_t stream) {
  C10_CUDA_CHECK(cudaMemcpy2DAsync(
      dst,
      dst_pitch,
      src,
      src_pitch,
      width,
      rows,
      cudaMemcpyDefault,
      stream));
}

c10::intrusive_ptr<symm::SymmetricMemory> get_window_handle(
    const at::Tensor& window,
    const std::string& group_name) {
  TORCH_CHECK(window.is_cuda(), "symm_mem Ulysses window must be CUDA");
  TORCH_CHECK(window.is_contiguous(), "symm_mem Ulysses window must be contiguous");
  auto handle = symm::rendezvous(window, group_name);
  TORCH_CHECK(handle != nullptr, "window must be allocated by PyTorch symmetric memory");
  TORCH_CHECK(
      handle->world_within_direct_access(),
      "all Ulysses ranks must support direct peer access");
  TORCH_CHECK(
      handle->get_world_size() >= 1 && handle->get_world_size() <= kMaxWorldSize,
      "symm_mem Ulysses world_size must be in [1, 8]");
  return handle;
}

void validate_scatter_shapes(
    const at::Tensor& input,
    const at::Tensor& output,
    int world_size) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "symm_mem Ulysses requires CUDA tensors");
  TORCH_CHECK(input.dim() == 4 && output.dim() == 4, "symm_mem Ulysses tensors must be 4-D");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "symm_mem Ulysses tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "symm_mem Ulysses dtype mismatch");
  TORCH_CHECK(input.device() == output.device(), "symm_mem Ulysses device mismatch");

  const int64_t batch = input.size(0);
  const int64_t seq_local = input.size(1);
  const int64_t heads = input.size(2);
  const int64_t head_dim = input.size(3);
  TORCH_CHECK(heads > 0, "symm_mem Ulysses requires at least one head");
  const bool shard_heads = heads % world_size == 0;
  const bool replicate_heads = world_size % heads == 0;
  TORCH_CHECK(
      shard_heads || replicate_heads,
      "head count and Ulysses world_size must form nested partitions");
  const int64_t heads_local = shard_heads ? heads / world_size : 1;
  TORCH_CHECK(
      output.sizes() ==
          at::IntArrayRef({batch, seq_local * world_size, heads_local, head_dim}),
      "invalid output shape for Ulysses head scatter");
}

void emit_scatter_peer(
    const at::Tensor& input,
    const c10::intrusive_ptr<symm::SymmetricMemory>& output_handle,
    int peer,
    cudaStream_t stream) {
  const int rank = output_handle->get_rank();
  const int world_size = output_handle->get_world_size();
  const int64_t batch = input.size(0);
  const int64_t seq_local = input.size(1);
  const int64_t heads = input.size(2);
  const int64_t head_dim = input.size(3);
  const bool shard_heads = heads % world_size == 0;
  const int64_t heads_local = shard_heads ? heads / world_size : 1;
  const size_t d_bytes =
      static_cast<size_t>(head_dim) * input.element_size();
  const size_t src_batch = static_cast<size_t>(seq_local * heads) * d_bytes;
  const size_t dst_batch =
      static_cast<size_t>(seq_local * world_size * heads_local) * d_bytes;
  const size_t src_pitch = static_cast<size_t>(heads) * d_bytes;
  const size_t dst_pitch = static_cast<size_t>(heads_local) * d_bytes;
  const int64_t source_head = shard_heads
      ? static_cast<int64_t>(peer) * heads_local
      : peer / (world_size / heads);

  const auto peer_ptrs = output_handle->get_buffer_ptrs();
  TORCH_CHECK(
      static_cast<int>(peer_ptrs.size()) == world_size,
      "symmetric-memory peer count does not match process group");
  const auto* src = static_cast<const uint8_t*>(input.data_ptr());
  auto* peer_base = static_cast<uint8_t*>(peer_ptrs[peer]);
  for (int64_t b = 0; b < batch; ++b) {
    const void* src_ptr = src + static_cast<size_t>(b) * src_batch +
        static_cast<size_t>(source_head) * d_bytes;
    void* dst_ptr = peer_base + static_cast<size_t>(b) * dst_batch +
        static_cast<size_t>(rank * seq_local * heads_local) * d_bytes;
    copy_2d(
        dst_ptr,
        dst_pitch,
        src_ptr,
        src_pitch,
        dst_pitch,
        seq_local,
        stream);
  }
}

void emit_scatter_remote_peers(
    const at::Tensor& input,
    const c10::intrusive_ptr<symm::SymmetricMemory>& output_handle,
    cudaStream_t stream) {
  const int rank = output_handle->get_rank();
  const int world_size = output_handle->get_world_size();
  for (int step = 1; step < world_size; ++step) {
    const int peer = rank ^ step;
    if (peer < world_size) {
      emit_scatter_peer(input, output_handle, peer, stream);
    }
  }
  if ((world_size & (world_size - 1)) != 0) {
    for (int peer = 0; peer < world_size; ++peer) {
      if (peer == rank || (peer ^ rank) < world_size) {
        continue;
      }
      emit_scatter_peer(input, output_handle, peer, stream);
    }
  }
}

void ce_ulysses_scatter_kv_(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& key_output,
    at::Tensor& value_output,
    const std::string& group_name) {
  TORCH_CHECK(
      key.sizes() == value.sizes(),
      "fused symm_mem K/V exchange requires matching input shapes");
  TORCH_CHECK(
      key.scalar_type() == value.scalar_type() && key.device() == value.device(),
      "fused symm_mem K/V exchange requires matching dtype and device");
  auto key_handle = get_window_handle(key_output, group_name);
  auto value_handle = get_window_handle(value_output, group_name);
  const int rank = key_handle->get_rank();
  const int world_size = key_handle->get_world_size();
  TORCH_CHECK(
      value_handle->get_rank() == rank &&
          value_handle->get_world_size() == world_size,
      "fused symm_mem K/V windows must use the same process group");
  validate_scatter_shapes(key, key_output, world_size);
  validate_scatter_shapes(value, value_output, world_size);

  c10::cuda::CUDAGuard device_guard(key.device());
  const int device_index = key.get_device();
  const cudaStream_t caller = at::cuda::getCurrentCUDAStream(device_index);
  const cudaStream_t transfer = get_transfer_stream(device_index);
  const auto signal_pad_ptrs = key_handle->get_signal_pad_ptrs();

  // One reuse barrier protects both K and V because their previous consumers
  // are ordered before this point on every rank's caller stream.
  launch_barrier(caller, signal_pad_ptrs, rank, world_size);
  const Event ready;
  const Event done;
  C10_CUDA_CHECK(cudaEventRecord(ready, caller));
  C10_CUDA_CHECK(cudaStreamWaitEvent(transfer, ready, 0));

  emit_scatter_remote_peers(key, key_handle, transfer);
  emit_scatter_remote_peers(value, value_handle, transfer);
  emit_scatter_peer(key, key_handle, rank, caller);
  emit_scatter_peer(value, value_handle, rank, caller);

  C10_CUDA_CHECK(cudaEventRecord(done, transfer));
  C10_CUDA_CHECK(cudaStreamWaitEvent(caller, done, 0));
  launch_barrier(caller, signal_pad_ptrs, rank, world_size);
}

void ce_ulysses_a2a(
    const at::Tensor& input,
    at::Tensor& output,
    int64_t direction,
    const std::string& group_name) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "symm_mem Ulysses requires CUDA tensors");
  TORCH_CHECK(input.dim() == 4, "symm_mem Ulysses input must be 4-D");
  TORCH_CHECK(output.dim() == 4, "symm_mem Ulysses output must be 4-D");
  TORCH_CHECK(input.is_contiguous(), "symm_mem Ulysses input must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "symm_mem Ulysses output must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "symm_mem Ulysses dtype mismatch");
  TORCH_CHECK(input.device() == output.device(), "symm_mem Ulysses device mismatch");
  TORCH_CHECK(
      direction == kScatterHeads || direction == kGatherHeads,
      "symm_mem Ulysses direction must be 0 (scatter heads) or 1 (gather heads)");

  auto handle = symm::rendezvous(output, group_name);
  TORCH_CHECK(handle != nullptr, "output must be allocated by PyTorch symmetric memory");
  TORCH_CHECK(handle->world_within_direct_access(), "all Ulysses ranks must support direct peer access");

  const int rank = handle->get_rank();
  const int world_size = handle->get_world_size();
  TORCH_CHECK(
      world_size >= 1 && world_size <= kMaxWorldSize,
      "symm_mem Ulysses world_size must be in [1, 8]");

  const int64_t batch = input.size(0);
  const int64_t x1 = input.size(1);
  const int64_t x2 = input.size(2);
  const int64_t head_dim = input.size(3);
  const size_t element_size = input.element_size();
  const size_t d_bytes = static_cast<size_t>(head_dim) * element_size;

  c10::cuda::CUDAGuard device_guard(input.device());
  const int device_index = input.get_device();
  cudaDeviceProp device_properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&device_properties, device_index));
  TORCH_CHECK(
      device_properties.major >= 7,
      "symm_mem Ulysses requires compute capability 7.0 or newer for its "
      "system-scope barrier");
  const cudaStream_t caller = at::cuda::getCurrentCUDAStream(device_index);
  const cudaStream_t transfer = get_transfer_stream(device_index);

  // The opening barrier prevents a new exchange from overwriting a window
  // before every rank has enqueued its previous consumer on this stream.
  const auto signal_pad_ptrs = handle->get_signal_pad_ptrs();
  launch_barrier(caller, signal_pad_ptrs, rank, world_size);

  const Event ready;
  const Event done;
  C10_CUDA_CHECK(cudaEventRecord(ready, caller));
  C10_CUDA_CHECK(cudaStreamWaitEvent(transfer, ready, 0));

  const auto peer_ptrs = handle->get_buffer_ptrs();
  TORCH_CHECK(
      static_cast<int>(peer_ptrs.size()) == world_size,
      "symmetric-memory peer count does not match process group");
  const auto* src = static_cast<const uint8_t*>(input.data_ptr());

  if (direction == kScatterHeads) {
    const int64_t seq_local = x1;
    const int64_t heads = x2;
    TORCH_CHECK(heads > 0, "symm_mem Ulysses requires at least one head");
    const bool shard_heads = heads % world_size == 0;
    const bool replicate_heads = world_size % heads == 0;
    TORCH_CHECK(
        shard_heads || replicate_heads,
        "head count and Ulysses world_size must form nested partitions");
    const int64_t heads_local = shard_heads ? heads / world_size : 1;
    TORCH_CHECK(
        output.sizes() == at::IntArrayRef({batch, seq_local * world_size, heads_local, head_dim}),
        "invalid output shape for Ulysses head scatter");

    const size_t src_batch = static_cast<size_t>(seq_local * heads) * d_bytes;
    const size_t dst_batch = static_cast<size_t>(seq_local * world_size * heads_local) * d_bytes;
    const size_t src_pitch = static_cast<size_t>(heads) * d_bytes;
    const size_t dst_pitch = static_cast<size_t>(heads_local) * d_bytes;
    const size_t width = dst_pitch;

    auto emit_scatter = [&](int peer, cudaStream_t stream) {
      auto* peer_base = static_cast<uint8_t*>(peer_ptrs[peer]);
      const int64_t source_head = shard_heads
          ? static_cast<int64_t>(peer) * heads_local
          : peer / (world_size / heads);
      for (int64_t b = 0; b < batch; ++b) {
        const void* src_ptr = src + static_cast<size_t>(b) * src_batch +
            static_cast<size_t>(source_head) * d_bytes;
        void* dst_ptr = peer_base + static_cast<size_t>(b) * dst_batch +
            static_cast<size_t>(rank * seq_local * heads_local) * d_bytes;
        copy_2d(dst_ptr, dst_pitch, src_ptr, src_pitch, width, seq_local, stream);
      }
    };

    // XOR order pairs ranks without coordinating an additional schedule.
    for (int step = 1; step < world_size; ++step) {
      const int peer = rank ^ step;
      if (peer >= world_size) {
        continue;
      }
      emit_scatter(peer, transfer);
    }
    // XOR enumerates all peers only for power-of-two world sizes.
    if ((world_size & (world_size - 1)) != 0) {
      for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank || (peer ^ rank) < world_size) {
          continue;
        }
        emit_scatter(peer, transfer);
      }
    }
    // The local share does not cross a link and can overlap remote CE copies.
    emit_scatter(rank, caller);
  } else {
    const int64_t seq_global = x1;
    const int64_t heads_local = x2;
    TORCH_CHECK(seq_global % world_size == 0, "global sequence length must divide Ulysses world_size");
    const int64_t seq_local = seq_global / world_size;
    const int64_t heads = heads_local * world_size;
    TORCH_CHECK(
        output.sizes() == at::IntArrayRef({batch, seq_local, heads, head_dim}),
        "invalid output shape for Ulysses sequence scatter");

    const size_t src_batch = static_cast<size_t>(seq_global * heads_local) * d_bytes;
    const size_t dst_batch = static_cast<size_t>(seq_local * heads) * d_bytes;
    const size_t src_pitch = static_cast<size_t>(heads_local) * d_bytes;
    const size_t dst_pitch = static_cast<size_t>(heads) * d_bytes;
    const size_t width = src_pitch;

    auto emit_gather = [&](int peer, cudaStream_t stream) {
      auto* peer_base = static_cast<uint8_t*>(peer_ptrs[peer]);
      for (int64_t b = 0; b < batch; ++b) {
        const void* src_ptr = src + static_cast<size_t>(b) * src_batch +
            static_cast<size_t>(peer * seq_local * heads_local) * d_bytes;
        void* dst_ptr = peer_base + static_cast<size_t>(b) * dst_batch +
            static_cast<size_t>(rank * heads_local) * d_bytes;
        copy_2d(dst_ptr, dst_pitch, src_ptr, src_pitch, width, seq_local, stream);
      }
    };

    for (int step = 1; step < world_size; ++step) {
      const int peer = rank ^ step;
      if (peer >= world_size) {
        continue;
      }
      emit_gather(peer, transfer);
    }
    if ((world_size & (world_size - 1)) != 0) {
      for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank || (peer ^ rank) < world_size) {
          continue;
        }
        emit_gather(peer, transfer);
      }
    }
    emit_gather(rank, caller);
  }

  C10_CUDA_CHECK(cudaEventRecord(done, transfer));
  C10_CUDA_CHECK(cudaStreamWaitEvent(caller, done, 0));

  // Once every rank reaches this barrier, all peer writes into this rank's
  // window are complete and subsequent attention on the caller stream is safe.
  launch_barrier(caller, signal_pad_ptrs, rank, world_size);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(vllm_omni_symm_mem, m) {
  m.def("init_ulysses_window_(Tensor(a!) window, str group_name) -> ()");
  m.def("ce_ulysses_a2a(Tensor input, Tensor(a!) output, int direction, str group_name) -> ()");
  m.def(
      "ce_ulysses_scatter_kv_(Tensor key, Tensor value, Tensor(a!) key_output, "
      "Tensor(b!) value_output, str group_name) -> ()");
}

TORCH_LIBRARY_IMPL(vllm_omni_symm_mem, CUDA, m) {
  m.impl("init_ulysses_window_", TORCH_FN(init_ulysses_window_));
  m.impl("ce_ulysses_a2a", TORCH_FN(ce_ulysses_a2a));
  m.impl("ce_ulysses_scatter_kv_", TORCH_FN(ce_ulysses_scatter_kv_));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
