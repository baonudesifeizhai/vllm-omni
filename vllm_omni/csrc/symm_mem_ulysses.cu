// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from NVIDIA TensorRT-LLM's Async Ulysses kernels.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace vllm_omni::symm_mem_ulysses {

namespace {

constexpr int kVec = 8;
constexpr int kBlockRows = 32;
constexpr int kThreads = 128;

__global__ void __launch_bounds__(kThreads) permute_scatter_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ send,
    __nv_bfloat16* __restrict__ recv,
    int rank,
    int rows,
    int heads,
    int head_dim,
    int local_heads) {
  const int row_block = blockIdx.x;
  const int head = blockIdx.y;
  const int peer = head / local_heads;
  const int local_head = head - peer * local_heads;
  const int slot = peer == rank ? rank : peer;
  __nv_bfloat16* dst = peer == rank ? recv : send;

  const int vecs_per_head = head_dim / kVec;
  const int tasks = kBlockRows * vecs_per_head;
  const int row_base = row_block * kBlockRows;
  const auto* input_vec = reinterpret_cast<const uint4*>(input);
  auto* dst_vec = reinterpret_cast<uint4*>(dst);

  for (int task = threadIdx.x; task < tasks; task += blockDim.x) {
    const int row_in_block = task / vecs_per_head;
    const int vec = task - row_in_block * vecs_per_head;
    const int row = row_base + row_in_block;
    if (row >= rows) {
      continue;
    }

    const int64_t src =
        static_cast<int64_t>(row) * heads * vecs_per_head +
        head * vecs_per_head + vec;
    const int64_t out =
        static_cast<int64_t>(slot) * rows * local_heads * vecs_per_head +
        static_cast<int64_t>(row) * local_heads * vecs_per_head +
        local_head * vecs_per_head + vec;
    dst_vec[out] = input_vec[src];
  }
}

__global__ void post_unscatter_qkv_kernel(
    const __nv_bfloat16* __restrict__ query_in,
    const __nv_bfloat16* __restrict__ key_in,
    const __nv_bfloat16* __restrict__ value_in,
    __nv_bfloat16* __restrict__ query_out,
    __nv_bfloat16* __restrict__ key_out,
    __nv_bfloat16* __restrict__ value_out,
    int world_size,
    int batch,
    int shard_seq_len,
    int query_heads,
    int key_heads,
    int value_heads,
    int head_dim) {
  const int tensor_idx = blockIdx.z;
  const __nv_bfloat16* input =
      tensor_idx == 0 ? query_in : (tensor_idx == 1 ? key_in : value_in);
  __nv_bfloat16* output =
      tensor_idx == 0 ? query_out : (tensor_idx == 1 ? key_out : value_out);
  const int heads =
      tensor_idx == 0 ? query_heads : (tensor_idx == 1 ? key_heads : value_heads);
  const int vecs_per_head = head_dim / kVec;
  const int task = threadIdx.x;
  const int tasks = heads * vecs_per_head;
  if (task >= tasks) {
    return;
  }

  const int head = task / vecs_per_head;
  const int vec = task - head * vecs_per_head;
  const int peer_seq = blockIdx.x;
  const int peer = peer_seq / shard_seq_len;
  const int seq = peer_seq - peer * shard_seq_len;
  const int batch_idx = blockIdx.y;
  const int global_seq_len = world_size * shard_seq_len;

  const int64_t src =
      (((static_cast<int64_t>(peer) * batch + batch_idx) * shard_seq_len +
        seq) *
           heads +
       head) *
          vecs_per_head +
      vec;
  const int64_t dst =
      ((static_cast<int64_t>(batch_idx) * global_seq_len + peer_seq) * heads +
       head) *
          vecs_per_head +
      vec;
  reinterpret_cast<uint4*>(output)[dst] =
      reinterpret_cast<const uint4*>(input)[src];
}

}  // namespace

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
    cudaStream_t stream) {
  const int rows = batch * shard_seq_len;
  const int local_heads = heads / world_size;
  const dim3 grid((rows + kBlockRows - 1) / kBlockRows, heads);
  permute_scatter_kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(send),
      static_cast<__nv_bfloat16*>(recv),
      rank,
      rows,
      heads,
      head_dim,
      local_heads);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  const int threads =
      std::max({query_heads, key_heads, value_heads}) * (head_dim / kVec);
  const dim3 grid(world_size * shard_seq_len, batch, 3);
  post_unscatter_qkv_kernel<<<grid, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query_in),
      static_cast<const __nv_bfloat16*>(key_in),
      static_cast<const __nv_bfloat16*>(value_in),
      static_cast<__nv_bfloat16*>(query_out),
      static_cast<__nv_bfloat16*>(key_out),
      static_cast<__nv_bfloat16*>(value_out),
      world_size,
      batch,
      shard_seq_len,
      query_heads,
      key_heads,
      value_heads,
      head_dim);
  return cudaGetLastError();
}

}  // namespace vllm_omni::symm_mem_ulysses
