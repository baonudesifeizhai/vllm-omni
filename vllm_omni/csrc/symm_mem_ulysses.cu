// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>

#include <cstdint>

namespace vllm_omni::symm_mem_ulysses {

namespace {

constexpr int kElementsPerVec = 8;
constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads) pack_qkv_kernel(
    const void* __restrict__ query,
    const void* __restrict__ key,
    const void* __restrict__ value,
    void* __restrict__ send,
    void* __restrict__ recv,
    int64_t total_vecs,
    int rank,
    int batch,
    int shard_seq_len,
    int query_heads,
    int key_heads,
    int value_heads,
    int head_dim,
    int world_size,
    int query_shard_heads,
    int key_shard_heads,
    int value_shard_heads,
    int packed_shard_heads) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_vecs) {
    return;
  }

  const int vecs_per_head = head_dim / kElementsPerVec;
  int64_t tmp = linear;
  const int vec = tmp % vecs_per_head;
  tmp /= vecs_per_head;
  const int packed_head = tmp % packed_shard_heads;
  tmp /= packed_shard_heads;
  const int batch_idx = tmp % batch;
  tmp /= batch;
  const int seq_idx = tmp % shard_seq_len;
  const int peer = tmp / shard_seq_len;

  const auto* query_vec = reinterpret_cast<const uint4*>(query);
  const auto* key_vec = reinterpret_cast<const uint4*>(key);
  const auto* value_vec = reinterpret_cast<const uint4*>(value);
  auto* send_vec = reinterpret_cast<uint4*>(send);
  auto* recv_vec = reinterpret_cast<uint4*>(recv);

  const uint4 data = [&]() {
    if (packed_head < query_shard_heads) {
      const int src_head = peer * query_shard_heads + packed_head;
      const int64_t src =
          (((static_cast<int64_t>(batch_idx) * shard_seq_len + seq_idx) *
                query_heads +
            src_head) *
               vecs_per_head +
           vec);
      return query_vec[src];
    }
    if (packed_head < query_shard_heads + key_shard_heads) {
      const int local_head = packed_head - query_shard_heads;
      const int src_head = peer * key_shard_heads + local_head;
      const int64_t src =
          (((static_cast<int64_t>(batch_idx) * shard_seq_len + seq_idx) *
                key_heads +
            src_head) *
               vecs_per_head +
           vec);
      return key_vec[src];
    }
    const int local_head = packed_head - query_shard_heads - key_shard_heads;
    const int src_head = peer * value_shard_heads + local_head;
    const int64_t src =
        (((static_cast<int64_t>(batch_idx) * shard_seq_len + seq_idx) *
              value_heads +
          src_head) *
             vecs_per_head +
         vec);
    return value_vec[src];
  }();

  const int64_t dst =
      ((((static_cast<int64_t>(peer) * shard_seq_len + seq_idx) * batch +
         batch_idx) *
            packed_shard_heads +
        packed_head) *
           vecs_per_head +
       vec);
  if (peer == rank) {
    recv_vec[dst] = data;
  } else {
    send_vec[dst] = data;
  }
}

__global__ void __launch_bounds__(kThreads) unpack_qkv_kernel(
    const void* __restrict__ recv,
    void* __restrict__ query_out,
    void* __restrict__ key_out,
    void* __restrict__ value_out,
    int64_t total_vecs,
    int batch,
    int shard_seq_len,
    int head_dim,
    int world_size,
    int query_shard_heads,
    int key_shard_heads,
    int value_shard_heads,
    int packed_shard_heads) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_vecs) {
    return;
  }

  const int vecs_per_head = head_dim / kElementsPerVec;
  int64_t tmp = linear;
  const int vec = tmp % vecs_per_head;
  tmp /= vecs_per_head;
  const int packed_head = tmp % packed_shard_heads;
  tmp /= packed_shard_heads;
  const int global_seq_idx = tmp % (world_size * shard_seq_len);
  const int batch_idx = tmp / (world_size * shard_seq_len);
  const int peer = global_seq_idx / shard_seq_len;
  const int seq_idx = global_seq_idx - peer * shard_seq_len;

  const auto* recv_vec = reinterpret_cast<const uint4*>(recv);
  auto* query_vec = reinterpret_cast<uint4*>(query_out);
  auto* key_vec = reinterpret_cast<uint4*>(key_out);
  auto* value_vec = reinterpret_cast<uint4*>(value_out);

  const int64_t src =
      ((((static_cast<int64_t>(peer) * shard_seq_len + seq_idx) * batch +
         batch_idx) *
            packed_shard_heads +
        packed_head) *
           vecs_per_head +
       vec);
  const uint4 data = recv_vec[src];

  if (packed_head < query_shard_heads) {
    const int64_t dst =
        (((static_cast<int64_t>(batch_idx) * world_size * shard_seq_len +
           global_seq_idx) *
              query_shard_heads +
          packed_head) *
             vecs_per_head +
         vec);
    query_vec[dst] = data;
    return;
  }
  if (packed_head < query_shard_heads + key_shard_heads) {
    const int local_head = packed_head - query_shard_heads;
    const int64_t dst =
        (((static_cast<int64_t>(batch_idx) * world_size * shard_seq_len +
           global_seq_idx) *
              key_shard_heads +
          local_head) *
             vecs_per_head +
         vec);
    key_vec[dst] = data;
    return;
  }

  const int local_head = packed_head - query_shard_heads - key_shard_heads;
  const int64_t dst =
      (((static_cast<int64_t>(batch_idx) * world_size * shard_seq_len +
         global_seq_idx) *
            value_shard_heads +
        local_head) *
           vecs_per_head +
       vec);
  value_vec[dst] = data;
}

}  // namespace

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
    cudaStream_t stream) {
  const int query_shard_heads = query_heads / world_size;
  const int key_shard_heads = key_heads / world_size;
  const int value_shard_heads = value_heads / world_size;
  const int packed_shard_heads =
      query_shard_heads + key_shard_heads + value_shard_heads;
  const int vecs_per_head = head_dim / kElementsPerVec;
  const int64_t total_vecs =
      static_cast<int64_t>(world_size) * shard_seq_len * batch *
      packed_shard_heads * vecs_per_head;
  const int blocks = static_cast<int>((total_vecs + kThreads - 1) / kThreads);

  pack_qkv_kernel<<<blocks, kThreads, 0, stream>>>(
      query,
      key,
      value,
      send,
      recv,
      total_vecs,
      rank,
      batch,
      shard_seq_len,
      query_heads,
      key_heads,
      value_heads,
      head_dim,
      world_size,
      query_shard_heads,
      key_shard_heads,
      value_shard_heads,
      packed_shard_heads);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  const int packed_shard_heads =
      query_shard_heads + key_shard_heads + value_shard_heads;
  const int vecs_per_head = head_dim / kElementsPerVec;
  const int64_t total_vecs =
      static_cast<int64_t>(batch) * world_size * shard_seq_len *
      packed_shard_heads * vecs_per_head;
  const int blocks = static_cast<int>((total_vecs + kThreads - 1) / kThreads);

  unpack_qkv_kernel<<<blocks, kThreads, 0, stream>>>(
      recv,
      query_out,
      key_out,
      value_out,
      total_vecs,
      batch,
      shard_seq_len,
      head_dim,
      world_size,
      query_shard_heads,
      key_shard_heads,
      value_shard_heads,
      packed_shard_heads);
  return cudaGetLastError();
}

}  // namespace vllm_omni::symm_mem_ulysses
