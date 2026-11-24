// SPDX-License-Identifier: Apache-2.0
// Instantiations from kern/tools/flashinfer-sampling at the pinned runtime revision.
#include <flashinfer/sampling.cuh>

namespace fs = flashinfer::sampling;

constexpr uint32_t BLOCK = 1024;
constexpr uint32_t VEC = 4;

template __global__ void fs::OnlineSoftmaxFusedKernel<BLOCK, VEC, float, false>(float*, float*, float*, float,
                                                                               uint32_t);

template __global__ void fs::RadixTopKRenormProbKernel_MultiCTA<BLOCK, VEC, false, true, float, int>(
    float*, float*, int*, uint32_t, uint32_t, uint32_t, fs::RadixRowState*, uint32_t, uint32_t);

template __global__ void fs::TopPRenormProbKernel<BLOCK, fs::REDUCE_ALGO, VEC, float>(float*, float*, float*, float,
                                                                                     uint32_t);

template __global__ void fs::MinPSamplingFromProbKernel<BLOCK, fs::SCAN_ALGO, fs::REDUCE_ALGO, VEC, true, float, int>(
    float*, float*, int*, bool*, int*, float, uint32_t, uint64_t*, uint64_t, uint64_t*, uint64_t, uint32_t, uint32_t);

// ---------------------------------------------------------------------------
// Temperature-guarded verify sampling kernels: skip greedy (raw_temp <= 0)
// rows immediately (0 DRAM traffic), and execute bit-identical FlashInfer
// Softmax / Multi-CTA RadixTopK / TopP / MinP on stochastic (raw_temp > 0) rows.
// ---------------------------------------------------------------------------

extern "C" __global__ FLASHINFER_SAMPLING_LAUNCH_BOUNDS(BLOCK) void kern_verify_online_softmax_f32(
    float* logits, float* output, float* temperature_arr, float temperature_val, uint32_t d,
    const float* __restrict__ raw_temperature) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  if (raw_temperature[bx] <= 0.0f) return;

  using TempStorage = fs::OnlineSoftmaxTempStorage<BLOCK>;
  extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<TempStorage&>(smem);

  flashinfer::vec_t<float, VEC> logits_vec;
  float running_max = -cuda::std::numeric_limits<float>::infinity();
  float running_denominator = 0.0f;
  float threadlocal_running_denominator = 0.0f;

  float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
  const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;

#pragma unroll 2
  for (uint32_t i = 0; i < flashinfer::ceil_div(d, BLOCK * VEC); ++i) {
    logits_vec.fill(-cuda::std::numeric_limits<float>::infinity());
    if ((i * BLOCK + tx) * VEC < d) {
      logits_vec.cast_load(logits + bx * d + (i * BLOCK + tx) * VEC);
#pragma unroll
      for (uint32_t j = 0; j < VEC; ++j) {
        logits_vec[j] = __fmul_rn(logits_vec[j], inv_temp);
      }
    }
    float thread_max = -cuda::std::numeric_limits<float>::infinity();
#pragma unroll
    for (uint32_t j = 0; j < VEC; ++j) {
      thread_max = max(thread_max, logits_vec[j]);
    }
    float block_max = cub::BlockReduce<float, BLOCK>(temp_storage.block_prim.reduce)
                          .Reduce(thread_max, cuda::maximum<>{});
    if (tx == 0) temp_storage.shared_state.max_val = block_max;
    __syncthreads();
    block_max = temp_storage.shared_state.max_val;
    if (!isinf(block_max)) {
      float threadlocal_sum = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < VEC; ++j) {
        threadlocal_sum += __expf(logits_vec[j] - block_max);
      }
      float new_max = max(running_max, block_max);
      threadlocal_running_denominator =
          threadlocal_running_denominator * __expf(running_max - new_max) +
          threadlocal_sum * __expf(block_max - new_max);
      running_max = new_max;
    }
  }

  running_denominator = cub::BlockReduce<float, BLOCK>(temp_storage.block_prim.reduce)
                            .Sum(threadlocal_running_denominator);
  if (tx == 0) temp_storage.shared_state.denominator = running_denominator;
  __syncthreads();
  running_denominator = temp_storage.shared_state.denominator;

  const float final_max = running_max;
  const float inv_denominator = 1.0f / running_denominator;

  flashinfer::vec_t<float, VEC> prob_vec;
  for (uint32_t i = 0; i < flashinfer::ceil_div(d, BLOCK * VEC); ++i) {
    if ((i * BLOCK + tx) * VEC < d) {
      logits_vec.cast_load(logits + bx * d + (i * BLOCK + tx) * VEC);
#pragma unroll
      for (uint32_t j = 0; j < VEC; ++j) {
        logits_vec[j] = __fmul_rn(logits_vec[j], inv_temp);
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC; ++j) {
      prob_vec[j] = __expf(logits_vec[j] - final_max) * inv_denominator;
    }
    if ((i * BLOCK + tx) * VEC < d) {
      prob_vec.cast_store(output + bx * d + (i * BLOCK + tx) * VEC);
    }
  }
}

extern "C" __global__ void __launch_bounds__(BLOCK) kern_verify_radix_topk_f32(
    float* probs,
    float* renormed_prob,
    int* top_k_arr,
    uint32_t top_k_val, uint32_t vocab_size, uint32_t batch_size,
    fs::RadixRowState* row_states,
    uint32_t chunk_size,
    uint32_t ctas_per_group,
    const float* __restrict__ raw_temperature) {
  using Traits = fs::RadixTopKTraits<float>;
  using OrderedType = typename Traits::OrderedType;
  constexpr uint32_t RADIX = 256;

  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  extern __shared__ uint8_t smem[];
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + 4) + sizeof(float);
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;
  size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
  OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);

  fs::RadixRowState* state = &row_states[group_id];
  uint32_t num_groups = gridDim.x / ctas_per_group;
  uint32_t total_iterations = (batch_size + num_groups - 1) / num_groups;

  int barrier_phase = 0;
  uint32_t active_iter = 0;
  bool touched_state = false;

  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    uint32_t row_idx = group_id + iter * num_groups;
    if (row_idx >= batch_size) break;
    if (raw_temperature[row_idx] <= 0.0f) continue;

    touched_state = true;
    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);
    const uint32_t actual_chunk_size = chunk_end - chunk_start;

    uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];
    float pivot = 0.0f;
    float normalizer = 1.0f;

    if (k >= vocab_size) {
      float thread_sum = 0.0f;
      flashinfer::vec_t<float, VEC> data_vec;
      const uint32_t aligned_size = (actual_chunk_size / VEC) * VEC;
#pragma unroll 2
      for (uint32_t i = tx * VEC; i < aligned_size; i += BLOCK * VEC) {
        data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
        for (uint32_t j = 0; j < VEC; ++j) thread_sum += data_vec[j];
      }
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK) {
        thread_sum += probs[row_idx * vocab_size + chunk_start + i];
      }
      typedef cub::BlockReduce<float, BLOCK> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
      __syncthreads();

      if (tx == 0 && cta_in_group == 0) {
        state->sum_topk_fixed = 0;
      }
      fs::AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
      if (tx == 0 && block_sum > 0) {
        atomicAdd(&state->sum_topk_fixed,
                  static_cast<unsigned long long>(
                      double(block_sum) * fs::kRenormDeterministicScale + 0.5));
      }
      fs::AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
      float group_sum = float(double(state->sum_topk_fixed) / fs::kRenormDeterministicScale);
      normalizer = flashinfer::math::ptx_rcp(max(group_sum, 1e-8f));

#pragma unroll 2
      for (uint32_t i = tx * VEC; i < aligned_size; i += BLOCK * VEC) {
        data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
        for (uint32_t j = 0; j < VEC; ++j) data_vec[j] = data_vec[j] * normalizer;
        data_vec.store(renormed_prob + row_idx * vocab_size + chunk_start + i);
      }
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK) {
        renormed_prob[row_idx * vocab_size + chunk_start + i] =
            probs[row_idx * vocab_size + chunk_start + i] * normalizer;
      }

      constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;
      uint32_t next_first_hist_idx = ((active_iter + 1) * NUM_ROUNDS) % 3;
      if (cta_in_group == 0) {
        for (uint32_t i = tx; i < RADIX; i += BLOCK) {
          state->histogram[next_first_hist_idx][i] = 0;
        }
      }
      active_iter++;
      continue;
    }

    uint32_t local_gt_count = 0;
    uint32_t local_eq_count = 0;
    auto ordered_pivot = fs::RadixSelectFindPivot<BLOCK, VEC, false, false, float>(
        probs + row_idx * vocab_size, shared_ordered, local_histogram, suffix_sum, shared_scalars,
        state, chunk_start, actual_chunk_size, k, barrier_phase, ctas_per_group, cta_in_group, tx,
        active_iter, local_gt_count, local_eq_count);
    pivot = Traits::FromOrdered(ordered_pivot);

    float thread_sum = 0.0f;
    flashinfer::vec_t<float, VEC> data_vec;
    const uint32_t aligned_size = (actual_chunk_size / VEC) * VEC;
#pragma unroll 2
    for (uint32_t i = tx * VEC; i < aligned_size; i += BLOCK * VEC) {
      data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC; ++j) {
        if (data_vec[j] >= pivot) thread_sum += data_vec[j];
      }
    }
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK) {
      float val = probs[row_idx * vocab_size + chunk_start + i];
      if (val >= pivot) thread_sum += val;
    }
    typedef cub::BlockReduce<float, BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    __syncthreads();

    if (tx == 0 && cta_in_group == 0) {
      state->sum_topk_fixed = 0;
    }
    fs::AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
    if (tx == 0 && block_sum > 0) {
      atomicAdd(&state->sum_topk_fixed,
                static_cast<unsigned long long>(
                    double(block_sum) * fs::kRenormDeterministicScale + 0.5));
    }
    fs::AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
    float group_sum = float(double(state->sum_topk_fixed) / fs::kRenormDeterministicScale);
    normalizer = flashinfer::math::ptx_rcp(max(group_sum, 1e-8f));

#pragma unroll 2
    for (uint32_t i = tx * VEC; i < aligned_size; i += BLOCK * VEC) {
      data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC; ++j) {
        data_vec[j] = (data_vec[j] >= pivot) ? (data_vec[j] * normalizer) : 0.0f;
      }
      data_vec.store(renormed_prob + row_idx * vocab_size + chunk_start + i);
    }
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK) {
      float val = probs[row_idx * vocab_size + chunk_start + i];
      renormed_prob[row_idx * vocab_size + chunk_start + i] =
          (val >= pivot) ? (val * normalizer) : 0.0f;
    }
    active_iter++;
  }

  if (touched_state) {
    fs::RadixGroupResetStateLastCTA<BLOCK>(state, nullptr, barrier_phase, ctas_per_group, tx);
  }
}

extern "C" __global__ FLASHINFER_SAMPLING_LAUNCH_BOUNDS(BLOCK) void kern_verify_topp_renorm_f32(
    float* probs, float* renormed_prob, float* top_p_arr, float top_p_val, uint32_t d,
    const float* __restrict__ raw_temperature) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  if (raw_temperature[bx] <= 0.0f) return;

  const uint32_t row_idx = bx;
  float p = top_p_arr == nullptr ? top_p_val : top_p_arr[bx];

  extern __shared__ __align__(alignof(fs::RenormTempStorage<BLOCK, fs::REDUCE_ALGO>))
      uint8_t smem_renorm[];
  auto& temp_storage =
      reinterpret_cast<fs::RenormTempStorage<BLOCK, fs::REDUCE_ALGO>&>(smem_renorm);
  flashinfer::vec_t<float, VEC> probs_vec;

  if (p >= 1.0f) {
    float thread_sum = 0.0f;
    const uint32_t num_iters = flashinfer::ceil_div(d, BLOCK * VEC);
    for (uint32_t i = 0; i < num_iters; ++i) {
      probs_vec.fill(0.0f);
      const uint32_t base_idx = (i * BLOCK + tx) * VEC;
      if (base_idx < d) probs_vec.cast_load(probs + row_idx * d + base_idx);
#pragma unroll
      for (uint32_t j = 0; j < VEC; ++j) {
        if (base_idx + j < d) thread_sum += probs_vec[j];
      }
    }
    float row_sum =
        cub::BlockReduce<float, BLOCK, fs::REDUCE_ALGO>(temp_storage.block_prim.reduce)
            .Sum(thread_sum);
    if (tx == 0) temp_storage.row_sum = row_sum;
    __syncthreads();
    row_sum = temp_storage.row_sum;
    const float denom = (row_sum <= 1e-8f) ? 1.0f : row_sum;
    const float normalizer = flashinfer::math::ptx_rcp(denom);
    for (uint32_t i = 0; i < num_iters; ++i) {
      probs_vec.fill(0.0f);
      const uint32_t base_idx = (i * BLOCK + tx) * VEC;
      if (base_idx < d) probs_vec.cast_load(probs + row_idx * d + base_idx);
#pragma unroll
      for (uint32_t j = 0; j < VEC; ++j) {
        probs_vec[j] = (base_idx + j < d) ? (probs_vec[j] * normalizer) : 0.0f;
      }
      if (base_idx < d) probs_vec.cast_store(renormed_prob + row_idx * d + base_idx);
    }
    return;
  }

  temp_storage.max_val = 0;
  float max_val = fs::GetMaxValue<VEC, BLOCK, fs::REDUCE_ALGO,
                                  fs::RenormTempStorage<BLOCK, fs::REDUCE_ALGO>>(
      probs, row_idx, d, temp_storage);

  float low = 0, high = max_val;
  float min_gt_low, max_le_high;
  float sum_low = 1;
  do {
    float pivot_0 = fs::ieee_div(fs::ieee_add(high, fs::ieee_mul(2.f, low)), 3.f);
    float pivot_1 = fs::ieee_div(fs::ieee_add(fs::ieee_mul(2.f, high), low), 3.f);
    float aggregate_gt_pivot_0 = 0, aggregate_gt_pivot_1 = 0;
    min_gt_low = high;
    max_le_high = low;
    float threadlocal_aggregate_gt_pivot_0 = 0;
    float threadlocal_aggregate_gt_pivot_1 = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < flashinfer::ceil_div(d, BLOCK * VEC); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK + tx) * VEC < d) {
        probs_vec.cast_load(probs + row_idx * d + i * BLOCK * VEC + tx * VEC);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC; ++j) {
        float g0 = (probs_vec[j] > pivot_0) ? probs_vec[j] : 0;
        float g1 = (probs_vec[j] > pivot_1) ? probs_vec[j] : 0;
        if (probs_vec[j] > low && (i * BLOCK + tx) * VEC + j < d) {
          min_gt_low = min(min_gt_low, probs_vec[j]);
        }
        if (probs_vec[j] <= high && (i * BLOCK + tx) * VEC + j < d) {
          max_le_high = max(max_le_high, probs_vec[j]);
        }
        threadlocal_aggregate_gt_pivot_0 += g0;
        threadlocal_aggregate_gt_pivot_1 += g1;
      }
    }
    aggregate_gt_pivot_0 =
        cub::BlockReduce<float, BLOCK, fs::REDUCE_ALGO>(temp_storage.block_prim.reduce)
            .Sum(threadlocal_aggregate_gt_pivot_0);
    __syncthreads();
    aggregate_gt_pivot_1 =
        cub::BlockReduce<float, BLOCK, fs::REDUCE_ALGO>(temp_storage.block_prim.reduce)
            .Sum(threadlocal_aggregate_gt_pivot_1);
    __syncthreads();
    min_gt_low = cub::BlockReduce<float, BLOCK, fs::REDUCE_ALGO>(temp_storage.block_prim.reduce)
                     .Reduce(min_gt_low, cuda::minimum<>{});
    __syncthreads();
    max_le_high = cub::BlockReduce<float, BLOCK, fs::REDUCE_ALGO>(temp_storage.block_prim.reduce)
                      .Reduce(max_le_high, cuda::maximum<>{});
    if (tx == 0) {
      temp_storage.block_aggregate.values[0] = aggregate_gt_pivot_0;
      temp_storage.block_aggregate.values[1] = aggregate_gt_pivot_1;
      temp_storage.min_val = min_gt_low;
      temp_storage.max_val = max_le_high;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.values[0];
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.values[1];
    min_gt_low = temp_storage.min_val;
    max_le_high = temp_storage.max_val;
    if (aggregate_gt_pivot_1 >= p) {
      low = pivot_1;
      sum_low = aggregate_gt_pivot_1;
    } else if (aggregate_gt_pivot_0 >= p) {
      low = pivot_0;
      high = min(pivot_1, max_le_high);
      sum_low = aggregate_gt_pivot_0;
    } else {
      high = min(pivot_0, max_le_high);
    }
  } while (min_gt_low < max_le_high && nextafterf(min_gt_low, max_le_high) < max_le_high);

  float normalizer = flashinfer::math::ptx_rcp(max(sum_low, 1e-8));
#pragma unroll 2
  for (uint32_t i = 0; i < flashinfer::ceil_div(d, BLOCK * VEC); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK + tx) * VEC < d) {
      probs_vec.cast_load(probs + row_idx * d + i * BLOCK * VEC + tx * VEC);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC; ++j) {
      probs_vec[j] = (probs_vec[j] > low) ? probs_vec[j] * normalizer : 0;
    }
    if ((i * BLOCK + tx) * VEC < d) {
      probs_vec.cast_store(renormed_prob + row_idx * d + i * BLOCK * VEC + tx * VEC);
    }
  }
}

extern "C" __global__ FLASHINFER_SAMPLING_LAUNCH_BOUNDS(BLOCK) void kern_verify_minp_sample_f32(
    float* probs, float* min_p_arr, int* output, bool* valid, int* indices, float min_p_val,
    uint32_t d, uint64_t* seed_arr, uint64_t seed_val, uint64_t* offset_arr, uint64_t offset_val,
    uint32_t seed_stride, uint32_t offset_stride,
    const float* __restrict__ raw_temperature) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  if (raw_temperature[bx] <= 0.0f) return;

  uint64_t philox_seed = seed_arr ? seed_arr[bx * seed_stride] : seed_val;
  uint64_t philox_offset = offset_arr ? offset_arr[bx * offset_stride] : offset_val;

  curandStatePhilox4_32_10_t state;
  curand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
  float p = (min_p_arr == nullptr) ? min_p_val : min_p_arr[row_idx];

  extern __shared__ __align__(
      alignof(fs::SamplingTempStorage<BLOCK, fs::SCAN_ALGO, fs::REDUCE_ALGO>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<fs::SamplingTempStorage<BLOCK, fs::SCAN_ALGO, fs::REDUCE_ALGO>&>(
          smem_sampling);

  float max_val = fs::GetMaxValue<VEC, BLOCK, fs::REDUCE_ALGO,
                                  fs::SamplingTempStorage<BLOCK, fs::SCAN_ALGO, fs::REDUCE_ALGO>>(
      probs, row_idx, d, temp_storage);
  float pivot = max_val * p;

  flashinfer::vec_t<float, VEC> probs_vec;
  float aggregate_gt_pivot = 0;
  float threadlocal_aggregate_gt_pivot = 0;
#pragma unroll 2
  for (uint32_t i = 0; i < flashinfer::ceil_div(d, BLOCK * VEC); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK + tx) * VEC < d) {
      probs_vec.cast_load(probs + row_idx * d + (i * BLOCK + tx) * VEC);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC; ++j) {
      float g = (probs_vec[j] >= pivot) ? probs_vec[j] : 0;
      threadlocal_aggregate_gt_pivot += g;
    }
  }

  aggregate_gt_pivot += cub::BlockReduce<float, BLOCK>(temp_storage.block_prim.reduce)
                            .Sum(threadlocal_aggregate_gt_pivot);
  if (tx == 0) temp_storage.block_aggregate.value = aggregate_gt_pivot;
  __syncthreads();

  float aggregate = 0;
  float q = temp_storage.block_aggregate.value;
  int sampled_id;
  temp_storage.sampled_id = d;
  temp_storage.last_valid_id = -1;
  int last_valid_index = -1;
  __syncthreads();
  float u = curand_uniform(&state) * q;
#pragma unroll 2
  for (uint32_t i = 0; i < flashinfer::ceil_div(d, BLOCK * VEC); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK + tx) * VEC < d) {
      probs_vec.cast_load(probs + row_idx * d + (i * BLOCK + tx) * VEC);
    }
    fs::DeviceSamplingFromProb<VEC, BLOCK, fs::SCAN_ALGO, fs::REDUCE_ALGO, true>(
        i, d, [&](float x) { return x > 0.0f && x >= pivot; }, u, probs_vec, aggregate,
        last_valid_index, &temp_storage);
    if (aggregate > u) break;
  }
  sampled_id = temp_storage.sampled_id;
  if (sampled_id == d) {
    if (temp_storage.last_valid_id == -1) {
      if (tx == 0) {
        output[bx] = 0;
        valid[bx] = false;
      }
      return;
    }
    sampled_id = temp_storage.last_valid_id;
  }
  output[bx] = sampled_id;
  valid[bx] = true;
}
