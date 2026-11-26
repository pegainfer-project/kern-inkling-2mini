// SPDX-License-Identifier: Apache-2.0
// SM120 (Blackwell) Target Verify Attention Kernels:
//   1. kern_attn_swa_verify_split: Split-KV sliding-window (W=1024) FP8 E4M3 attention
//      over states.seq_state ring + KCur/VCur for speculative verify (QLEN=8, H=24, HKV=8, D=128).
//      - Transposed MMA geometry: K @ Q^T [64x24] and V^T @ P^T [128x24] (100% Tensor Core util)
//      - Hardware ldmatrix.m8n8.x4 (FP8 K) + ldmatrix.m16n16.x2.trans.b8 (permuted-swizzled FP8 V)
//      - Direct coalesced Q register load (zero SMEM Q buffer, zero prologue sync)
//      - 3-buffer KV SMEM (k[2]+v[1]) + separate rel & p buffers (~32.7 KB -> 3 CTAs/SM on SM120!)
//      - Early PDL K0+V0 prefetch from SeqState BEFORE griddepcontrol.wait (overlaps attn_prologue!)
//      - Blackwell native 1-instr cvt.rn.bf16x2.e4m3x2 (F2FP.BF16.E4M3) + L2::256B prefetch
//      - Optional dedicated cur-only split (ring_splits < num_splits) to eliminate last-split straggler
//   2. kern_attn_full_verify_split: 2-phase (pure-bulk dist>=1024 + causal/Rel tail)
//      paged FP8 E4M3 full attention for speculative verify (QLEN=8, PAGE=64 == BN=64).
//      - 3 CTAs/SM (REG<=168, STACK:0, SMEM:33280 B), hardware ldmatrix K/V + direct-reg Q
//      - Proportional dynamic split-K across sequences (grid=[total_splits, 8, 1], total_splits=41/40)
//        eliminating uneven-KV stragglers at seqs=4/8 and filling all 320-328 SM slots at seqs=1..8
//      - Early PDL K0+V0 prefetch for bulk pages BEFORE griddepcontrol.wait (overlaps attn_prologue!)
//   3. kern_attn_verify_combine_dyn (4-warp float4 PDL dynamic split-KV combine + Gate + Sink)
//      and kern_attn_verify_combine_s{4,5,8,9,10,16,17,20,32,41} (1-warp float4 PDL combine).

#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

using bf16 = __nv_bfloat16;

namespace {

constexpr int kHeads = 24;
constexpr int kKvHeads = 8;
constexpr int kGroup = 3;
constexpr int kHeadDim = 128;
constexpr int kRelExtent = 1024;
constexpr int kRing = 1024;
constexpr int kWindow = 1024;
constexpr int kQlen = 8;
constexpr int kM = kQlen * kGroup;  // 24 query rows per (seq, kv_head)
// dyn split tables: seqs > 8 overran sh_sp[8]/sh_np[8]; seqs must also stay <= total_splits (40)
constexpr int kMaxDynSeqs = 64;
static_assert(4 * kM >= kMaxDynSeqs, "split table must fit smem.m / smem.l");
constexpr int kBN = 64;             // keys per tile
constexpr int kThreads = 128;       // 4 warps
constexpr int kPRow = 72;           // 72 bf16 = 144 B = 36 banks (0 bank conflicts)

constexpr int64_t kSeqStateBytes = 90005376LL;
constexpr int64_t kSwaHeadRingBytes = 131072LL;   // 1024 * 128
constexpr int64_t kSwaHalfVisitBytes = 1048576LL; // 8 * 1024 * 128

constexpr int kPage = 64;                         // == kBN (1 tile == 1 full KV page!)
constexpr int64_t kKvPageBytes = 1179648LL;       // 9 * 2 * 8 * 64 * 128
constexpr int64_t kKvHalfPageBytes = 65536LL;     // 8 * 64 * 128
constexpr int64_t kKvHeadPageBytes = 8192LL;      // 64 * 128

constexpr float kLog2E = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
constexpr float kNegInf = -INFINITY;

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_cg_16(void* dst_smem, const void* src_gmem) {
  uint32_t dst = smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" :: "r"(dst), "l"(src_gmem) : "memory");
}

__device__ __forceinline__ void cp_async_cg_16_pred(void* dst_smem, const void* src_gmem, int src_size) {
  uint32_t dst = smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(dst), "l"(src_gmem), "r"(src_size) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ uint32_t cvt_e4m3x2_to_bf16x2(uint16_t x) {
  uint32_t res;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(res) : "h"(x));
  return res;
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
  uint32_t addr = smem_u32(smem_ptr);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans_b8(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
  uint32_t addr = smem_u32(smem_ptr);
  asm volatile(
      "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float (&c)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float lg2_approx(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + ex2_approx(-x * kLog2E));
}

__device__ __forceinline__ int swiz_c16(int row, int c16) {
  return c16 ^ (row & 7);
}

// Load 64x128 FP8 ring tile (K or V) from SeqState into dst[512] uint4 with 16B XOR swizzle.
template <bool PERM_V>
__device__ __forceinline__ void cp_async_ring_tile(
    uint4* __restrict__ dst,
    const uint8_t* __restrict__ ring_base,
    int k0,
    int past,
    int tid,
    bool full_valid) {
  (void)PERM_V;
  const int c16 = tid & 7;
  const int r0 = tid >> 3;
  const int col_byte = c16 << 4;
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    const int r = r0 + step * 16;
    const int k = k0 + r;
    const int slot = k & (kRing - 1);
    const uint8_t* src = ring_base + static_cast<int64_t>(slot) * kHeadDim + col_byte;
    if (full_valid) {
      cp_async_cg_16(&dst[r * 8 + swiz_c16(r, c16)], src);
    } else {
      const int sz = (k >= 0 && k < past) ? 16 : 0;
      cp_async_cg_16_pred(&dst[r * 8 + swiz_c16(r, c16)], src, sz);
    }
  }
}

// Load 64x128 FP8 paged KV tile (K or V) from KVState into dst[512] uint4 with 16B XOR swizzle.
template <bool IS_V, bool PERM_V>
__device__ __forceinline__ void cp_async_page_tile(
    uint4* __restrict__ dst,
    const uint8_t* __restrict__ KVState,
    int kv_head,
    int page,
    int k0,
    int seqlen_k,
    int tid,
    bool full_valid) {
  (void)PERM_V;
  const int c16 = tid & 7;
  const int r0 = tid >> 3;
  const int col_byte = c16 << 4;
  const bool page_ok = (page >= 0);
  const int64_t safe_page = page_ok ? static_cast<int64_t>(page) : 0LL;
  const uint8_t* base = KVState
      + safe_page * kKvPageBytes
      + static_cast<int64_t>(kv_head) * kKvHeadPageBytes
      + (IS_V ? kKvHalfPageBytes : 0LL)
      + col_byte;
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    const int r = r0 + step * 16;
    const uint8_t* src = base + static_cast<int64_t>(r) * kHeadDim;
    if (full_valid && page_ok) {
      cp_async_cg_16(&dst[r * 8 + swiz_c16(r, c16)], src);
    } else {
      const int sz = (page_ok && (k0 + r) < seqlen_k) ? 16 : 0;
      cp_async_cg_16_pred(&dst[r * 8 + swiz_c16(r, c16)], src, sz);
    }
  }
}

// Load Rel[q_start..q_start+7, kv_head*3..+2, d_align..d_align+71] into dst[24][72].
__device__ __forceinline__ void cp_async_rel_tile(
    bf16 (&dst)[kM][kPRow],
    const bf16* __restrict__ Rel,
    int q_start,
    int kv_head,
    int past,
    int k0,
    int tid,
    bool full_interior) {
#pragma unroll
  for (int idx = tid; idx < kM * 9; idx += kThreads) {
    const int m = idx / 9;
    const int chunk = idx - m * 9;
    const int tok = m / kGroup;
    const int head = kv_head * kGroup + (m - tok * kGroup);
    const int d_lo = (past + tok) - (k0 + kBN - 1);
    const int d_align = d_lo & ~7;
    const int d_c = d_align + (chunk << 3);
    const int64_t row_off = static_cast<int64_t>(q_start + tok) * (kHeads * kRelExtent)
                          + static_cast<int64_t>(head) * kRelExtent;
    if (full_interior) {
      if (chunk < 8 || (d_lo & 7) != 0) {
        cp_async_cg_16(&dst[m][chunk << 3], Rel + row_off + d_c);
      }
    } else {
      const bool ok = (static_cast<unsigned>(d_c) < static_cast<unsigned>(kRelExtent));
      const int safe_dc = ok ? d_c : 0;
      cp_async_cg_16_pred(&dst[m][chunk << 3], Rel + row_off + safe_dc, ok ? 16 : 0);
    }
  }
}

// Load Q[24][128] BF16 directly into registers matching ldmatrix_x4 K layout (d = ds*16 + tig*4..+3).
__device__ __forceinline__ void load_q_direct_reg(
    uint32_t (&q_b0)[8][3],
    uint32_t (&q_b1)[8][3],
    const bf16* __restrict__ Q,
    int q_start,
    int kv_head,
    int gid,
    int tig) {
#pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    const int m = mt * 8 + gid;
    const int tok = m / kGroup;
    const int head = kv_head * kGroup + (m - tok * kGroup);
    const bf16* q_row = Q + (static_cast<int64_t>(q_start + tok) * kHeads + head) * kHeadDim;
#pragma unroll
    for (int ds = 0; ds < 8; ++ds) {
      const uint2 qv = *reinterpret_cast<const uint2*>(q_row + ds * 16 + tig * 4);
      q_b0[ds][mt] = qv.x;
      q_b1[ds][mt] = qv.y;
    }
  }
}

// Compute S^T = K[64x128] @ Q^T[128x24] for warp w (keys w*16..w*16+15) via 4 hardware ldmatrix_x4!
__device__ __forceinline__ void compute_kq_tile_ldm(
    float (&s)[3][4],
    const uint4* __restrict__ smem_k,
    const uint32_t (&q_b0)[8][3],
    const uint32_t (&q_b1)[8][3],
    int w,
    int lane) {
#pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    s[mt][0] = 0.0f; s[mt][1] = 0.0f; s[mt][2] = 0.0f; s[mt][3] = 0.0f;
  }
  const int k_ld_row = w * 16 + ((lane >> 4) << 3) + (lane & 7);
  const int k_ld_c16_off = (lane >> 3) & 1;

#pragma unroll
  for (int ds2 = 0; ds2 < 4; ++ds2) {
    uint32_t b0, b1, b2, b3;
    ldmatrix_x4(b0, b1, b2, b3, &smem_k[k_ld_row * 8 + swiz_c16(k_ld_row, ds2 * 2 + k_ld_c16_off)]);
    const int ds_lo = ds2 * 2;
    const int ds_hi = ds_lo + 1;
    uint32_t a0 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b0));
    uint32_t a1 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b2));
    uint32_t a2 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b0 >> 16));
    uint32_t a3 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b2 >> 16));
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      mma_m16n8k16_bf16(s[mt], a0, a1, a2, a3, q_b0[ds_lo][mt], q_b1[ds_lo][mt]);
    }
    uint32_t c0 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b1));
    uint32_t c1 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b3));
    uint32_t c2 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b1 >> 16));
    uint32_t c3 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b3 >> 16));
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      mma_m16n8k16_bf16(s[mt], c0, c1, c2, c3, q_b0[ds_hi][mt], q_b1[ds_hi][mt]);
    }
  }
}

// Accumulate O^T += V^T[128xK] @ P^T[Kx24] for warp w (dims w*32..w*32+31) via hardware ldmatrix_x2_trans_b8!
template <int K_STEPS>
__device__ __forceinline__ void compute_vp_tile_ldm(
    float (&acc)[2][3][4],
    const uint4* __restrict__ smem_v,
    const bf16 (&smem_p)[kM][kPRow],
    int w,
    int lane,
    int gid,
    int tig) {
  const int v_dp = w * 2 + (lane >> 4);
  const int v_r_off = lane & 15;
#pragma unroll
  for (int ks = 0; ks < K_STEPS; ++ks) {
    const int k_base = ks * 16;
    const int vr = k_base + v_r_off;
    uint32_t vb0, vb1, vb2, vb3;
    ldmatrix_x2_trans_b8(vb0, vb1, vb2, vb3, &smem_v[vr * 8 + swiz_c16(vr, v_dp)]);

    uint32_t b0[3], b1[3];
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m = mt * 8 + gid;
      // ldmatrix.m16n16.x2.trans.b8 returns contiguous keys {tig*4..tig*4+3} -> single 64-bit load!
      const uint2 pv = *reinterpret_cast<const uint2*>(&smem_p[m][k_base + tig * 4]);
      b0[mt] = pv.x;
      b1[mt] = pv.y;
    }

    uint32_t a0 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb0));
    uint32_t a1 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb1));
    uint32_t a2 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb0 >> 16));
    uint32_t a3 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb1 >> 16));
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      mma_m16n8k16_bf16(acc[0][mt], a0, a1, a2, a3, b0[mt], b1[mt]);
    }

    uint32_t c0 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb2));
    uint32_t c1 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb3));
    uint32_t c2 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb2 >> 16));
    uint32_t c3 = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb3 >> 16));
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      mma_m16n8k16_bf16(acc[1][mt], c0, c1, c2, c3, b0[mt], b1[mt]);
    }
  }
}

__device__ __forceinline__ float col_reduce_max8(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
  return v;
}

__device__ __forceinline__ float col_reduce_sum8(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  return v;
}

// Deterministic exact-integer proportional split allocation across seqs (1..kMaxDynSeqs) into SMEM sh_sp, sh_np.
// Guarantees: sum(sh_sp) == total_splits (when any seq active), uniform seqs get exact floor/ceil equal splits,
// skewed seqs get splits strictly proportional to page counts (fixing uneven-KV SOL collapse!).
__device__ __forceinline__ void compute_dyn_splits_smem(
    const int32_t* __restrict__ SeqLens,
    int seqs,
    int total_splits,
    int* __restrict__ sh_sp,
    int* __restrict__ sh_np) {
  int sum_p = 0;
  int active = 0;
  for (int r = 0; r < seqs; ++r) {
    const int sl = SeqLens[r];
    const int pages = (sl > 0) ? ((sl + kPage - 1) >> 6) : 0;
    sh_np[r] = pages;
    sh_sp[r] = (pages > 0) ? 1 : 0;
    sum_p += pages;
    active += (pages > 0) ? 1 : 0;
  }
  const int rem_s = total_splits - active;
  if (rem_s > 0 && sum_p > 0) {
    int used = 0;
    for (int r = 0; r < seqs; ++r) {
      if (sh_np[r] > 0) {
        const int add = (rem_s * sh_np[r]) / sum_p;
        sh_sp[r] += add;
        used += add;
      }
    }
    int left = rem_s - used;
    while (left > 0) {
      int best = -1;
      for (int r = 0; r < seqs; ++r) {
        if (sh_np[r] <= 0) continue;
        if (best < 0 ||
            static_cast<int64_t>(sh_np[r]) * sh_sp[best] > static_cast<int64_t>(sh_np[best]) * sh_sp[r]) {
          best = r;
        }
      }
      if (best < 0) break;
      sh_sp[best] += 1;
      --left;
    }
  }
}

// Total SMEM = 32,256 B (+1 KB runtime = 33,280 B -> 3 CTAs/SM = 99,840 B <= 101,376 B on SM120!)
struct SwaVerifySmem {
  alignas(16) uint4 k[2][kBN * 8];          // 16,384 B (reused for O epilogue staging)
  alignas(16) uint4 v[kBN * 8];             //  8,192 B
  alignas(16) bf16 rel[kM][kPRow];          //  3,456 B (separate from p so Rel_{it+1} prefetches with K_{it+1}!)
  alignas(16) bf16 p[kM][kPRow];            //  3,456 B
  alignas(16) float m[4][kM];               //    384 B (4-warp rowmax scratch; also holds dyn split table before loop)
  alignas(16) float l[4][kM];               //    384 B (4-warp rowsum scratch)
};

}  // namespace

// ---------------------------------------------------------------------------
// 1. SWA Verify Split-KV Attention Kernel (SM120, 3 CTAs/SM, PDL + ldmatrix)
// Grid: [num_splits, 8 (kv_heads), seqs], Block: [128, 1, 1]
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(kThreads, 3) kern_attn_swa_verify_split(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ SeqState,
    const uint8_t* __restrict__ KCur,
    const uint8_t* __restrict__ VCur,
    const bf16* __restrict__ Rel,
    const int32_t* __restrict__ Lines,
    const int32_t* __restrict__ SeqLens,
    float* __restrict__ OPart,
    float* __restrict__ LsePart,
    int num_tokens,
    float sm_scale,
    float v_scale,
    int num_splits,
    int ring_splits) {
  const int split = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int req = blockIdx.z;
  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;
  const int tig = lane & 3;

  const int q_start = req * kQlen;
  const int line = Lines[req];
  const int seqlen_k = SeqLens[req];

  if (line <= 0 || seqlen_k <= 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (tid < kM) {
      const int tok = tid / kGroup;
      const int head = kv_head * kGroup + (tid - tok * kGroup);
      const int64_t part = (static_cast<int64_t>(split) * num_tokens + (q_start + tok)) * kHeads + head;
      LsePart[part] = kNegInf;
    }
    return;
  }

  const int past = seqlen_k - kQlen;
  const int k_min = (past - (kWindow - 1) > 0) ? (past - (kWindow - 1)) : 0;
  const int n_ring = (past > k_min) ? (past - k_min) : 0;
  const int n_ring_tiles = (n_ring + kBN - 1) / kBN;

  int tile_start = 0;
  int tile_count = 0;
  bool do_cur = false;
  if (split < ring_splits) {
    const int base_tiles = n_ring_tiles / ring_splits;
    const int rem_tiles = n_ring_tiles % ring_splits;
    tile_start = split * base_tiles + (split < rem_tiles ? split : rem_tiles);
    tile_count = base_tiles + (split < rem_tiles ? 1 : 0);
    do_cur = (ring_splits == num_splits) && (split == num_splits - 1);
  } else {
    do_cur = (split == num_splits - 1);
  }

  if (tile_count == 0 && !do_cur) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (tid < kM) {
      const int tok = tid / kGroup;
      const int head = kv_head * kGroup + (tid - tok * kGroup);
      const int64_t part = (static_cast<int64_t>(split) * num_tokens + (q_start + tok)) * kHeads + head;
      LsePart[part] = kNegInf;
    }
    return;
  }

  __shared__ SwaVerifySmem smem;

  const uint8_t* ring_k_base = SeqState
      + static_cast<int64_t>(line) * kSeqStateBytes
      + static_cast<int64_t>(kv_head) * kSwaHeadRingBytes;
  const uint8_t* ring_v_base = ring_k_base + kSwaHalfVisitBytes;

  // Prefetch BOTH first ring K and V tiles into smem BEFORE griddepcontrol.wait (SeqState untouched by prologue!).
  int k0_first = 0;
  bool full_first = false;
  if (tile_count > 0) {
    k0_first = k_min + tile_start * kBN;
    full_first = (k0_first + kBN <= past) && (past + 7 - k0_first < kWindow);
    cp_async_ring_tile<false>(smem.k[0], ring_k_base, k0_first, past, tid, full_first);
    cp_async_ring_tile<true>(smem.v, ring_v_base, k0_first, past, tid, full_first);
    cp_async_commit();
  }

  // Wait for attn_prologue_swa_verify to finish writing Q, Rel, KCur, VCur.
  asm volatile("griddepcontrol.wait;" ::: "memory");

  // Kick off Rel_0 cp.async in parallel with direct Q register load (zero smem_q, zero prologue sync!).
  if (tile_count > 0) {
    cp_async_rel_tile(smem.rel, Rel, q_start, kv_head, past, k0_first, tid, full_first);
    cp_async_commit();
  }

  uint32_t q_b0[8][3];
  uint32_t q_b1[8][3];
  load_q_direct_reg(q_b0, q_b1, Q, q_start, kv_head, gid, tig);

  const float qk_scale = sm_scale * kLog2E;

  float m_i[3][2];
  float l_i[3][2];
  float acc[2][3][4];
#pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    m_i[mt][0] = kNegInf; m_i[mt][1] = kNegInf;
    l_i[mt][0] = 0.0f;    l_i[mt][1] = 0.0f;
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
      acc[dt][mt][0] = 0.0f; acc[dt][mt][1] = 0.0f;
      acc[dt][mt][2] = 0.0f; acc[dt][mt][3] = 0.0f;
    }
  }

  const int k_local0 = w * 16 + gid;
  const int k_local1 = k_local0 + 8;
  const int rev0 = (kBN - 1) - k_local0;
  const int rev1 = (kBN - 1) - k_local1;

  // ---------------- Main Ring Loop (BN=64, hardware ldmatrix K/V + overlapped K+Rel / V) ----------------
  for (int it = 0; it < tile_count; ++it) {
    const int cur = it & 1;
    const int nxt = cur ^ 1;
    const int k0 = k_min + (tile_start + it) * kBN;
    const bool is_full_interior = (k0 + kBN <= past) && (past + 7 - k0 < kWindow);

    if (it == 0) {
      // K_0 + V_0 (started during prologue!) and Rel_0 are both in flight: wait for all.
      cp_async_wait<0>();
    } else {
      // Issue V_it while K_it + Rel_it finish and K_it @ Q^T runs!
      cp_async_ring_tile<true>(smem.v, ring_v_base, k0, past, tid, is_full_interior);
      cp_async_commit();
      cp_async_wait<1>();
    }
    __syncthreads();

    float s[3][4];
    compute_kq_tile_ldm(s, smem.k[cur], q_b0, q_b1, w, lane);

    const int k_abs0 = k0 + k_local0;
    const int k_abs1 = k0 + k_local1;

#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const int tok0 = m0 / kGroup;
      const int tok1 = m1 / kGroup;
      const int sh0 = ((past + tok0) - (k0 + kBN - 1)) & 7;
      const int sh1 = ((past + tok1) - (k0 + kBN - 1)) & 7;

      if (is_full_interior) {
        const float r00 = __bfloat162float(smem.rel[m0][rev0 + sh0]);
        const float r01 = __bfloat162float(smem.rel[m1][rev0 + sh1]);
        const float r10 = __bfloat162float(smem.rel[m0][rev1 + sh0]);
        const float r11 = __bfloat162float(smem.rel[m1][rev1 + sh1]);
        s[mt][0] = __fmaf_rn(s[mt][0], qk_scale, r00 * kLog2E);
        s[mt][1] = __fmaf_rn(s[mt][1], qk_scale, r01 * kLog2E);
        s[mt][2] = __fmaf_rn(s[mt][2], qk_scale, r10 * kLog2E);
        s[mt][3] = __fmaf_rn(s[mt][3], qk_scale, r11 * kLog2E);
      } else {
        const int dist00 = (past + tok0) - k_abs0;
        const int dist01 = (past + tok1) - k_abs0;
        const int dist10 = (past + tok0) - k_abs1;
        const int dist11 = (past + tok1) - k_abs1;
        const bool v00 = (k_abs0 >= 0) && (k_abs0 < past) && (dist00 >= 0) && (dist00 < kWindow);
        const bool v01 = (k_abs0 >= 0) && (k_abs0 < past) && (dist01 >= 0) && (dist01 < kWindow);
        const bool v10 = (k_abs1 >= 0) && (k_abs1 < past) && (dist10 >= 0) && (dist10 < kWindow);
        const bool v11 = (k_abs1 >= 0) && (k_abs1 < past) && (dist11 >= 0) && (dist11 < kWindow);
        s[mt][0] = v00 ? __fmaf_rn(s[mt][0], qk_scale, __bfloat162float(smem.rel[m0][rev0 + sh0]) * kLog2E) : kNegInf;
        s[mt][1] = v01 ? __fmaf_rn(s[mt][1], qk_scale, __bfloat162float(smem.rel[m1][rev0 + sh1]) * kLog2E) : kNegInf;
        s[mt][2] = v10 ? __fmaf_rn(s[mt][2], qk_scale, __bfloat162float(smem.rel[m0][rev1 + sh0]) * kLog2E) : kNegInf;
        s[mt][3] = v11 ? __fmaf_rn(s[mt][3], qk_scale, __bfloat162float(smem.rel[m1][rev1 + sh1]) * kLog2E) : kNegInf;
      }

      const float wm0 = col_reduce_max8(fmaxf(s[mt][0], s[mt][2]));
      const float wm1 = col_reduce_max8(fmaxf(s[mt][1], s[mt][3]));
      if (gid == 0) {
        smem.m[w][m0] = wm0;
        smem.m[w][m1] = wm1;
      }
    }
    __syncthreads();

    float alpha[3][2];
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const float tm0 = fmaxf(fmaxf(smem.m[0][m0], smem.m[1][m0]), fmaxf(smem.m[2][m0], smem.m[3][m0]));
      const float tm1 = fmaxf(fmaxf(smem.m[0][m1], smem.m[1][m1]), fmaxf(smem.m[2][m1], smem.m[3][m1]));
      const float mn0 = fmaxf(m_i[mt][0], tm0);
      const float mn1 = fmaxf(m_i[mt][1], tm1);
      const float ms0 = (mn0 == kNegInf) ? 0.0f : mn0;
      const float ms1 = (mn1 == kNegInf) ? 0.0f : mn1;
      alpha[mt][0] = ex2_approx(m_i[mt][0] - ms0);
      alpha[mt][1] = ex2_approx(m_i[mt][1] - ms1);
      m_i[mt][0] = mn0;
      m_i[mt][1] = mn1;

      const float p00 = ex2_approx(s[mt][0] - ms0);
      const float p01 = ex2_approx(s[mt][1] - ms1);
      const float p10 = ex2_approx(s[mt][2] - ms0);
      const float p11 = ex2_approx(s[mt][3] - ms1);

      smem.p[m0][k_local0] = __float2bfloat16_rn(p00);
      smem.p[m1][k_local0] = __float2bfloat16_rn(p01);
      smem.p[m0][k_local1] = __float2bfloat16_rn(p10);
      smem.p[m1][k_local1] = __float2bfloat16_rn(p11);

      const float wl0 = col_reduce_sum8(p00 + p10);
      const float wl1 = col_reduce_sum8(p01 + p11);
      if (gid == 0) {
        smem.l[w][m0] = wl0;
        smem.l[w][m1] = wl1;
      }
    }

    if (it + 1 < tile_count) {
      const int k0_nxt = k_min + (tile_start + it + 1) * kBN;
      const bool full_nxt = (k0_nxt + kBN <= past) && (past + 7 - k0_nxt < kWindow);
      cp_async_ring_tile<false>(smem.k[nxt], ring_k_base, k0_nxt, past, tid, full_nxt);
      cp_async_rel_tile(smem.rel, Rel, q_start, kv_head, past, k0_nxt, tid, full_nxt);
      cp_async_commit();
      if (it > 0) cp_async_wait<1>();
    } else if (it > 0) {
      cp_async_wait<0>();
    }
    __syncthreads();

#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const float sum0 = (smem.l[0][m0] + smem.l[1][m0]) + (smem.l[2][m0] + smem.l[3][m0]);
      const float sum1 = (smem.l[0][m1] + smem.l[1][m1]) + (smem.l[2][m1] + smem.l[3][m1]);
      l_i[mt][0] = __fmaf_rn(l_i[mt][0], alpha[mt][0], sum0);
      l_i[mt][1] = __fmaf_rn(l_i[mt][1], alpha[mt][1], sum1);
#pragma unroll
      for (int dt = 0; dt < 2; ++dt) {
        acc[dt][mt][0] *= alpha[mt][0];
        acc[dt][mt][1] *= alpha[mt][1];
        acc[dt][mt][2] *= alpha[mt][0];
        acc[dt][mt][3] *= alpha[mt][1];
      }
    }

    compute_vp_tile_ldm<4>(acc, smem.v, smem.p, w, lane, gid, tig);

    if (it + 1 < tile_count || do_cur) {
      __syncthreads();
    }
  }

  // ---------------- 8-key Cur Micro-Tile (ldmatrix K/V!) ----------------
  if (do_cur) {
    const int r = tid >> 3;          // 0..15
    const int c16 = tid & 7;         // 0..7
    const int col_byte = c16 << 4;
    const int sw = swiz_c16(r, c16);
    if (r < kQlen) {
      const int k_real = past + r;
      const bool valid_k = (k_real >= 0 && k_real < seqlen_k);
      const int64_t cur_off = (static_cast<int64_t>(q_start + r) * kKvHeads + kv_head) * kHeadDim + col_byte;
      cp_async_cg_16_pred(&smem.k[0][r * 8 + sw], KCur + cur_off, valid_k ? 16 : 0);
      cp_async_cg_16_pred(&smem.v[r * 8 + sw], VCur + cur_off, valid_k ? 16 : 0);
    } else {
      smem.k[0][r * 8 + sw] = make_uint4(0u, 0u, 0u, 0u);
      smem.v[r * 8 + sw] = make_uint4(0u, 0u, 0u, 0u);
    }
    if (tid < kM) {
      const int tok = tid / kGroup;
      const int head = kv_head * kGroup + (tid - tok * kGroup);
      const int64_t row_off = static_cast<int64_t>(q_start + tok) * (kHeads * kRelExtent)
                            + static_cast<int64_t>(head) * kRelExtent;
      cp_async_cg_16(&smem.rel[tid][0], Rel + row_off);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    if (w == 0) {
      float s[3][4];
      compute_kq_tile_ldm(s, smem.k[0], q_b0, q_b1, 0, lane);

      const int k_real = past + gid;
      const bool k_ok = (k_real >= 0 && k_real < seqlen_k);
#pragma unroll
      for (int mt = 0; mt < 3; ++mt) {
        const int m0 = mt * 8 + tig * 2;
        const int m1 = m0 + 1;
        const int d0 = (m0 / kGroup) - gid;
        const int d1 = (m1 / kGroup) - gid;
        const bool v0 = k_ok && (d0 >= 0);
        const bool v1 = k_ok && (d1 >= 0);
        const float r0 = v0 ? __bfloat162float(smem.rel[m0][d0]) : 0.0f;
        const float r1 = v1 ? __bfloat162float(smem.rel[m1][d1]) : 0.0f;
        const float s0 = v0 ? __fmaf_rn(s[mt][0], qk_scale, r0 * kLog2E) : kNegInf;
        const float s1 = v1 ? __fmaf_rn(s[mt][1], qk_scale, r1 * kLog2E) : kNegInf;

        const float tm0 = col_reduce_max8(s0);
        const float tm1 = col_reduce_max8(s1);
        const float mn0 = fmaxf(m_i[mt][0], tm0);
        const float mn1 = fmaxf(m_i[mt][1], tm1);
        const float ms0 = (mn0 == kNegInf) ? 0.0f : mn0;
        const float ms1 = (mn1 == kNegInf) ? 0.0f : mn1;

        const float p0 = ex2_approx(s0 - ms0);
        const float p1 = ex2_approx(s1 - ms1);
        const float sum0 = col_reduce_sum8(p0);
        const float sum1 = col_reduce_sum8(p1);

        if (gid == 0) {
          smem.m[0][m0] = tm0;
          smem.m[0][m1] = tm1;
          smem.l[0][m0] = sum0;
          smem.l[0][m1] = sum1;
        }
        smem.p[m0][gid] = __float2bfloat16_rn(p0);
        smem.p[m1][gid] = __float2bfloat16_rn(p1);
        smem.p[m0][gid + 8] = __float2bfloat16_rn(0.0f);
        smem.p[m1][gid + 8] = __float2bfloat16_rn(0.0f);
      }
    }
    __syncthreads();

#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const float tm0 = smem.m[0][m0];
      const float tm1 = smem.m[0][m1];
      const float sum0 = smem.l[0][m0];
      const float sum1 = smem.l[0][m1];
      const float mn0 = fmaxf(m_i[mt][0], tm0);
      const float mn1 = fmaxf(m_i[mt][1], tm1);
      const float ms0 = (mn0 == kNegInf) ? 0.0f : mn0;
      const float ms1 = (mn1 == kNegInf) ? 0.0f : mn1;
      const float a0 = ex2_approx(m_i[mt][0] - ms0);
      const float a1 = ex2_approx(m_i[mt][1] - ms1);
      m_i[mt][0] = mn0;
      m_i[mt][1] = mn1;
      l_i[mt][0] = __fmaf_rn(l_i[mt][0], a0, sum0);
      l_i[mt][1] = __fmaf_rn(l_i[mt][1], a1, sum1);
#pragma unroll
      for (int dt = 0; dt < 2; ++dt) {
        acc[dt][mt][0] *= a0;
        acc[dt][mt][1] *= a1;
        acc[dt][mt][2] *= a0;
        acc[dt][mt][3] *= a1;
      }
    }
    compute_vp_tile_ldm<1>(acc, smem.v, smem.p, w, lane, gid, tig);
  }

  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  __syncthreads();
  float (*smem_o)[kHeadDim] = reinterpret_cast<float (*)[kHeadDim]>(&smem.k[0][0]);

#pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    const int m0 = mt * 8 + tig * 2;
    const int m1 = m0 + 1;
    const float inv0 = (l_i[mt][0] == 0.0f) ? 0.0f : (v_scale / l_i[mt][0]);
    const float inv1 = (l_i[mt][1] == 0.0f) ? 0.0f : (v_scale / l_i[mt][1]);
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
      const int d0 = w * 32 + dt * 16 + gid;
      const int d1 = d0 + 8;
      smem_o[m0][d0] = acc[dt][mt][0] * inv0;
      smem_o[m1][d0] = acc[dt][mt][1] * inv1;
      smem_o[m0][d1] = acc[dt][mt][2] * inv0;
      smem_o[m1][d1] = acc[dt][mt][3] * inv1;
    }
    if (w == 0 && gid == 0) {
      const int tok0 = m0 / kGroup;
      const int tok1 = m1 / kGroup;
      const int h0 = kv_head * kGroup + (m0 - tok0 * kGroup);
      const int h1 = kv_head * kGroup + (m1 - tok1 * kGroup);
      const float lse0 = (l_i[mt][0] == 0.0f) ? kNegInf : (m_i[mt][0] + lg2_approx(l_i[mt][0]));
      const float lse1 = (l_i[mt][1] == 0.0f) ? kNegInf : (m_i[mt][1] + lg2_approx(l_i[mt][1]));
      LsePart[(static_cast<int64_t>(split) * num_tokens + (q_start + tok0)) * kHeads + h0] = lse0;
      LsePart[(static_cast<int64_t>(split) * num_tokens + (q_start + tok1)) * kHeads + h1] = lse1;
    }
  }
  __syncthreads();

#pragma unroll
  for (int step = 0; step < 6; ++step) {
    const int idx = tid + step * kThreads;
    const int m = idx >> 5;
    const int d4 = (idx & 31) << 2;
    const int tok = m / kGroup;
    const int head = kv_head * kGroup + (m - tok * kGroup);
    const int64_t part = (static_cast<int64_t>(split) * num_tokens + (q_start + tok)) * kHeads + head;
    *reinterpret_cast<float4*>(OPart + part * kHeadDim + d4) =
        *reinterpret_cast<const float4*>(&smem_o[m][d4]);
  }
}

// ---------------------------------------------------------------------------
// 2. Full Attention Verify Dynamic Split-KV Kernel (SM120, 3 CTAs/SM, PDL + ldmatrix)
// Grid: [total_splits, 8 (kv_heads), 1], Block: [128, 1, 1]
// Proportionally allocates total_splits (41 for seqs==1, 40 for seqs>=2) across
// active sequences in proportion to page count (eliminating uneven-KV stragglers!).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(kThreads, 3) kern_attn_full_verify_split(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KVState,
    const bf16* __restrict__ Rel,
    const int32_t* __restrict__ BlockTable,
    const int32_t* __restrict__ SeqLens,
    float* __restrict__ OPart,
    float* __restrict__ LsePart,
    int num_tokens,
    int table_width,
    float sm_scale,
    float v_scale,
    int total_splits) {
  const int gsplit = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;
  const int tig = lane & 3;

  __shared__ SwaVerifySmem smem;

  const int seqs = num_tokens >> 3;
  int req = 0;
  int split = gsplit;
  int num_splits = total_splits;
  int n_pages = 0;

  if (seqs <= 1) {
    const int sl = SeqLens[0];
    n_pages = (sl > 0) ? ((sl + kPage - 1) >> 6) : 0;
  } else {
    int* sh_sp = reinterpret_cast<int*>(&smem.m[0][0]);
    int* sh_np = reinterpret_cast<int*>(&smem.l[0][0]);  // m and l each hold 4*kM = 96 ints
    if (tid == 0) {
      compute_dyn_splits_smem(SeqLens, seqs, total_splits, sh_sp, sh_np);
    }
    __syncthreads();
    int rem = gsplit;
    req = -1;
    for (int r = 0; r < seqs; ++r) {
      const int sp_r = sh_sp[r];
      if (rem < sp_r) {
        req = r;
        split = rem;
        num_splits = sp_r;
        n_pages = sh_np[r];
        break;
      }
      rem -= sp_r;
    }
    if (req < 0) {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
      return;
    }
  }

  const int q_start = req * kQlen;
  const int seqlen_k = SeqLens[req];

  const int base_pages = (num_splits > 0) ? (n_pages / num_splits) : 0;
  const int rem_pages = (num_splits > 0) ? (n_pages % num_splits) : 0;
  const int page_start = split * base_pages + (split < rem_pages ? split : rem_pages);
  const int page_count = base_pages + (split < rem_pages ? 1 : 0);
  const int page_end = page_start + page_count;

  if (page_count <= 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (tid < kM) {
      const int tok = tid / kGroup;
      const int head = kv_head * kGroup + (tid - tok * kGroup);
      const int64_t part = (static_cast<int64_t>(split) * num_tokens + (q_start + tok)) * kHeads + head;
      LsePart[part] = kNegInf;
    }
    return;
  }

  const int past = seqlen_k - kQlen;
  const int n_bulk_pages = (past >= kRelExtent - 1) ? ((past - (kRelExtent - 1)) >> 6) : 0;
  const int bulk_end = (page_end < n_bulk_pages) ? page_end : n_bulk_pages;
  const int bulk_count = (bulk_end > page_start) ? (bulk_end - page_start) : 0;
  const int tail_start = (page_start > n_bulk_pages) ? page_start : n_bulk_pages;
  const int tail_count = (page_end > tail_start) ? (page_end - tail_start) : 0;

  const int32_t* bt = BlockTable + static_cast<int64_t>(req) * table_width;
  int cur_page = bt[page_start];

  // For bulk splits (page_start < n_bulk_pages, strictly >= 16 pages older than newest page written by prologue!),
  // prefetch BOTH K_0 and V_0 BEFORE griddepcontrol.wait to overlap DRAM latency with attn_prologue_full_verify!
  if (bulk_count > 0) {
    cp_async_page_tile<false, false>(smem.k[0], KVState, kv_head, cur_page, page_start << 6, seqlen_k, tid, true);
    cp_async_page_tile<true, true>(smem.v, KVState, kv_head, cur_page, page_start << 6, seqlen_k, tid, true);
    cp_async_commit();
  }

  // Wait for attn_prologue_full_verify to finish writing Q, Rel, and newest KV page.
  asm volatile("griddepcontrol.wait;" ::: "memory");

  if (bulk_count == 0) {
    cp_async_page_tile<false, false>(smem.k[0], KVState, kv_head, cur_page, tail_start << 6, seqlen_k, tid, false);
    cp_async_rel_tile(smem.rel, Rel, q_start, kv_head, past, tail_start << 6, tid, false);
    cp_async_commit();
  }

  uint32_t q_b0[8][3];
  uint32_t q_b1[8][3];
  load_q_direct_reg(q_b0, q_b1, Q, q_start, kv_head, gid, tig);

  const float qk_scale = sm_scale * kLog2E;

  float m_i[3][2];
  float l_i[3][2];
  float acc[2][3][4];
#pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    m_i[mt][0] = kNegInf; m_i[mt][1] = kNegInf;
    l_i[mt][0] = 0.0f;    l_i[mt][1] = 0.0f;
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
      acc[dt][mt][0] = 0.0f; acc[dt][mt][1] = 0.0f;
      acc[dt][mt][2] = 0.0f; acc[dt][mt][3] = 0.0f;
    }
  }

  const int k_local0 = w * 16 + gid;
  const int k_local1 = k_local0 + 8;

  // ---------------- Phase A: Ultra-Tight Pure-Bulk Loop (dist >= 1024, ZERO Rel, ZERO Mask) ----------------
  for (int it = 0; it < bulk_count; ++it) {
    const int cur = it & 1;
    const int nxt = cur ^ 1;
    const int p_idx = page_start + it;
    const int page = cur_page;

    if (it == 0) {
      // K_0 + V_0 were prefetched together before griddepcontrol.wait!
      cp_async_wait<0>();
    } else {
      cp_async_page_tile<true, true>(smem.v, KVState, kv_head, page, p_idx << 6, seqlen_k, tid, true);
      cp_async_commit();
      cp_async_wait<1>();
    }
    __syncthreads();

    float s[3][4];
    compute_kq_tile_ldm(s, smem.k[cur], q_b0, q_b1, w, lane);

    const bool page_ok = (page >= 0);
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const float wm0 = page_ok ? (col_reduce_max8(fmaxf(s[mt][0], s[mt][2])) * qk_scale) : kNegInf;
      const float wm1 = page_ok ? (col_reduce_max8(fmaxf(s[mt][1], s[mt][3])) * qk_scale) : kNegInf;
      if (gid == 0) {
        smem.m[w][m0] = wm0;
        smem.m[w][m1] = wm1;
      }
    }
    __syncthreads();

    float alpha[3][2];
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const float tm0 = fmaxf(fmaxf(smem.m[0][m0], smem.m[1][m0]), fmaxf(smem.m[2][m0], smem.m[3][m0]));
      const float tm1 = fmaxf(fmaxf(smem.m[0][m1], smem.m[1][m1]), fmaxf(smem.m[2][m1], smem.m[3][m1]));
      const float mn0 = fmaxf(m_i[mt][0], tm0);
      const float mn1 = fmaxf(m_i[mt][1], tm1);
      const float ms0 = (mn0 == kNegInf) ? 0.0f : mn0;
      const float ms1 = (mn1 == kNegInf) ? 0.0f : mn1;
      alpha[mt][0] = ex2_approx(m_i[mt][0] - ms0);
      alpha[mt][1] = ex2_approx(m_i[mt][1] - ms1);
      m_i[mt][0] = mn0;
      m_i[mt][1] = mn1;

      const float p00 = page_ok ? ex2_approx(__fmaf_rn(s[mt][0], qk_scale, -ms0)) : 0.0f;
      const float p01 = page_ok ? ex2_approx(__fmaf_rn(s[mt][1], qk_scale, -ms1)) : 0.0f;
      const float p10 = page_ok ? ex2_approx(__fmaf_rn(s[mt][2], qk_scale, -ms0)) : 0.0f;
      const float p11 = page_ok ? ex2_approx(__fmaf_rn(s[mt][3], qk_scale, -ms1)) : 0.0f;

      smem.p[m0][k_local0] = __float2bfloat16_rn(p00);
      smem.p[m1][k_local0] = __float2bfloat16_rn(p01);
      smem.p[m0][k_local1] = __float2bfloat16_rn(p10);
      smem.p[m1][k_local1] = __float2bfloat16_rn(p11);

      const float wl0 = col_reduce_sum8(p00 + p10);
      const float wl1 = col_reduce_sum8(p01 + p11);
      if (gid == 0) {
        smem.l[w][m0] = wl0;
        smem.l[w][m1] = wl1;
      }
    }

    if (it + 1 < bulk_count) {
      cur_page = bt[p_idx + 1];
      cp_async_page_tile<false, false>(smem.k[nxt], KVState, kv_head, cur_page, (p_idx + 1) << 6, seqlen_k, tid, true);
      cp_async_commit();
      if (it > 0) cp_async_wait<1>();
    } else if (tail_count > 0) {
      cur_page = bt[tail_start];
      cp_async_page_tile<false, false>(smem.k[0], KVState, kv_head, cur_page, tail_start << 6, seqlen_k, tid, false);
      cp_async_rel_tile(smem.rel, Rel, q_start, kv_head, past, tail_start << 6, tid, false);
      cp_async_commit();
      if (it > 0) cp_async_wait<1>();
    } else if (it > 0) {
      cp_async_wait<0>();
    }
    __syncthreads();

#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const float sum0 = (smem.l[0][m0] + smem.l[1][m0]) + (smem.l[2][m0] + smem.l[3][m0]);
      const float sum1 = (smem.l[0][m1] + smem.l[1][m1]) + (smem.l[2][m1] + smem.l[3][m1]);
      l_i[mt][0] = __fmaf_rn(l_i[mt][0], alpha[mt][0], sum0);
      l_i[mt][1] = __fmaf_rn(l_i[mt][1], alpha[mt][1], sum1);
#pragma unroll
      for (int dt = 0; dt < 2; ++dt) {
        acc[dt][mt][0] *= alpha[mt][0];
        acc[dt][mt][1] *= alpha[mt][1];
        acc[dt][mt][2] *= alpha[mt][0];
        acc[dt][mt][3] *= alpha[mt][1];
      }
    }

    compute_vp_tile_ldm<4>(acc, smem.v, smem.p, w, lane, gid, tig);

    if (it + 1 < bulk_count || tail_count > 0) {
      __syncthreads();
    }
  }

  // ---------------- Phase B: Causal + Rel-Bias Tail Loop (<= 17 pages at long context) ----------------
  const int rev0 = (kBN - 1) - k_local0;
  const int rev1 = (kBN - 1) - k_local1;
  for (int it = 0; it < tail_count; ++it) {
    const int cur = it & 1;
    const int nxt = cur ^ 1;
    const int p_idx = tail_start + it;
    const int k0 = p_idx << 6;
    const int page = cur_page;

    cp_async_page_tile<true, true>(smem.v, KVState, kv_head, page, k0, seqlen_k, tid, false);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();

    float s[3][4];
    compute_kq_tile_ldm(s, smem.k[cur], q_b0, q_b1, w, lane);

    const bool page_ok = (page >= 0);
    const int k_abs0 = k0 + k_local0;
    const int k_abs1 = k0 + k_local1;
    const bool kv_ok0 = page_ok && (k_abs0 < seqlen_k);
    const bool kv_ok1 = page_ok && (k_abs1 < seqlen_k);

#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const int tok0 = m0 / kGroup;
      const int tok1 = m1 / kGroup;
      const int sh0 = ((past + tok0) - (k0 + kBN - 1)) & 7;
      const int sh1 = ((past + tok1) - (k0 + kBN - 1)) & 7;

      const int dist00 = (past + tok0) - k_abs0;
      const int dist01 = (past + tok1) - k_abs0;
      const int dist10 = (past + tok0) - k_abs1;
      const int dist11 = (past + tok1) - k_abs1;
      const bool v00 = kv_ok0 && (dist00 >= 0);
      const bool v01 = kv_ok0 && (dist01 >= 0);
      const bool v10 = kv_ok1 && (dist10 >= 0);
      const bool v11 = kv_ok1 && (dist11 >= 0);

      s[mt][0] = v00 ? __fmaf_rn(s[mt][0], qk_scale, __bfloat162float(smem.rel[m0][rev0 + sh0]) * kLog2E) : kNegInf;
      s[mt][1] = v01 ? __fmaf_rn(s[mt][1], qk_scale, __bfloat162float(smem.rel[m1][rev0 + sh1]) * kLog2E) : kNegInf;
      s[mt][2] = v10 ? __fmaf_rn(s[mt][2], qk_scale, __bfloat162float(smem.rel[m0][rev1 + sh0]) * kLog2E) : kNegInf;
      s[mt][3] = v11 ? __fmaf_rn(s[mt][3], qk_scale, __bfloat162float(smem.rel[m1][rev1 + sh1]) * kLog2E) : kNegInf;

      const float wm0 = col_reduce_max8(fmaxf(s[mt][0], s[mt][2]));
      const float wm1 = col_reduce_max8(fmaxf(s[mt][1], s[mt][3]));
      if (gid == 0) {
        smem.m[w][m0] = wm0;
        smem.m[w][m1] = wm1;
      }
    }
    __syncthreads();

    float alpha[3][2];
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const float tm0 = fmaxf(fmaxf(smem.m[0][m0], smem.m[1][m0]), fmaxf(smem.m[2][m0], smem.m[3][m0]));
      const float tm1 = fmaxf(fmaxf(smem.m[0][m1], smem.m[1][m1]), fmaxf(smem.m[2][m1], smem.m[3][m1]));
      const float mn0 = fmaxf(m_i[mt][0], tm0);
      const float mn1 = fmaxf(m_i[mt][1], tm1);
      const float ms0 = (mn0 == kNegInf) ? 0.0f : mn0;
      const float ms1 = (mn1 == kNegInf) ? 0.0f : mn1;
      alpha[mt][0] = ex2_approx(m_i[mt][0] - ms0);
      alpha[mt][1] = ex2_approx(m_i[mt][1] - ms1);
      m_i[mt][0] = mn0;
      m_i[mt][1] = mn1;

      const float p00 = ex2_approx(s[mt][0] - ms0);
      const float p01 = ex2_approx(s[mt][1] - ms1);
      const float p10 = ex2_approx(s[mt][2] - ms0);
      const float p11 = ex2_approx(s[mt][3] - ms1);

      smem.p[m0][k_local0] = __float2bfloat16_rn(p00);
      smem.p[m1][k_local0] = __float2bfloat16_rn(p01);
      smem.p[m0][k_local1] = __float2bfloat16_rn(p10);
      smem.p[m1][k_local1] = __float2bfloat16_rn(p11);

      const float wl0 = col_reduce_sum8(p00 + p10);
      const float wl1 = col_reduce_sum8(p01 + p11);
      if (gid == 0) {
        smem.l[w][m0] = wl0;
        smem.l[w][m1] = wl1;
      }
    }

    if (it + 1 < tail_count) {
      cur_page = bt[p_idx + 1];
      cp_async_page_tile<false, false>(smem.k[nxt], KVState, kv_head, cur_page, (p_idx + 1) << 6, seqlen_k, tid, false);
      cp_async_rel_tile(smem.rel, Rel, q_start, kv_head, past, (p_idx + 1) << 6, tid, false);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const float sum0 = (smem.l[0][m0] + smem.l[1][m0]) + (smem.l[2][m0] + smem.l[3][m0]);
      const float sum1 = (smem.l[0][m1] + smem.l[1][m1]) + (smem.l[2][m1] + smem.l[3][m1]);
      l_i[mt][0] = __fmaf_rn(l_i[mt][0], alpha[mt][0], sum0);
      l_i[mt][1] = __fmaf_rn(l_i[mt][1], alpha[mt][1], sum1);
#pragma unroll
      for (int dt = 0; dt < 2; ++dt) {
        acc[dt][mt][0] *= alpha[mt][0];
        acc[dt][mt][1] *= alpha[mt][1];
        acc[dt][mt][2] *= alpha[mt][0];
        acc[dt][mt][3] *= alpha[mt][1];
      }
    }

    compute_vp_tile_ldm<4>(acc, smem.v, smem.p, w, lane, gid, tig);

    if (it + 1 < tail_count) {
      __syncthreads();
    }
  }

  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  __syncthreads();
  float (*smem_o)[kHeadDim] = reinterpret_cast<float (*)[kHeadDim]>(&smem.k[0][0]);

#pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    const int m0 = mt * 8 + tig * 2;
    const int m1 = m0 + 1;
    const float inv0 = (l_i[mt][0] == 0.0f) ? 0.0f : (v_scale / l_i[mt][0]);
    const float inv1 = (l_i[mt][1] == 0.0f) ? 0.0f : (v_scale / l_i[mt][1]);
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
      const int d0 = w * 32 + dt * 16 + gid;
      const int d1 = d0 + 8;
      smem_o[m0][d0] = acc[dt][mt][0] * inv0;
      smem_o[m1][d0] = acc[dt][mt][1] * inv1;
      smem_o[m0][d1] = acc[dt][mt][2] * inv0;
      smem_o[m1][d1] = acc[dt][mt][3] * inv1;
    }
    if (w == 0 && gid == 0) {
      const int tok0 = m0 / kGroup;
      const int tok1 = m1 / kGroup;
      const int h0 = kv_head * kGroup + (m0 - tok0 * kGroup);
      const int h1 = kv_head * kGroup + (m1 - tok1 * kGroup);
      const float lse0 = (l_i[mt][0] == 0.0f) ? kNegInf : (m_i[mt][0] + lg2_approx(l_i[mt][0]));
      const float lse1 = (l_i[mt][1] == 0.0f) ? kNegInf : (m_i[mt][1] + lg2_approx(l_i[mt][1]));
      LsePart[(static_cast<int64_t>(split) * num_tokens + (q_start + tok0)) * kHeads + h0] = lse0;
      LsePart[(static_cast<int64_t>(split) * num_tokens + (q_start + tok1)) * kHeads + h1] = lse1;
    }
  }
  __syncthreads();

#pragma unroll
  for (int step = 0; step < 6; ++step) {
    const int idx = tid + step * kThreads;
    const int m = idx >> 5;
    const int d4 = (idx & 31) << 2;
    const int tok = m / kGroup;
    const int head = kv_head * kGroup + (m - tok * kGroup);
    const int64_t part = (static_cast<int64_t>(split) * num_tokens + (q_start + tok)) * kHeads + head;
    *reinterpret_cast<float4*>(OPart + part * kHeadDim + d4) =
        *reinterpret_cast<const float4*>(&smem_o[m][d4]);
  }
}

// ---------------------------------------------------------------------------
// 3. Fast 1-Warp 4-Way Pipelined Float4 Dynamic Split-KV Combine + Gate + Sink (SM120)
// Grid: [num_tokens, 24 (heads), 1], Block: [32, 1, 1]
// Issues up to 4 float4 OPart loads concurrently (4x MLP) while accumulating in
// exact serial split order (100% bit-identical to static combine!).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void accum_active_splits_4x(
    float& o0, float& o1, float& o2, float& o3,
    uint32_t active_mask,
    float w_norm_lane,
    int s_base,
    const float* __restrict__ OPart,
    int64_t split_stride,
    int64_t base_row_head,
    int d4) {
  while (active_mask != 0u) {
    const int b0 = __ffs(active_mask) - 1;
    active_mask &= (active_mask - 1u);
    const float w0 = __shfl_sync(0xffffffffu, w_norm_lane, b0);
    const float4 v0 = *reinterpret_cast<const float4*>(
        OPart + (static_cast<int64_t>(s_base + b0) * split_stride + base_row_head) * kHeadDim + d4);
    if (active_mask != 0u) {
      const int b1 = __ffs(active_mask) - 1;
      active_mask &= (active_mask - 1u);
      const float w1 = __shfl_sync(0xffffffffu, w_norm_lane, b1);
      const float4 v1 = *reinterpret_cast<const float4*>(
          OPart + (static_cast<int64_t>(s_base + b1) * split_stride + base_row_head) * kHeadDim + d4);
      if (active_mask != 0u) {
        const int b2 = __ffs(active_mask) - 1;
        active_mask &= (active_mask - 1u);
        const float w2 = __shfl_sync(0xffffffffu, w_norm_lane, b2);
        const float4 v2 = *reinterpret_cast<const float4*>(
            OPart + (static_cast<int64_t>(s_base + b2) * split_stride + base_row_head) * kHeadDim + d4);
        if (active_mask != 0u) {
          const int b3 = __ffs(active_mask) - 1;
          active_mask &= (active_mask - 1u);
          const float w3 = __shfl_sync(0xffffffffu, w_norm_lane, b3);
          const float4 v3 = *reinterpret_cast<const float4*>(
              OPart + (static_cast<int64_t>(s_base + b3) * split_stride + base_row_head) * kHeadDim + d4);
          o0 = __fmaf_rn(w0, v0.x, o0); o1 = __fmaf_rn(w0, v0.y, o1); o2 = __fmaf_rn(w0, v0.z, o2); o3 = __fmaf_rn(w0, v0.w, o3);
          o0 = __fmaf_rn(w1, v1.x, o0); o1 = __fmaf_rn(w1, v1.y, o1); o2 = __fmaf_rn(w1, v1.z, o2); o3 = __fmaf_rn(w1, v1.w, o3);
          o0 = __fmaf_rn(w2, v2.x, o0); o1 = __fmaf_rn(w2, v2.y, o1); o2 = __fmaf_rn(w2, v2.z, o2); o3 = __fmaf_rn(w2, v2.w, o3);
          o0 = __fmaf_rn(w3, v3.x, o0); o1 = __fmaf_rn(w3, v3.y, o1); o2 = __fmaf_rn(w3, v3.z, o2); o3 = __fmaf_rn(w3, v3.w, o3);
          continue;
        }
        o0 = __fmaf_rn(w0, v0.x, o0); o1 = __fmaf_rn(w0, v0.y, o1); o2 = __fmaf_rn(w0, v0.z, o2); o3 = __fmaf_rn(w0, v0.w, o3);
        o0 = __fmaf_rn(w1, v1.x, o0); o1 = __fmaf_rn(w1, v1.y, o1); o2 = __fmaf_rn(w1, v1.z, o2); o3 = __fmaf_rn(w1, v1.w, o3);
        o0 = __fmaf_rn(w2, v2.x, o0); o1 = __fmaf_rn(w2, v2.y, o1); o2 = __fmaf_rn(w2, v2.z, o2); o3 = __fmaf_rn(w2, v2.w, o3);
        continue;
      }
      o0 = __fmaf_rn(w0, v0.x, o0); o1 = __fmaf_rn(w0, v0.y, o1); o2 = __fmaf_rn(w0, v0.z, o2); o3 = __fmaf_rn(w0, v0.w, o3);
      o0 = __fmaf_rn(w1, v1.x, o0); o1 = __fmaf_rn(w1, v1.y, o1); o2 = __fmaf_rn(w1, v1.z, o2); o3 = __fmaf_rn(w1, v1.w, o3);
      continue;
    }
    o0 = __fmaf_rn(w0, v0.x, o0); o1 = __fmaf_rn(w0, v0.y, o1); o2 = __fmaf_rn(w0, v0.z, o2); o3 = __fmaf_rn(w0, v0.w, o3);
  }
}

extern "C" __global__ void __launch_bounds__(32) kern_attn_verify_combine_dyn(
    const float* __restrict__ OPart,
    const float* __restrict__ LsePart,
    const bf16* __restrict__ Gate,
    const bf16* __restrict__ Sink,
    const int32_t* __restrict__ SeqLens,
    bf16* __restrict__ Y,
    int num_tokens,
    int stride_gate,
    int stride_sink,
    int total_splits) {
  const int64_t row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  const int d4 = lane << 2;

  __shared__ int sh_sp[kMaxDynSeqs];
  __shared__ int sh_np[kMaxDynSeqs];

  const int seqs = num_tokens >> 3;
  const int req = static_cast<int>(row >> 3);
  int ns = total_splits;
  if (seqs <= 1) {
    ns = (SeqLens[0] > 0) ? total_splits : 0;
  } else {
    if (lane == 0) {
      compute_dyn_splits_smem(SeqLens, seqs, total_splits, sh_sp, sh_np);
    }
    __syncwarp();
    ns = sh_sp[req];
  }

  // Preload Gate and Sink BEFORE griddepcontrol.wait!
  const uint2 gate_raw = *reinterpret_cast<const uint2*>(
      Gate + row * stride_gate + head * kHeadDim + d4);
  const bf16* g_bf = reinterpret_cast<const bf16*>(&gate_raw);
  const float g0 = __bfloat162float(g_bf[0]);
  const float g1 = __bfloat162float(g_bf[1]);
  const float g2 = __bfloat162float(g_bf[2]);
  const float g3 = __bfloat162float(g_bf[3]);
  const float sink_val = __bfloat162float(Sink[row * stride_sink + head]);

  asm volatile("griddepcontrol.wait;" ::: "memory");

  bf16* dst = Y + row * (kHeads * kHeadDim) + head * kHeadDim + d4;
  if (ns <= 0) {
    *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
    return;
  }

  const int64_t base_row_head = row * kHeads + head;
  const int64_t split_stride = static_cast<int64_t>(num_tokens) * kHeads;

  const float lse_s = (lane < ns)
      ? LsePart[static_cast<int64_t>(lane) * split_stride + base_row_head]
      : kNegInf;
  const float lse_s1 = ((lane + 32) < ns)
      ? LsePart[static_cast<int64_t>(lane + 32) * split_stride + base_row_head]
      : kNegInf;
  float lse_max = fmaxf(lse_s, lse_s1);
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    lse_max = fmaxf(lse_max, __shfl_xor_sync(0xffffffffu, lse_max, off));
  }

  if (lse_max == kNegInf) {
    *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
    return;
  }

  const float w_lane = (lane < ns) ? ex2_approx(lse_s - lse_max) : 0.0f;
  const float w_lane1 = ((lane + 32) < ns) ? ex2_approx(lse_s1 - lse_max) : 0.0f;
  float w_sum = w_lane + w_lane1;
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    w_sum += __shfl_xor_sync(0xffffffffu, w_sum, off);
  }

  if (w_sum == 0.0f) {
    *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
    return;
  }

  const float inv_w_sum = 1.0f / w_sum;
  const float wn0 = w_lane * inv_w_sum;
  const float wn1 = w_lane1 * inv_w_sum;

  float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
  accum_active_splits_4x(
      o0, o1, o2, o3, __ballot_sync(0xffffffffu, w_lane > 0.0f),
      wn0, 0, OPart, split_stride, base_row_head, d4);
  if (ns > 32) {
    accum_active_splits_4x(
        o0, o1, o2, o3, __ballot_sync(0xffffffffu, w_lane1 > 0.0f),
        wn1, 32, OPart, split_stride, base_row_head, d4);
  }

  const float lse_nat = (lg2_approx(w_sum) + lse_max) * kLn2;
  const float sink_scale = sigmoid_fast(lse_nat - sink_val);

  uint2 out_raw;
  bf16* out_bf = reinterpret_cast<bf16*>(&out_raw);
  out_bf[0] = __float2bfloat16_rn(o0 * sigmoid_fast(g0) * sink_scale);
  out_bf[1] = __float2bfloat16_rn(o1 * sigmoid_fast(g1) * sink_scale);
  out_bf[2] = __float2bfloat16_rn(o2 * sigmoid_fast(g2) * sink_scale);
  out_bf[3] = __float2bfloat16_rn(o3 * sigmoid_fast(g3) * sink_scale);
  *reinterpret_cast<uint2*>(dst) = out_raw;
}

// ---------------------------------------------------------------------------
// 4. Fast 1-Warp Float4 Static Split-KV Combine + Gate + Sink Epilogue (SM120)
// Grid: [num_tokens, 24 (heads), 1], Block: [32, 1, 1]
// ---------------------------------------------------------------------------
template <int NUM_SPLITS>
__device__ __forceinline__ void attn_verify_combine_impl(
    const float* __restrict__ OPart,
    const float* __restrict__ LsePart,
    const bf16* __restrict__ Gate,
    const bf16* __restrict__ Sink,
    bf16* __restrict__ Y,
    int num_tokens,
    int stride_gate,
    int stride_sink) {
  const int64_t row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  const int d4 = lane << 2;

  const uint2 gate_raw = *reinterpret_cast<const uint2*>(
      Gate + row * stride_gate + head * kHeadDim + d4);
  const bf16* g_bf = reinterpret_cast<const bf16*>(&gate_raw);
  const float g0 = __bfloat162float(g_bf[0]);
  const float g1 = __bfloat162float(g_bf[1]);
  const float g2 = __bfloat162float(g_bf[2]);
  const float g3 = __bfloat162float(g_bf[3]);
  const float sink_val = __bfloat162float(Sink[row * stride_sink + head]);

  asm volatile("griddepcontrol.wait;" ::: "memory");

  const int64_t base_row_head = row * kHeads + head;
  const int64_t split_stride = static_cast<int64_t>(num_tokens) * kHeads;

  float lse_s = (lane < NUM_SPLITS) ? LsePart[static_cast<int64_t>(lane) * split_stride + base_row_head] : kNegInf;
  float lse_s1 = kNegInf;
  float lse_max = lse_s;
  if constexpr (NUM_SPLITS > 32) {
    lse_s1 = ((lane + 32) < NUM_SPLITS)
        ? LsePart[static_cast<int64_t>(lane + 32) * split_stride + base_row_head]
        : kNegInf;
    lse_max = fmaxf(lse_s, lse_s1);
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    lse_max = fmaxf(lse_max, __shfl_xor_sync(0xffffffffu, lse_max, off));
  }

  bf16* dst = Y + row * (kHeads * kHeadDim) + head * kHeadDim + d4;
  if (lse_max == kNegInf) {
    *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
    return;
  }

  float w_lane = (lane < NUM_SPLITS) ? ex2_approx(lse_s - lse_max) : 0.0f;
  float w_lane1 = 0.0f;
  float w_sum = w_lane;
  if constexpr (NUM_SPLITS > 32) {
    w_lane1 = ((lane + 32) < NUM_SPLITS) ? ex2_approx(lse_s1 - lse_max) : 0.0f;
    w_sum = w_lane + w_lane1;
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    w_sum += __shfl_xor_sync(0xffffffffu, w_sum, off);
  }

  if (w_sum == 0.0f) {
    *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
    return;
  }

  const float inv_w_sum = 1.0f / w_sum;
  const float w_norm_lane = w_lane * inv_w_sum;

  float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
  accum_active_splits_4x(
      o0, o1, o2, o3, __ballot_sync(0xffffffffu, w_lane > 0.0f),
      w_norm_lane, 0, OPart, split_stride, base_row_head, d4);
  if constexpr (NUM_SPLITS > 32) {
    const float w_norm_lane1 = w_lane1 * inv_w_sum;
    accum_active_splits_4x(
        o0, o1, o2, o3, __ballot_sync(0xffffffffu, w_lane1 > 0.0f),
        w_norm_lane1, 32, OPart, split_stride, base_row_head, d4);
  }

  const float lse_nat = (lg2_approx(w_sum) + lse_max) * kLn2;
  const float sink_scale = sigmoid_fast(lse_nat - sink_val);

  uint2 out_raw;
  bf16* out_bf = reinterpret_cast<bf16*>(&out_raw);
  out_bf[0] = __float2bfloat16_rn(o0 * sigmoid_fast(g0) * sink_scale);
  out_bf[1] = __float2bfloat16_rn(o1 * sigmoid_fast(g1) * sink_scale);
  out_bf[2] = __float2bfloat16_rn(o2 * sigmoid_fast(g2) * sink_scale);
  out_bf[3] = __float2bfloat16_rn(o3 * sigmoid_fast(g3) * sink_scale);
  *reinterpret_cast<uint2*>(dst) = out_raw;
}

#define DEFINE_COMBINE(S) \
  extern "C" __global__ void __launch_bounds__(32) kern_attn_verify_combine_s##S( \
      const float* __restrict__ OPart, const float* __restrict__ LsePart, \
      const bf16* __restrict__ Gate, const bf16* __restrict__ Sink, bf16* __restrict__ Y, \
      int num_tokens, int stride_gate, int stride_sink) { \
    attn_verify_combine_impl<S>(OPart, LsePart, Gate, Sink, Y, num_tokens, stride_gate, stride_sink); \
  }

DEFINE_COMBINE(4)
DEFINE_COMBINE(5)
DEFINE_COMBINE(8)
DEFINE_COMBINE(9)
DEFINE_COMBINE(10)
DEFINE_COMBINE(16)
DEFINE_COMBINE(17)
DEFINE_COMBINE(20)
DEFINE_COMBINE(32)
DEFINE_COMBINE(41)
#undef DEFINE_COMBINE
