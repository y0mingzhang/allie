"""Allie 2.0's fast CPU backend: model.py's step() as one call into C++ kernels.

The kernels (SOURCE) are compiled for this machine on first use with the system C++ compiler
(a few seconds, cached under ~/.cache/allie/kernels) and loaded with ctypes: no torch headers,
no build at install, nothing beyond the standard library. They run the whole step (input
embedding, board CNN, every block, the head) on a pool of pinned threads that stays alive
between steps, with spin barriers between the phases of a block. Matrices are read in place
from model.w (int8 with per-row scales, or BF16) and multiplied as they stream from memory;
the experts of a step are grouped so each expert's weights are read once for all its tokens.
Activations are rounded to BF16 wherever model.py's BF16 tensors round them, with FP32 sums,
so the result matches the PyTorch reference up to summation order. Without a compiler,
model.py falls back to PyTorch.
"""

import ctypes
import functools
import hashlib
import os
import platform
import subprocess
import weakref
from pathlib import Path

import torch
from torch.nn import functional as F

SOURCE = r"""
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

typedef uint16_t bf16;
static inline float f32(bf16 v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
static inline float f32(int8_t v) { return v; }
static inline float f32(float v) { return v; }
static inline bf16 tobf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)(u >> 16 | 0x40);
  return (bf16)((u + 0x7fffu + (u >> 16 & 1)) >> 16);
}
static inline float rb(float f) { return f32(tobf(f)); }  // round to BF16
static inline float sigm(float x) { return 1.f / (1.f + expf(-x)); }

#if defined(__AVX512F__)
#define VL 16
#define MB 6
#define RB 4
typedef __m512 vf;
static inline vf vzero() { return _mm512_setzero_ps(); }
static inline vf vset(float a) { return _mm512_set1_ps(a); }
static inline vf vld(const float* p) { return _mm512_loadu_ps(p); }
static inline vf vld(const int8_t* p) {
  return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)p)));
}
static inline vf vld(const bf16* p) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p)), 16));
}
static inline void vst(float* p, vf a) { _mm512_storeu_ps(p, a); }
static inline vf vadd(vf a, vf b) { return _mm512_add_ps(a, b); }
static inline vf vsub(vf a, vf b) { return _mm512_sub_ps(a, b); }
static inline vf vmul(vf a, vf b) { return _mm512_mul_ps(a, b); }
static inline vf vdiv(vf a, vf b) { return _mm512_div_ps(a, b); }
static inline vf vmax(vf a, vf b) { return _mm512_max_ps(a, b); }
static inline vf vmin(vf a, vf b) { return _mm512_min_ps(a, b); }
static inline vf vfma(vf a, vf b, vf c) { return _mm512_fmadd_ps(a, b, c); }
static inline float vsum(vf a) { return _mm512_reduce_add_ps(a); }
static inline vf vfloor(vf a) { return _mm512_roundscale_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
static inline vf vpow2(vf n) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23));
}
static inline vf vrb(vf x) {
  __m512i u = _mm512_castps_si512(x);
  u = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1))));
  return _mm512_castsi512_ps(_mm512_and_si512(u, _mm512_set1_epi32((int)0xffff0000u)));
}
#define SIMD 1
#elif defined(__AVX2__) && defined(__FMA__)
#define VL 8
#define MB 6
#define RB 2
typedef __m256 vf;
static inline vf vzero() { return _mm256_setzero_ps(); }
static inline vf vset(float a) { return _mm256_set1_ps(a); }
static inline vf vld(const float* p) { return _mm256_loadu_ps(p); }
static inline vf vld(const int8_t* p) {
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)p)));
}
static inline vf vld(const bf16* p) {
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)), 16));
}
static inline void vst(float* p, vf a) { _mm256_storeu_ps(p, a); }
static inline vf vadd(vf a, vf b) { return _mm256_add_ps(a, b); }
static inline vf vsub(vf a, vf b) { return _mm256_sub_ps(a, b); }
static inline vf vmul(vf a, vf b) { return _mm256_mul_ps(a, b); }
static inline vf vdiv(vf a, vf b) { return _mm256_div_ps(a, b); }
static inline vf vmax(vf a, vf b) { return _mm256_max_ps(a, b); }
static inline vf vmin(vf a, vf b) { return _mm256_min_ps(a, b); }
static inline vf vfma(vf a, vf b, vf c) { return _mm256_fmadd_ps(a, b, c); }
static inline float vsum(vf a) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
}
static inline vf vfloor(vf a) { return _mm256_floor_ps(a); }
static inline vf vpow2(vf n) {
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23));
}
static inline vf vrb(vf x) {
  __m256i u = _mm256_castps_si256(x);
  u = _mm256_add_epi32(u, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1))));
  return _mm256_castsi256_ps(_mm256_and_si256(u, _mm256_set1_epi32((int)0xffff0000u)));
}
#define SIMD 1
#else
#define VL 8
#define MB 6
#define RB 2
struct vf { float v[VL]; };
#define VOP(name, e) static inline vf name(vf a, vf b) { vf r; for (int i = 0; i < VL; i++) r.v[i] = e; return r; }
VOP(vadd, a.v[i] + b.v[i]) VOP(vsub, a.v[i] - b.v[i]) VOP(vmul, a.v[i] * b.v[i]) VOP(vdiv, a.v[i] / b.v[i])
VOP(vmax, std::max(a.v[i], b.v[i])) VOP(vmin, std::min(a.v[i], b.v[i]))
static inline vf vset(float x) { vf r; for (int i = 0; i < VL; i++) r.v[i] = x; return r; }
static inline vf vzero() { return vset(0.f); }
template <class T> static inline vf vld(const T* p) { vf r; for (int i = 0; i < VL; i++) r.v[i] = f32(p[i]); return r; }
static inline void vst(float* p, vf a) { memcpy(p, a.v, sizeof a.v); }
static inline vf vfma(vf a, vf b, vf c) { vf r; for (int i = 0; i < VL; i++) r.v[i] = a.v[i] * b.v[i] + c.v[i]; return r; }
static inline float vsum(vf a) { float s = 0; for (int i = 0; i < VL; i++) s += a.v[i]; return s; }
static inline vf vexp(vf a) { vf r; for (int i = 0; i < VL; i++) r.v[i] = expf(a.v[i]); return r; }
static inline vf vrb(vf a) { vf r; for (int i = 0; i < VL; i++) r.v[i] = rb(a.v[i]); return r; }
#endif

#ifdef SIMD
static inline vf vexp(vf x) {  // Cephes expf: within 2 ulp
  x = vmin(vmax(x, vset(-87.3f)), vset(88.f));
  vf n = vfloor(vfma(x, vset(1.44269504088896341f), vset(0.5f)));
  x = vsub(x, vmul(n, vset(0.693359375f)));
  x = vsub(x, vmul(n, vset(-2.12194440e-4f)));
  vf y = vset(1.9875691500e-4f);
  y = vfma(y, x, vset(1.3981999507e-3f));
  y = vfma(y, x, vset(8.3334519073e-3f));
  y = vfma(y, x, vset(4.1665795894e-2f));
  y = vfma(y, x, vset(1.6666665459e-1f));
  y = vfma(y, x, vset(5.0000001201e-1f));
  y = vfma(y, vmul(x, x), vadd(x, vset(1.f)));
  return vmul(y, vpow2(n));
}
#endif

static inline void cpu_relax() {
#if defined(__x86_64__)
  _mm_pause();
#elif defined(__aarch64__)
  asm volatile("yield");
#endif
}

// ---- thread pool: workers stay alive between steps, spin briefly for work, then sleep ----

struct Pool {
  int n;
  bool pin;
  double spin;
  std::vector<int> cpus;
  std::vector<std::thread> th;
  alignas(64) std::atomic<uint32_t> epoch{0};
  alignas(64) std::atomic<int> left{0};
  alignas(64) std::atomic<int> count{0};
  alignas(64) std::atomic<uint32_t> gen{0};
  std::atomic<uint32_t> pokes{0};
  std::mutex mu;
  std::condition_variable cv;
  bool stop = false;
  void (*fn)(void*, int) = nullptr;
  void* arg = nullptr;

  Pool(int n_, const int32_t* cpu, double spin_) : n(n_), pin(cpu != nullptr), spin(spin_) {  // cpu: thread t's CPU
    if (pin) cpus.assign(cpu, cpu + n);
    for (int t = 1; t < n; t++) th.emplace_back([this, t] { work(t); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
      epoch.fetch_add(1, std::memory_order_release);
    }
    cv.notify_all();
    for (auto& t : th) t.join();
  }
  void bind(int t) {
#if defined(__linux__)
    if (!pin) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpus[t], &set);
    sched_setaffinity(0, sizeof set, &set);
#endif
  }
  void work(int t) {
    bind(t);
    uint32_t seen = 0;
    for (;;) {
      auto t0 = std::chrono::steady_clock::now();
      for (int k = 1; epoch.load(std::memory_order_acquire) == seen; k++) {
        cpu_relax();
        if (k % 1024 == 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > spin) {
          uint32_t p = pokes.load();
          std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&] { return epoch.load(std::memory_order_acquire) != seen || pokes.load() != p; });
          t0 = std::chrono::steady_clock::now();  // poked: spin again, work is coming
        }
      }
      seen = epoch.load(std::memory_order_acquire);
      if (stop) return;
      fn(arg, t);
      left.fetch_sub(1, std::memory_order_release);
    }
  }
  void run(void (*f)(void*, int), void* a) {
#if defined(__linux__)
    cpu_set_t old;
    bool restore = pin && sched_getaffinity(0, sizeof old, &old) == 0;
    if (restore) bind(0);
#endif
    fn = f, arg = a;
    left.store(n - 1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(mu);
      epoch.fetch_add(1, std::memory_order_release);
    }
    cv.notify_all();
    f(a, 0);
    while (left.load(std::memory_order_acquire) > 0) cpu_relax();
#if defined(__linux__)
    if (restore) sched_setaffinity(0, sizeof old, &old);
#endif
  }
  void wake() {  // sleeping workers spin again, ready for a step
    {
      std::lock_guard<std::mutex> lk(mu);
      pokes.fetch_add(1);
    }
    cv.notify_all();
  }
  void barrier() {
    if (n == 1) return;
    uint32_t g = gen.load(std::memory_order_acquire);
    if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
      count.store(0, std::memory_order_relaxed);
      gen.store(g + 1, std::memory_order_release);
    } else {
      while (gen.load(std::memory_order_acquire) == g) cpu_relax();
    }
  }
};

static inline void split(int n, int t, int nt, int align, int& lo, int& hi) {
  int chunks = (n + align - 1) / align;
  lo = std::min(n, (int)((long)chunks * t / nt) * align);
  hi = std::min(n, (int)((long)chunks * (t + 1) / nt) * align);
}

// ---- matrix kernels: rows of W (int8, BF16 or FP32) against FP32 activations ----

#if defined(SIMD) && defined(__GNUC__)
#define KEEP(v) asm("" : "+v"(v))
#else
#define KEEP(v)
#endif
#if defined(__clang__)
#define UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define UNROLL _Pragma("GCC unroll 16")
#else
#define UNROLL
#endif

// acc[m * R + i] = x[m] . r[i] over K: R rows of W against M tokens, R * M sums in registers
template <class W, int R, int M>
static inline void block(const float* const* x, const W* const* r, int K, float* acc) {
  vf a[M][R];
  UNROLL for (int m = 0; m < M; m++)
    UNROLL for (int i = 0; i < R; i++) a[m][i] = vzero();
  int k = 0, kv = K - K % VL;
  if (M == 1) {  // two partial sums per row: 2R independent chains
    vf b[R];
    UNROLL for (int i = 0; i < R; i++) b[i] = vzero();
    for (; k + 2 * VL <= kv; k += 2 * VL) {
      vf x0 = vld(x[0] + k), x1 = vld(x[0] + k + VL);
      KEEP(x0);
      KEEP(x1);
      UNROLL for (int i = 0; i < R; i++) {
        a[0][i] = vfma(vld(r[i] + k), x0, a[0][i]);
        b[i] = vfma(vld(r[i] + k + VL), x1, b[i]);
      }
    }
    UNROLL for (int i = 0; i < R; i++) a[0][i] = vadd(a[0][i], b[i]);
  }
  for (; k < kv; k += VL) {
    vf w[R];
    UNROLL for (int i = 0; i < R; i++) w[i] = vld(r[i] + k);  // each weight converted once for M tokens
    UNROLL for (int m = 0; m < M; m++) {
      vf xm = vld(x[m] + k);
      KEEP(xm);  // one load for the R rows, not one folded into each multiply-add
      UNROLL for (int i = 0; i < R; i++) a[m][i] = vfma(w[i], xm, a[m][i]);
    }
  }
  UNROLL for (int m = 0; m < M; m++)
    UNROLL for (int i = 0; i < R; i++) {
      float s = vsum(a[m][i]);
      for (int j = kv; j < K; j++) s += x[m][j] * f32(r[i][j]);
      acc[m * R + i] = s;
    }
}

// rows [j, j + R) of w (the last row repeated past n) against tokens [m0, m1) in groups of MB
template <class W, int R>
static inline void rows(const float* const* x, int m0, int m1, const W* w, int K, int j, int n, float* out, int ldo) {
  const W* r[R];
  for (int i = 0; i < R; i++) r[i] = w + (size_t)std::min(j + i, n - 1) * K;
  int nr = std::min(R, n - j);
  float acc[MB * R];
  for (int m = m0; m < m1; m += MB) {
    int mb = std::min(MB, m1 - m);
    switch (R == 4 && m1 - m0 == 1 ? 0 : mb) {
      case 0: block<W, R, 1>(x + m, r, K, acc); break;
      case 1: block<W, R, 1>(x + m, r, K, acc); break;
      case 2: block<W, R, 2>(x + m, r, K, acc); break;
      case 3: block<W, R, 3>(x + m, r, K, acc); break;
      case 4: block<W, R, 4>(x + m, r, K, acc); break;
      case 5: block<W, R, 5>(x + m, r, K, acc); break;
      default: block<W, R, 6>(x + m, r, K, acc); break;
    }
    for (int a = 0; a < mb; a++)
      for (int i = 0; i < nr; i++) out[(size_t)(m + a) * ldo + j + i] = acc[a * R + i];
  }
}

// out[m * ldo + j] = x[m] . w[j] for rows j < n of w (row stride K), tokens m < M
template <class W>
static void dots(const float* const* x, int M, const W* w, int K, int n, float* out, int ldo) {
  if (M == 1) {  // a matrix-vector product: four rows at a time, eight chains
    for (int j = 0; j < n; j += 4) rows<W, 4>(x, 0, 1, w, K, j, n, out, ldo);
    return;
  }
  int tile = M <= 64 ? M : 48;
  for (int m0 = 0; m0 < M; m0 += tile)
    for (int j = 0; j < n; j += RB) rows<W, RB>(x, m0, std::min(M, m0 + tile), w, K, j, n, out, ldo);
}

struct Mat {
  const void* w;
  const bf16* s;  // int8 rows' scales
  int t;          // 0 BF16, 1 int8, 2 FP32
};

// y[m * ldy + j] = (row r0 + j of A) . x[m], scaled, rounded to BF16 unless FP32 (FP32 rows)
static void mm(const Mat& A, int K, int r0, int n, const float* const* x, int M, float* y, int ldy) {
  if (n <= 0 || M <= 0) return;
  if (A.t == 1) dots(x, M, (const int8_t*)A.w + (size_t)r0 * K, K, n, y, ldy);
  else if (A.t == 0) dots(x, M, (const bf16*)A.w + (size_t)r0 * K, K, n, y, ldy);
  else return dots(x, M, (const float*)A.w + (size_t)r0 * K, K, n, y, ldy);
  for (int m = 0; m < M; m++) {
    float* ym = y + (size_t)m * ldy;
    int j = 0;
    if (A.s) {
      for (; j + VL <= n; j += VL) vst(ym + j, vrb(vmul(vld(ym + j), vld(A.s + r0 + j))));
      for (; j < n; j++) ym[j] = rb(ym[j] * f32(A.s[r0 + j]));
    } else {
      for (; j + VL <= n; j += VL) vst(ym + j, vrb(vld(ym + j)));
      for (; j < n; j++) ym[j] = rb(ym[j]);
    }
  }
}

// y[m * N + d] = x[m] @ W[:, d] for d in [d0, d1), W: K x N BF16 (the x @ W layout)
static void kn(const float* x, int K, int M, const bf16* W, int N, int d0, int d1, float* y) {
  for (int m = 0; m < M; m++) {
    const float* xm = x + (size_t)m * K;
    float* ym = y + (size_t)m * N;
    int d = d0;
    for (; d + 2 * VL <= d1; d += 2 * VL) {
      vf a = vzero(), b = vzero();
      for (int k = 0; k < K; k++) {
        if (xm[k] == 0.f) continue;
        vf s = vset(xm[k]);
        const bf16* w = W + (size_t)k * N + d;
        a = vfma(vld(w), s, a);
        b = vfma(vld(w + VL), s, b);
      }
      vst(ym + d, a);
      vst(ym + d + VL, b);
    }
    for (; d < d1; d++) {
      float a = 0;
      for (int k = 0; k < K; k++) a += xm[k] * f32(W[(size_t)k * N + d]);
      ym[d] = a;
    }
  }
}

static inline float dot(const float* a, const bf16* b, int n) {
  float s = 0;
  for (int i = 0; i < n; i++) s += a[i] * f32(b[i]);
  return s;
}

static void norm(const float* x, float* y, int n) {  // rms_norm, eps = FP32 epsilon, to BF16
  vf a = vzero();
  int i = 0;
  for (; i + VL <= n; i += VL) a = vfma(vld(x + i), vld(x + i), a);
  float s = vsum(a);
  for (; i < n; i++) s += x[i] * x[i];
  float r = 1.f / sqrtf(s / n + 1.1920928955078125e-07f);
  vf vr = vset(r);
  for (i = 0; i + VL <= n; i += VL) vst(y + i, vrb(vmul(vld(x + i), vr)));
  for (; i < n; i++) y[i] = rb(x[i] * r);
}

// y = rb(a + rb(c * b))
static void axpy_rb(float* y, const float* a, float c, const float* b, int n) {
  int i = 0;
  for (; i + VL <= n; i += VL) vst(y + i, vrb(vadd(vld(a + i), vrb(vmul(vset(c), vld(b + i))))));
  for (; i < n; i++) y[i] = rb(a[i] + rb(c * b[i]));
}

// y = rb(y + b)
static void add_rb(float* y, const float* b, int n) {
  int i = 0;
  for (; i + VL <= n; i += VL) vst(y + i, vrb(vadd(vld(y + i), vld(b + i))));
  for (; i < n; i++) y[i] = rb(y[i] + b[i]);
}

static void gelu(float* a, int n) {  // tanh approximation, to BF16
  int i = 0;
  const float c = 0.7978845608028654f, k = 0.044715f;
  for (; i + VL <= n; i += VL) {
    vf x = vld(a + i);
    vf u = vmul(vset(c), vfma(vmul(vset(k), vmul(x, x)), x, x));
    vf t = vsub(vset(1.f), vdiv(vset(2.f), vadd(vexp(vmul(vset(2.f), vmin(vmax(u, vset(-9.f)), vset(9.f)))), vset(1.f))));
    vst(a + i, vmul(vmul(vset(0.5f), x), vadd(vset(1.f), t)));
  }
  for (; i < n; i++) a[i] = 0.5f * a[i] * (1.f + tanhf(c * (a[i] + k * a[i] * a[i] * a[i])));
  for (i = 0; i < n; i++) a[i] = rb(a[i]);
}

static void silu_mul(const float* a, const float* b, float* y, int n) {  // rb(rb(silu(a)) * b)
  int i = 0;
  float t[VL];
  for (; i + VL <= n; i += VL) {
    vf x = vld(a + i);
    vst(t, vdiv(x, vadd(vset(1.f), vexp(vsub(vzero(), x)))));
    for (int j = 0; j < VL; j++) y[i + j] = rb(rb(t[j]) * b[i + j]);
  }
  for (; i < n; i++) y[i] = rb(rb(a[i] / (1.f + expf(-a[i]))) * b[i]);
}

// ---- the model ----

enum { E_CNN, E_ROWS, E_SMEAR, B_NORM, B_QKV, B_ROTARY, B_ATTN, B_O, B_NORM2, B_ROUTER, B_TOPK, B_GROUP, B_UP,
       B_DOWN, H_NORM, H_HEAD, NPHASE };

struct Layer {
  Mat qkv, o, fc, proj, sup, sdown;
  const void *up, *down;
  const bf16 *ups, *downs, *gates;
  const float *router, *bias, *mu;
  int G, ve, skin, skout;
};

struct Seq {
  int64_t n0, len, cap, off;
  bf16 *k, *v, *e;
};

struct Seg {  // one up / fc matrix (rows: gate halves, then value halves) for some tokens
  Mat A;
  int pairs, ntok, ldo;
  const float* const* x;
  float* out;
};

struct Chunk {
  int seg, p0, n;
};

struct Engine {
  int L, D, H, hd, V, nve, E, topk, keep, eh, sh, dh, ctx, backout_layer;
  float scale, floor_;
  int q8;
  const bf16 *embed, *embed2, *lm_head, *feat, *smear_gate, *cosv, *sinv, *bfirst, *bres[2], *bsq, *bmeta, *bout,
      *skip_gate[3];
  const float *scal, *x0l;
  std::vector<float> w1, wr, wsq;  // board CNN weights, FP32, [piece][tap][co], [conv][tap][ci][co], [o][ci]
  std::vector<float> cnn[2];        // board CNN activations shared by the threads, [10 * 10][32]
  std::vector<const bf16*> ve;
  std::vector<Layer> layers;
  Pool* pool;

  // one step
  int T, S;
  const int64_t* ids;
  const float* feats;
  const uint8_t* boards;
  float* out;
  std::vector<Seq> seq;
  std::vector<int> tseq, last, units;
  std::vector<float> f64, b544, clk, brd, e, x, x0, x02, h, hf, qkv, g, q, y, tmp, skip[3], bko, rs, gate, acc, hid,
      shid, xf, z;
  std::vector<int> idx, cnt, start, at, stok;
  std::vector<float> sgate;
  std::vector<const float*> ph, py, phf, pxf, ptok, phid, pshid;
  std::vector<Seg> segs;
  std::vector<Chunk> upch;
  alignas(64) std::atomic<int> ctr[2];
  struct alignas(64) Count {
    int v;
  };
  std::vector<Count> phase;  // per thread: phases passed this step
  std::vector<std::vector<float>> scratch;
  struct Priv {  // a thread's own copy of the per-token steps (solo)
    std::vector<float> x, h, hf, g;
    std::vector<const float*> ph, phf;
  };
  std::vector<Priv> priv;
  int prof = 0;
  double ptime[NPHASE] = {0}, plast = 0;

  // barrier; thread 0 charges the time since the last one to phase p, and clears the work
  // counter of phase p for the phase after next
  void sync(int t, int p) {
    pool->barrier();
    if (t == 0) {
      ctr[phase[0].v & 1].store(0, std::memory_order_relaxed);
      if (prof) {
        double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        ptime[p] += now - plast;
        plast = now;
      }
    }
    phase[t].v++;
  }
  template <class F>
  void chunks(int t, int n, F f) {  // f(c) for c < n, shared out as threads come free
    std::atomic<int>& c = ctr[phase[t].v & 1];
    for (int i; (i = c.fetch_add(1, std::memory_order_relaxed)) < n;) f(i);
  }
  void embedding(int t, int nt);
  void resid(int i, int k, const float* src, float* dst, float* hk, float* gk);
  void rotary(int i, int k, int hh, const float* gk);
  void attend(int i, int k, int hh, const float* gk, float* sc);
  void blockstep(int i, int t, int nt);
  void ffn(int t, int nt, int dense, int i, bool solo);
  void head(int t, int nt);
  void run(int t) {
    int nt = pool->n;
    embedding(t, nt);
    for (int i = 0; i < L; i++) blockstep(i, t, nt);
    head(t, nt);
  }
};

static void trampoline(void* a, int t) { ((Engine*)a)->run(t); }

static void clockfeat(const float* t, float* f) {
  for (int j = 0; j < 64; j++) f[j] = 0;
  for (int j = 0; j < 3; j++) {
    if (!(t[j] >= 0)) continue;
    float v = log1pf(std::max(t[j], 0.f)) / 10.f;
    for (int k = 0; k < 8; k++) {
      float a = v * ((float)3.141592653589793 * (float)(1 << k));
      f[j * 18 + k] = rb(sinf(a));
      f[j * 18 + 8 + k] = rb(cosf(a));
    }
    f[j * 18 + 16] = rb(v);
    f[j * 18 + 17] = 1.f;
  }
}

// the board CNN on channels-last activations with a zero border: [10 * 10][32]
static inline int padded(int p) { return (p / 8 + 1) * 10 + p % 8 + 1; }

static void board_pieces(const uint8_t* st, int* piece) {
  for (int q = 0; q < 100; q++) piece[q] = 13;  // off the board: a zero weight row
  for (int p = 0; p < 64; p++) piece[padded(p)] = st[p];
}

// pass 0: the first conv on the one-hot board; passes 1, 2: residual convs. Squares [p0, p1)
// of out = gelu(conv(a)) (+ a for the residual passes), each rounded to BF16
static void board_pass(const Engine& m, int pass, const int* piece, const float* a, float* out, int p0, int p1) {
  enum { C = 32 / VL, NP = 8 / C };  // NP squares at once: eight independent sums
  alignas(64) float y[NP * 32];
  for (int p = p0; p < p1; p += NP) {
    vf acc[NP][C];
    for (int j = 0; j < NP; j++)
      for (int c = 0; c < C; c++) acc[j][c] = vzero();
    for (int tap = 0; tap < 9; tap++) {
      int q[NP];
      for (int j = 0; j < NP; j++) q[j] = ((p + j) / 8 + tap / 3) * 10 + (p + j) % 8 + tap % 3;
      if (pass == 0) {  // one-hot input: the weights of the piece on each neighbour
        for (int j = 0; j < NP; j++)
          for (int c = 0; c < C; c++) acc[j][c] = vadd(acc[j][c], vld(&m.w1[(piece[q[j]] * 9 + tap) * 32 + c * VL]));
        continue;
      }
      const float* w = &m.wr[((pass - 1) * 9 + tap) * 32 * 32];
      for (int ci = 0; ci < 32; ci++) {
        vf wc[C];
        for (int c = 0; c < C; c++) wc[c] = vld(w + ci * 32 + c * VL);
        for (int j = 0; j < NP; j++) {
          vf x = vset(a[q[j] * 32 + ci]);
          for (int c = 0; c < C; c++) acc[j][c] = vfma(x, wc[c], acc[j][c]);
        }
      }
    }
    for (int j = 0; j < NP; j++)
      for (int c = 0; c < C; c++) vst(y + j * 32 + c * VL, vrb(acc[j][c]));
    gelu(y, NP * 32);
    for (int j = 0; j < NP; j++) {
      const float* ap = a + padded(p + j) * 32;
      float* o = out + padded(p + j) * 32;
      for (int c = 0; c < 32; c++) o[c] = pass ? rb(ap[c] + y[j * 32 + c]) : y[j * 32 + c];
    }
  }
}

// squeeze, the castling / en-passant / side features, layer norm: board.output's 544 inputs
static void board_final(const Engine& m, const uint8_t* st, const float* a, float* v) {
  for (int o = 0; o < 8; o++)
    for (int p = 0; p < 64; p++) {
      const float* ap = a + padded(p) * 32;
      float s = 0;
      for (int c = 0; c < 32; c++) s += m.wsq[o * 32 + c] * ap[c];
      v[o * 64 + p] = rb(s);
    }
  for (int j = 0; j < 32; j++)
    v[512 + j] = rb(f32(m.bmeta[st[64] * 32 + j]) + f32(m.bmeta[(2 + st[65]) * 32 + j]) +
                    f32(m.bmeta[(18 + st[66]) * 32 + j]));
  float mean = 0, var = 0;
  for (int j = 0; j < 544; j++) mean += v[j];
  mean /= 544;
  for (int j = 0; j < 544; j++) var += (v[j] - mean) * (v[j] - mean);
  float r = 1.f / sqrtf(var / 544 + 1e-5f);
  for (int j = 0; j < 544; j++) v[j] = rb((v[j] - mean) * r);
}

void Engine::embedding(int t, int nt) {
  if (T >= nt) {  // a token per thread
    alignas(64) float a[2][100 * 32] = {};
    int piece[100];
    for (int k = t; k < T; k += nt) {
      clockfeat(feats + 3 * k, &f64[k * 64]);
      board_pieces(boards + 68 * k, piece);
      for (int pass = 0; pass < 3; pass++) board_pass(*this, pass, piece, a[pass & 1], a[~pass & 1], 0, 64);
      board_final(*this, boards + 68 * k, a[1], &b544[k * 544]);
    }
  } else {  // all threads on each token: a share of the squares, a barrier between passes
    int piece[100], p0, p1;
    split(64, t, nt, 8 / (32 / VL), p0, p1);
    for (int k = 0; k < T; k++) {
      if (t == 0) clockfeat(feats + 3 * k, &f64[k * 64]);
      board_pieces(boards + 68 * k, piece);
      for (int pass = 0; pass < 3; pass++) {
        board_pass(*this, pass, piece, &cnn[pass & 1][0], &cnn[~pass & 1][0], p0, p1);
        pool->barrier();
      }
      if (t == 0) board_final(*this, boards + 68 * k, &cnn[1][0], &b544[k * 544]);
      if (k + 1 < T) pool->barrier();  // the next token's first pass overwrites cnn[1]
    }
  }
  sync(t, E_CNN);
  int d0, d1;
  split(D, t, nt, 2 * VL, d0, d1);
  kn(f64.data(), 64, T, feat, D, d0, d1, clk.data());
  kn(b544.data(), 544, T, bout, D, d0, d1, brd.data());
  for (int k = 0; k < T; k++) {
    const bf16* em = embed + (size_t)ids[k] * D;
    float flag = boards[68 * k + 67];
    for (int d = d0; d < d1; d++) {
      size_t i = (size_t)k * D + d;
      e[i] = rb(rb(f32(em[d]) + rb(clk[i])) + rb(rb(brd[i]) * flag));
    }
  }
  sync(t, E_ROWS);
  float smear = scal[3 * L];
  std::vector<float>& prev = scratch[t];
  for (int k = t; k < T; k += nt) {
    const Seq& s = seq[tseq[k]];
    int64_t j = k - s.off;
    float* ek = &e[(size_t)k * D];
    if (j > 0)
      memcpy(prev.data(), ek - D, D * sizeof(float));
    else if (s.n0 > 0)
      for (int d = 0; d < D; d++) prev[d] = f32(s.e[(s.n0 - 1) * D + d]);
    else
      memset(prev.data(), 0, D * sizeof(float));
    float c = rb(smear * rb(sigm(rb(dot(ek, smear_gate, 16)))));
    float* xk = &x[(size_t)k * D];
    axpy_rb(xk, ek, c, prev.data(), D);
    norm(xk, xk, D);
    memcpy(&x0[(size_t)k * D], xk, D * sizeof(float));
    const bf16* e2 = embed2 + (size_t)ids[k] * D;
    float* x2 = &x02[(size_t)k * D];
    for (int d = 0; d < D; d++) x2[d] = f32(e2[d]);
    norm(x2, x2, D);
    bf16* ce = s.e + (s.n0 + j) * D;
    for (int d = 0; d < D; d++) ce[d] = tobf(ek[d]);
  }
  sync(t, E_SMEAR);
}

// token k's residual stream from src into dst: skip connection, x0 blend; hk = norm(dst); its gates
void Engine::resid(int i, int k, const float* src, float* dst, float* hk, float* gk) {
  const Layer& ly = layers[i];
  const float *a = &x0[(size_t)k * D], *b = &x02[(size_t)k * D];
  if (ly.skout >= 0) {
    int j = ly.skout;
    float gs = sigm(scal[3 * L + 2 + j]) * 2;
    float gg = rb(gs * rb(sigm(rb(dot(a, skip_gate[j], 16)))));
    axpy_rb(dst, src, gg, &skip[2 - j][(size_t)k * D], D);
    src = dst;
  }
  float c0 = x0l[2 * i], c1 = x0l[2 * i + 1], lam = scal[i];
  int d = 0;
  if (i == 0) {
    float c = (float)((double)lam + (double)c0);
    for (; d + VL <= D; d += VL)
      vst(dst + d, vrb(vadd(vrb(vmul(vset(c), vld(src + d))), vrb(vmul(vset(c1), vld(b + d))))));
    for (; d < D; d++) dst[d] = rb(rb(c * src[d]) + rb(c1 * b[d]));
  } else {
    for (; d + VL <= D; d += VL) {
      vf s = vrb(vadd(vrb(vmul(vset(c0), vld(a + d))), vrb(vmul(vset(c1), vld(b + d)))));
      vst(dst + d, vrb(vfma(vset(lam), vld(src + d), s)));
    }
    for (; d < D; d++) dst[d] = rb(rb(rb(c0 * a[d]) + rb(c1 * b[d])) + lam * src[d]);
  }
  norm(dst, hk, D);
  for (int r = 0; r < ly.G; r++) gk[r] = rb(sigm(rb(dot(hk, ly.gates + r * 16, 16))));
}

// token k, head hh: q and k normed and rotated, v (+ value embedding); k and v into the cache
void Engine::rotary(int i, int k, int hh, const float* gk) {
  const Layer& ly = layers[i];
  const Seq& s = seq[tseq[k]];
  int64_t p = s.n0 + (k - s.off);
  int half = hd / 2;
  const float* row = &qkv[(size_t)k * 3 * D];
  float qn[256], kk[256];
  norm(row + hh * hd, qn, hd);
  norm(row + (H + hh) * hd, kk, hd);
  const bf16 *cs = cosv + p * half, *sn = sinv + p * half;
  float* qo = &q[(size_t)k * D + hh * hd];
  bf16* kc = s.k + (((size_t)i * H + hh) * s.cap + p) * hd;
  bf16* vc = s.v + (((size_t)i * H + hh) * s.cap + p) * hd;
  for (int c = 0; c < half; c++) {
    float co = f32(cs[c]), si = f32(sn[c]);
    qo[c] = rb(rb(qn[c] * co) + rb(qn[c + half] * si));
    qo[c + half] = rb(rb(qn[c + half] * co) - rb(qn[c] * si));
    kc[c] = tobf(rb(rb(kk[c] * co) + rb(kk[c + half] * si)));
    kc[c + half] = tobf(rb(rb(kk[c + half] * co) - rb(kk[c] * si)));
  }
  const float* v = row + (2 * H + hh) * hd;
  if (ly.ve >= 0) {
    float g2 = rb(2 * gk[H + hh]);
    const bf16* vr = ve[ly.ve] + (size_t)ids[k] * D + hh * hd;
    for (int c = 0; c < hd; c++) vc[c] = tobf(rb(v[c] + rb(g2 * f32(vr[c]))));
  } else {
    for (int c = 0; c < hd; c++) vc[c] = tobf(v[c]);
  }
}

// token k's attention in head hh over its game's cache, times the output gate; sc: scores
void Engine::attend(int i, int k, int hh, const float* gk, float* sc) {
  const Seq& s = seq[tseq[k]];
  const bf16* K = s.k + ((size_t)i * H + hh) * s.cap * hd;
  const bf16* Vv = s.v + ((size_t)i * H + hh) * s.cap * hd;
  int64_t p = s.n0 + (k - s.off);
  const float* qv = &q[(size_t)k * D + hh * hd];
  vf qr[16];
  for (int c = 0; c < hd / VL; c++) qr[c] = vld(qv + c * VL);
  float mx = -INFINITY;
  for (int64_t r = 0; r <= p; r++) {
    vf a = vzero();
    for (int c = 0; c < hd / VL; c++) a = vfma(qr[c], vld(K + r * hd + c * VL), a);
    sc[r] = vsum(a) * scale;
    mx = std::max(mx, sc[r]);
  }
  int64_t r = 0, n = p + 1;
  vf vs = vzero();
  for (; r + VL <= n; r += VL) {
    vf ex = vexp(vsub(vld(sc + r), vset(mx)));
    vst(sc + r, ex);
    vs = vadd(vs, ex);
  }
  float sum = vsum(vs);
  for (; r < n; r++) sum += sc[r] = expf(sc[r] - mx);
  vf o[16];
  for (int c = 0; c < hd / VL; c++) o[c] = vzero();
  for (r = 0; r < n; r++) {
    vf pr = vset(sc[r]);
    for (int c = 0; c < hd / VL; c++) o[c] = vfma(pr, vld(Vv + r * hd + c * VL), o[c]);
  }
  float* yo = &y[(size_t)k * D + hh * hd];
  float og = gk[hh], ov[256];
  for (int c = 0; c < hd / VL; c++) vst(ov + c * VL, o[c]);
  for (int c = 0; c < hd; c++) yo[c] = rb(rb(ov[c] / sum) * og);
}

void Engine::blockstep(int i, int t, int nt) {
  const Layer& ly = layers[i];
  int G = ly.G;
  // solo: one or two tokens; every thread does the per-token steps itself, into its own buffers,
  // rather than wait at a barrier for one thread
  bool solo = T <= 2, dense = ly.fc.w != nullptr;
  Priv& pv = priv[t];
  float *xs = solo ? pv.x.data() : x.data(), *gs = solo ? pv.g.data() : g.data();
  for (int k = solo ? 0 : t; k < T; k += solo ? 1 : nt)
    resid(i, k, &x[(size_t)k * D], &xs[(size_t)k * D], solo ? &pv.h[(size_t)k * D] : &h[(size_t)k * D], &gs[k * G]);
  if (!solo) sync(t, B_NORM);
  const float* const* hp = solo ? pv.ph.data() : ph.data();
  chunks(t, (3 * D + 63) / 64, [&](int c) { mm(ly.qkv, D, 64 * c, std::min(64, 3 * D - 64 * c), hp, T, &qkv[64 * c], 3 * D); });
  sync(t, B_QKV);
  float* sc = scratch[t].data();
  if (T == S) {  // one new token a game: rotary and attention in one pass per (game, head)
    for (int u = t; u < T * H; u += nt) {
      rotary(i, u / H, u % H, &gs[(u / H) * G]);
      attend(i, u / H, u % H, &gs[(u / H) * G], sc);
    }
  } else {
    for (int u = t; u < T * H; u += nt) rotary(i, u / H, u % H, &gs[(u / H) * G]);
    sync(t, B_ROTARY);
    int nu = (int)units.size() / 3;  // (game, head, query block)
    for (int u = t; u < nu * H; u += nt)
      for (int k = units[3 * (u / H) + 1]; k < units[3 * (u / H) + 2]; k++) attend(i, k, u % H, &gs[k * G], sc);
  }
  sync(t, B_ATTN);
  chunks(t, (D + 31) / 32, [&](int c) {
    int lo = 32 * c, n = std::min(32, D - lo);
    mm(ly.o, D, lo, n, py.data(), T, &tmp[lo], D);
    for (int k = 0; k < T; k++) {
      size_t a = (size_t)k * D + lo;
      int j = 0;
      for (; j + VL <= n; j += VL) vst(&x[a + j], vrb(vadd(vld(&xs[a + j]), vld(&tmp[a + j]))));
      for (; j < n; j++) x[a + j] = rb(xs[a + j] + tmp[a + j]);
    }
  });
  sync(t, B_O);
  for (int k = solo ? 0 : t; k < T; k += solo ? 1 : nt) {
    float* hk = solo ? &pv.h[(size_t)k * D] : &h[(size_t)k * D];
    norm(&x[(size_t)k * D], hk, D);
    if (!dense) {
      float* fk = solo ? &pv.hf[(size_t)k * D] : &hf[(size_t)k * D];
      for (int d = 0; d < D; d++) fk[d] = hk[d] - ly.mu[d];
    }
    if (solo && t == 0) memcpy(&h[(size_t)k * D], hk, D * sizeof(float));  // for the experts
  }
  if (!solo || dense) sync(t, B_NORM2);  // solo: the router reads each thread's own copy
  ffn(t, nt, dense, i, solo);
}

static void route(Engine& m, const Layer& ly, int k) {  // top-k of one token, its gates
  const float* s = &m.rs[(size_t)k * m.E];
  float best[64];
  int bi[64], n = 0;
  for (int e = 0; e < m.E; e++) {
    float v = s[e] + ly.bias[e];
    if (n == m.topk && v <= best[n - 1]) continue;
    int j = n < m.topk ? n++ : n - 1;
    while (j > 0 && best[j - 1] < v) best[j] = best[j - 1], bi[j] = bi[j - 1], j--;
    best[j] = v, bi[j] = e;
  }
  float sum = 0;
  for (int j = 0; j < m.topk; j++) sum += s[bi[j]];
  float f = (float)std::sqrt((double)m.topk) / std::max(sum, m.floor_);
  for (int j = 0; j < m.keep; j++) {
    m.idx[(size_t)k * m.keep + j] = bi[j];
    m.gate[(size_t)k * m.keep + j] = rb(s[bi[j]] * f);
  }
}

void Engine::ffn(int t, int nt, int dense, int i, bool solo) {
  const Layer& ly = layers[i];
  if (dense) {
    if (t == 0) segs.assign(1, Seg{ly.fc, dh, T, std::max(sh, dh), ph.data(), shid.data()});
  } else {
    const float* const* fp = solo ? priv[t].phf.data() : phf.data();
    chunks(t, (E + 15) / 16, [&](int c) {
      int lo = 16 * c, n = std::min(16, E - lo);
      mm(Mat{ly.router, nullptr, 2}, D, lo, n, fp, T, &rs[lo], E);
      for (int k = 0; k < T; k++)
        for (int e = lo; e < lo + n; e++) rs[(size_t)k * E + e] = sigm(rs[(size_t)k * E + e]);
    });
    sync(t, B_ROUTER);
    if (!solo) {
      for (int k = t; k < T; k += nt) route(*this, ly, k);
      sync(t, B_TOPK);
    }
    if (t == 0) {  // tokens grouped by expert: each expert's weights read once
      if (solo)
        for (int k = 0; k < T; k++) route(*this, ly, k);
      std::fill(cnt.begin(), cnt.end(), 0);
      int na = T * keep;
      for (int a = 0; a < na; a++) cnt[idx[a]]++;
      for (int e = 0, s = 0; e < E; e++) start[e] = at[e] = s, s += cnt[e];
      for (int a = 0; a < na; a++) {
        int sl = at[idx[a]]++;
        stok[sl] = a / keep;
        sgate[sl] = gate[a];
        ptok[sl] = &h[(size_t)(a / keep) * D];
      }
      segs.assign(1, Seg{ly.sup, sh, T, std::max(sh, dh), ph.data(), shid.data()});
      size_t es = (size_t)2 * eh * D * (q8 ? 1 : 2);
      for (int e = 0; e < E; e++)
        if (cnt[e]) {
          Mat A{(const char*)ly.up + e * es, q8 ? ly.ups + (size_t)e * 2 * eh : nullptr, q8};
          segs.push_back(Seg{A, eh, cnt[e], eh, &ptok[start[e]], &hid[(size_t)start[e] * eh]});
        }
    }
  }
  if (t == 0) {  // 32 pairs (64 rows) a chunk, the shared expert's (most tokens) first
    upch.clear();
    for (int j = 0; j < (int)segs.size(); j++)
      for (int p = 0; p < segs[j].pairs; p += 32) upch.push_back(Chunk{j, p, std::min(32, segs[j].pairs - p)});
  }
  sync(t, B_GROUP);
  // up (gate and value halves), SwiGLU
  float* ta = scratch[t].data();
  chunks(t, (int)upch.size(), [&](int c) {
    const Chunk& u = upch[c];
    const Seg& s = segs[u.seg];
    float *a = ta, *b = ta + (size_t)s.ntok * u.n;
    mm(s.A, D, u.p0, u.n, s.x, s.ntok, a, u.n);
    mm(s.A, D, s.pairs + u.p0, u.n, s.x, s.ntok, b, u.n);
    for (int m = 0; m < s.ntok; m++)
      silu_mul(a + (size_t)m * u.n, b + (size_t)m * u.n, s.out + (size_t)m * s.ldo + u.p0, u.n);
  });
  sync(t, B_UP);
  // down: 16 output features a chunk, summed over the tokens' experts
  float* tb = scratch[t].data();
  chunks(t, (D + 15) / 16, [&](int c) {
    int lo = 16 * c, n = std::min(16, D - lo);
    if (dense) {
      mm(ly.proj, dh, lo, n, pshid.data(), T, tb, n);
      for (int k = 0; k < T; k++) add_rb(&x[(size_t)k * D + lo], &tb[(size_t)k * n], n);
    } else {
      for (int k = 0; k < T; k++) memset(&acc[(size_t)k * D + lo], 0, n * sizeof(float));
      size_t es = (size_t)D * eh * (q8 ? 1 : 2);
      for (int e = 0; e < E; e++) {
        if (!cnt[e]) continue;
        Mat A{(const char*)ly.down + e * es, q8 ? ly.downs + (size_t)e * D : nullptr, q8};
        int s0 = start[e];
        mm(A, eh, lo, n, &phid[s0], cnt[e], tb, n);
        for (int m = 0; m < cnt[e]; m++) {
          float gt = sgate[s0 + m], *ak = &acc[(size_t)stok[s0 + m] * D + lo];
          const float* bm = &tb[(size_t)m * n];
          int j = 0;
          for (; j + VL <= n; j += VL) vst(ak + j, vfma(vset(gt), vld(bm + j), vld(ak + j)));
          for (; j < n; j++) ak[j] += gt * bm[j];
        }
      }
      mm(ly.sdown, sh, lo, n, pshid.data(), T, tb, n);
      for (int k = 0; k < T; k++) {
        float *xa = &x[(size_t)k * D + lo], *aa = &acc[(size_t)k * D + lo];
        const float* bb = &tb[(size_t)k * n];
        int j = 0;
        for (; j + VL <= n; j += VL) vst(xa + j, vrb(vadd(vld(xa + j), vrb(vadd(vrb(vld(aa + j)), vld(bb + j))))));
        for (; j < n; j++) xa[j] = rb(xa[j] + rb(rb(aa[j]) + bb[j]));
      }
    }
    for (int k = 0; k < T; k++) {
      size_t a = (size_t)k * D + lo;
      if (ly.skin >= 0) memcpy(&skip[ly.skin][a], &x[a], n * sizeof(float));
      if (i == backout_layer) memcpy(&bko[a], &x[a], n * sizeof(float));
    }
  });
  sync(t, B_DOWN);
}

void Engine::head(int t, int nt) {
  float b = scal[3 * L + 1];
  for (int s = t; s < S; s += nt) {
    int k = last[s];
    float* xs = &xf[(size_t)s * D];
    for (int d = 0; d < D; d++) xs[d] = rb(x[(size_t)k * D + d] - rb(b * bko[(size_t)k * D + d]));
    norm(xs, xs, D);
  }
  sync(t, H_NORM);
  chunks(t, (V + 63) / 64, [&](int c) {
    int lo = 64 * c, hi = std::min(V, lo + 64);
    mm(Mat{lm_head, nullptr, 0}, D, lo, hi - lo, pxf.data(), S, &z[lo], V);
    for (int s = 0; s < S; s++)
      for (int v = lo; v < hi; v++) out[(size_t)s * V + v] = 23.f * sigm((z[(size_t)s * V + v] + 5.f) / 7.5f);
  });
  sync(t, H_HEAD);
}

extern "C" {

// cfg: L D H hd V nve E topk keep eh sh dh int8 ctx backout_layer threads (pin), then per layer
// G ve skip_in skip_out. glob: embed embed2 lm_head feat_embed smear_gate scalars x0_lambdas cos sin
// board.first board.residual.0 board.residual.1 board.squeeze board.meta board.output
// skip_gate.0-2 value_embed.*. per layer (20): qkv qkv_s o o_s gates fc fc_s proj proj_s router
// moe_bias mu up up_s down down_s shared_up shared_up_s shared_down shared_down_s
void* allie_new(const int64_t* cfg, const double* fl, void* const* glob, void* const* lay, const int32_t* cpus,
                double spin) {
  Engine* m = new Engine();
  m->L = cfg[0], m->D = cfg[1], m->H = cfg[2], m->hd = cfg[3], m->V = cfg[4], m->nve = cfg[5], m->E = cfg[6];
  m->topk = cfg[7], m->keep = cfg[8], m->eh = cfg[9], m->sh = cfg[10], m->dh = cfg[11], m->q8 = cfg[12];
  m->ctx = cfg[13], m->backout_layer = cfg[14];
  m->scale = fl[0], m->floor_ = fl[1];
  auto B = [&](int j) { return (const bf16*)glob[j]; };
  m->embed = B(0), m->embed2 = B(1), m->lm_head = B(2), m->feat = B(3), m->smear_gate = B(4);
  m->scal = (const float*)glob[5], m->x0l = (const float*)glob[6];
  m->cosv = B(7), m->sinv = B(8), m->bfirst = B(9), m->bres[0] = B(10), m->bres[1] = B(11);
  m->bsq = B(12), m->bmeta = B(13), m->bout = B(14);
  for (int j = 0; j < 3; j++) m->skip_gate[j] = B(15 + j);
  m->cnn[0].assign(100 * 32, 0.f), m->cnn[1].assign(100 * 32, 0.f);
  m->w1.assign(14 * 9 * 32, 0.f), m->wr.resize(2 * 9 * 32 * 32), m->wsq.resize(8 * 32);
  for (int co = 0; co < 32; co++)
    for (int tap = 0; tap < 9; tap++) {
      for (int ci = 0; ci < 13; ci++) m->w1[(ci * 9 + tap) * 32 + co] = f32(m->bfirst[(co * 13 + ci) * 9 + tap]);
      for (int j = 0; j < 2; j++)
        for (int ci = 0; ci < 32; ci++) m->wr[((j * 9 + tap) * 32 + ci) * 32 + co] = f32(m->bres[j][(co * 32 + ci) * 9 + tap]);
    }
  for (int i = 0; i < 8 * 32; i++) m->wsq[i] = f32(m->bsq[i]);
  for (int j = 0; j < m->nve; j++) m->ve.push_back(B(18 + j));
  int t8 = m->q8 ? 1 : 0;
  for (int i = 0; i < m->L; i++) {
    void* const* p = lay + 20 * i;
    const int64_t* c = cfg + 17 + 4 * i;
    Layer ly;
    ly.qkv = {p[0], (const bf16*)p[1], t8};
    ly.o = {p[2], (const bf16*)p[3], t8};
    ly.gates = (const bf16*)p[4];
    ly.fc = {p[5], (const bf16*)p[6], t8};
    ly.proj = {p[7], (const bf16*)p[8], t8};
    ly.router = (const float*)p[9], ly.bias = (const float*)p[10], ly.mu = (const float*)p[11];
    ly.up = p[12], ly.ups = (const bf16*)p[13], ly.down = p[14], ly.downs = (const bf16*)p[15];
    ly.sup = {p[16], (const bf16*)p[17], t8};
    ly.sdown = {p[18], (const bf16*)p[19], t8};
    ly.G = c[0], ly.ve = c[1], ly.skin = c[2], ly.skout = c[3];
    m->layers.push_back(ly);
  }
  m->pool = new Pool(cfg[15], cpus, spin);
  return m;
}

void allie_free(void* h) {
  Engine* m = (Engine*)h;
  delete m->pool;
  delete m;
}

int allie_threads(void* h) { return ((Engine*)h)->pool->n; }

void allie_wake(void* h) { ((Engine*)h)->pool->wake(); }

// meta: per sequence n0 len cap off; caches: per sequence k v e; out: S x V. Returns 0, or 1 for a
// token outside the vocabulary, 2 for a board state outside its ranges, 3 for spans that do not fit
int allie_step(void* h, int T, int S, const int64_t* ids, const float* feats, const uint8_t* boards,
               const int64_t* meta, void* const* caches, float* out) {
  Engine& m = *(Engine*)h;
  int D = m.D, nt = m.pool->n;
  for (int k = 0; k < T; k++) {
    const uint8_t* b = boards + 68 * k;
    if (ids[k] < 0 || ids[k] >= m.V) return 1;
    if (*std::max_element(b, b + 64) > 12 || b[64] > 1 || b[65] > 15 || b[66] > 8) return 2;
  }
  for (int s = 0, off = 0; s < S; s++) {
    const int64_t* q = meta + 4 * s;
    if (q[1] < 1 || q[0] < 0 || q[0] + q[1] > std::min<int64_t>(q[2], m.ctx) || q[3] != off) return 3;
    off += q[1];
    if (s == S - 1 && off != T) return 3;
  }
  m.T = T, m.S = S, m.ids = ids, m.feats = feats, m.boards = boards, m.out = out;
  m.seq.resize(S);
  m.tseq.resize(T);
  m.last.resize(S);
  m.units.clear();
  int qc = std::max(1, std::min(32, (int)(T * m.H / (4 * nt)) + 1));
  for (int s = 0; s < S; s++) {
    Seq& q = m.seq[s];
    q.n0 = meta[4 * s], q.len = meta[4 * s + 1], q.cap = meta[4 * s + 2], q.off = meta[4 * s + 3];
    q.k = (bf16*)caches[3 * s], q.v = (bf16*)caches[3 * s + 1], q.e = (bf16*)caches[3 * s + 2];
    for (int64_t j = 0; j < q.len; j++) m.tseq[q.off + j] = s;
    m.last[s] = q.off + q.len - 1;
    for (int64_t a = 0; a < q.len; a += qc) {
      m.units.push_back(s);
      m.units.push_back(q.off + a);
      m.units.push_back(q.off + std::min<int64_t>(q.len, a + qc));
    }
  }
  size_t TD = (size_t)T * D;
  for (auto* v : {&m.clk, &m.brd, &m.e, &m.x, &m.x0, &m.x02, &m.h, &m.hf, &m.q, &m.y, &m.tmp, &m.skip[0], &m.skip[1],
                  &m.skip[2], &m.bko, &m.acc})
    if (v->size() < TD) v->resize(TD);
  auto grow = [](std::vector<float>& v, size_t n) { if (v.size() < n) v.resize(n); };
  grow(m.f64, (size_t)T * 64);
  grow(m.b544, (size_t)T * 544);
  grow(m.qkv, 3 * TD);
  grow(m.g, (size_t)T * 2 * m.H);
  grow(m.rs, (size_t)T * m.E);
  grow(m.gate, (size_t)T * m.keep);
  grow(m.sgate, (size_t)T * m.keep);
  grow(m.hid, (size_t)T * m.keep * m.eh);
  grow(m.shid, (size_t)T * std::max(m.sh, m.dh));
  grow(m.xf, (size_t)S * D);
  grow(m.z, (size_t)S * m.V);
  if (m.idx.size() < (size_t)T * m.keep) m.idx.resize((size_t)T * m.keep), m.stok.resize((size_t)T * m.keep);
  m.cnt.resize(m.E), m.start.resize(m.E), m.at.resize(m.E);
  m.ph.resize(T), m.py.resize(T), m.phf.resize(T), m.pshid.resize(T), m.pxf.resize(S);
  m.ptok.resize((size_t)T * m.keep), m.phid.resize((size_t)T * m.keep);
  for (int k = 0; k < T; k++) {
    m.ph[k] = &m.h[(size_t)k * D], m.py[k] = &m.y[(size_t)k * D], m.phf[k] = &m.hf[(size_t)k * D];
    m.pshid[k] = &m.shid[(size_t)k * std::max(m.sh, m.dh)];
  }
  for (int s = 0; s < S; s++) m.pxf[s] = &m.xf[(size_t)s * D];
  for (size_t a = 0; a < (size_t)T * m.keep; a++) m.phid[a] = &m.hid[a * m.eh];
  size_t need = std::max({(size_t)m.ctx + 64, (size_t)D, (size_t)T * 128, (size_t)T * (D / nt + 8)});
  m.scratch.resize(nt);
  for (auto& s : m.scratch)
    if (s.size() < need) s.resize(need);
  m.priv.resize(nt);
  if (T <= 2)
    for (auto& p : m.priv) {
      for (auto* v : {&p.x, &p.h, &p.hf}) v->resize(TD);
      p.g.resize((size_t)T * 2 * m.H);
      p.ph.resize(T), p.phf.resize(T);
      for (int k = 0; k < T; k++) p.ph[k] = &p.h[(size_t)k * D], p.phf[k] = &p.hf[(size_t)k * D];
    }
  if (m.prof) m.plast = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  m.phase.assign(nt, Engine::Count{0});
  m.ctr[0].store(0), m.ctr[1].store(0);
  m.pool->run(trampoline, &m);
  return 0;
}

// seconds spent in each phase since the last call (out: NPHASE doubles); on: keep profiling
int allie_profile(void* h, double* out, int on) {
  Engine& m = *(Engine*)h;
  for (int j = 0; j < NPHASE; j++) out[j] = m.ptime[j], m.ptime[j] = 0;
  m.prof = on;
  return NPHASE;
}

// best read bandwidth (GB/s) of `threads` threads streaming a buffer of `bytes` (first touched
// by the thread that reads it), over `reps` passes
double allie_bandwidth(int threads, int64_t bytes, int reps, const int32_t* cpus) {
  Pool pool(threads, cpus, 0.01);
  size_t n = bytes / sizeof(float) / threads / 64 * 64;
  std::vector<float*> parts(threads);
  struct Arg { std::vector<float*>* parts; size_t n; int init; std::vector<float> sums; } arg{&parts, n, 1, std::vector<float>(threads)};
  auto body = [](void* a, int t) {
    Arg& g = *(Arg*)a;
    if (g.init) {
      (*g.parts)[t] = (float*)aligned_alloc(64, g.n * sizeof(float));
      for (size_t i = 0; i < g.n; i++) (*g.parts)[t][i] = (float)(i & 7);
      return;
    }
    const float* p = (*g.parts)[t];
    vf a0 = vzero(), a1 = vzero(), a2 = vzero(), a3 = vzero();
    for (size_t i = 0; i < g.n; i += 4 * VL) {
      a0 = vadd(a0, vld(p + i));
      a1 = vadd(a1, vld(p + i + VL));
      a2 = vadd(a2, vld(p + i + 2 * VL));
      a3 = vadd(a3, vld(p + i + 3 * VL));
    }
    g.sums[t] = vsum(vadd(vadd(a0, a1), vadd(a2, a3)));
  };
  pool.run(body, &arg);
  arg.init = 0;
  double best = 0;
  for (int r = 0; r < reps; r++) {
    auto t0 = std::chrono::steady_clock::now();
    pool.run(body, &arg);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    best = std::max(best, (double)n * threads * sizeof(float) / s / 1e9);
  }
  for (auto p : parts) free(p);
  return best;
}

// moves the pages of n ranges to NUMA nodes[0] (nn == 1) or interleaves them over nn nodes; returns
// the number of ranges the kernel refused
int allie_place(void* const* ptrs, const int64_t* bytes, int n, const int32_t* nodes, int nn) {
#if defined(__linux__) && defined(SYS_mbind)
  unsigned long mask[16] = {0};
  for (int j = 0; j < nn; j++)
    if (nodes[j] >= 0 && nodes[j] < 1024) mask[nodes[j] / 64] |= 1ul << (nodes[j] % 64);
  long page = sysconf(_SC_PAGESIZE), bad = 0;
  for (int i = 0; i < n; i++) {
    uintptr_t a = (uintptr_t)ptrs[i] / page * page, b = ((uintptr_t)ptrs[i] + bytes[i] + page - 1) / page * page;
    // MPOL_BIND = 2, MPOL_INTERLEAVE = 3, MPOL_MF_MOVE = 2
    if (b > a && syscall(SYS_mbind, a, b - a, nn == 1 ? 2 : 3, mask, 1025, 2) != 0) bad++;
  }
  return (int)bad;
#else
  return n;
#endif
}

const char* allie_isa() {
#if defined(__AVX512F__)
  return "avx512";
#elif defined(__AVX2__)
  return "avx2";
#else
  return "generic";
#endif
}
}
"""


def _signature():
    """The CPU the kernels are compiled for (-march=native): its model and feature flags."""
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
        keep = ("model name", "flags", "Features", "CPU part", "CPU implementer")
        return "\n".join(sorted({x for x in lines if x.startswith(keep)}))
    except OSError:
        return f"{platform.machine()} {platform.processor()}"


@functools.cache
def library():
    """The compiled kernels (ctypes), building them for this CPU on first use."""
    cxx = os.environ.get("CXX", "c++")
    march = os.environ.get("ALLIE_MARCH")  # e.g. "-march=haswell": an AVX2 build anywhere
    key = hashlib.sha256(f"{SOURCE}\n{cxx}\n{march}\n{_signature()}".encode()).hexdigest()[:16]
    cache = (
        Path(os.environ.get("ALLIE_CACHE", Path.home() / ".cache" / "allie"))
        / "kernels"
    )
    path = cache / f"allie-{key}.so"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        src, tmp = (
            cache / f"allie-{key}-{os.getpid()}.cpp",
            cache / f"allie-{key}-{os.getpid()}.so",
        )
        src.write_text(SOURCE)
        base = [
            cxx,
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            str(src),
            "-o",
            str(tmp),
        ]
        try:
            for arch in [march.split()] if march else (["-march=native"], ["-mcpu=native"], []):
                r = subprocess.run(base + arch, capture_output=True, text=True)
                if r.returncode == 0:
                    break
            else:
                raise RuntimeError(
                    f"cannot compile Allie's CPU kernels:\n{r.stderr[-3000:]}"
                )
            os.replace(tmp, path)
        finally:
            src.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)
    lib = ctypes.CDLL(str(path))
    P, I, D = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    lib.allie_new.restype, lib.allie_new.argtypes = P, [P, P, P, P, P, D]
    lib.allie_free.argtypes = [P]
    lib.allie_threads.restype, lib.allie_threads.argtypes = I, [P]
    lib.allie_wake.argtypes = [P]
    lib.allie_step.restype, lib.allie_step.argtypes = I, [P, I, I, P, P, P, P, P, P]
    lib.allie_bandwidth.restype = D
    lib.allie_bandwidth.argtypes = [I, ctypes.c_int64, I, P]
    lib.allie_place.restype, lib.allie_place.argtypes = I, [P, P, I, P, I]
    lib.allie_isa.restype = ctypes.c_char_p
    lib.allie_profile.restype, lib.allie_profile.argtypes = I, [P, P, I]
    return lib


def cpus():
    """The CPUs this process may run on."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:  # macOS, Windows
        return list(range(os.cpu_count() or 1))


def threads_default():
    return max(1, min(torch.get_num_threads(), len(cpus())))


def _sys(path, default=0):
    try:
        return int(Path(path).read_text().split(",")[0].split("-")[0])
    except (OSError, ValueError):
        return default


def cpu_order(n):
    """n CPUs of this process to pin threads to: within one NUMA node when they fit, spread over
    the node's L3 caches, one thread per core before the cores' second hardware threads."""
    ids = cpus()
    if len(ids) < n:
        return None
    path = "/sys/devices/system/cpu/cpu{}/"
    node = {c: next((int(d.name[4:]) for d in Path(path.format(c)).glob("node[0-9]*")), 0) for c in ids}
    l3 = {c: _sys(path.format(c) + "cache/index3/id") for c in ids}
    first = {c: _sys(path.format(c) + "topology/thread_siblings_list", c) == c for c in ids}
    nodes = sorted(set(node.values()), key=lambda k: -sum(node[c] == k for c in ids))
    big = [c for c in ids if node[c] == nodes[0]]
    out = []
    for k in nodes[:1] if len(big) >= n else nodes:
        for primary in (True, False):
            groups = {}
            for c in ids:
                if node[c] == k and first[c] == primary:
                    groups.setdefault(l3[c], []).append(c)
            lists = list(groups.values())
            out += [g[j] for j in range(max(map(len, lists), default=0)) for g in lists if j < len(g)]
    return out[:n], [node[c] for c in out[:n]]


PHASES = ("board cnn", "embedding rows", "smear", "norm", "qkv", "rotary", "attention", "o",
          "norm2", "router", "top-k", "group", "up", "down", "head norm", "head")  # fmt: skip
LAYER = ("qkv", "o", "gates", "fc", "proj", "router", "moe_bias", "mu", "up", "down", "shared_up",
         "shared_down")  # fmt: skip
SCALED = ("qkv", "o", "fc", "proj", "up", "down", "shared_up", "shared_down")


class Fast:
    """model.py's step() for a CPU Model with BF16 activations (int8 or BF16 matrices)."""

    def __init__(self, model, threads=None, pin=True, spin=0.002, place=True):
        assert model.device.type == "cpu" and model.dtype == torch.bfloat16
        c, w, n = model.config, model.w, model.layers
        assert model.head_dim % 16 == 0 and model.head_dim <= 256 and model.topk <= 64
        assert (
            w["board.first"].shape == (32, 13, 3, 3)
            and w["board.output"].shape[0] == 544
        )
        self.lib, self.model = library(), model
        self.threads = min(threads or threads_default(), len(cpus()))  # spinning: never oversubscribe
        int8 = bool(model.scales)
        ve = [None] * n
        for j in range(model.ve):
            ve[j], ve[n - model.ve + j] = j, j
        skip_in = [i * n // 16 for i in (2, 4, 6)]
        skip_out = [9 * n // 16 + i for i in range(3)]
        dense = next(
            (w[f"{i}.fc"].shape[0] // 2 for i in range(n) if f"{i}.fc" in w), 0
        )
        cfg = [n, model.width, model.heads, model.head_dim, c["vocab"], model.ve, c["experts"],
               model.topk, model.keep, c["expert_hidden"], c["shared_hidden"], dense, int(int8),
               w["cos"].shape[0], skip_out[-1], self.threads, int(pin)]  # fmt: skip
        for i in range(n):
            cfg += [w[f"{i}.gates"].shape[0], -1 if ve[i] is None else ve[i],
                    skip_in.index(i) if i in skip_in else -1,
                    skip_out.index(i) if i in skip_out else -1]  # fmt: skip
        glob = ["embed", "embed2", "lm_head", "feat_embed", "smear_gate", "scalars", "x0_lambdas",
                "cos", "sin", "board.first", "board.residual.0", "board.residual.1",
                "board.squeeze", "board.meta", "board.output", "skip_gate.0", "skip_gate.1",
                "skip_gate.2", *[f"value_embed.{j}" for j in range(model.ve)]]  # fmt: skip
        ptrs, self.tensors = [], []  # the kernels read these in place: keep them alive
        for k in glob:
            t = w[k]
            assert t.is_contiguous() and t.dtype == (
                torch.float32 if k in ("scalars", "x0_lambdas") else torch.bfloat16
            ), k
            ptrs.append(t.data_ptr())
            self.tensors.append(t)
        lay = []
        for i in range(n):
            for k in LAYER:
                t = w.get(f"{i}.{k}")
                if t is not None:
                    want = (
                        torch.float32
                        if k in ("router", "moe_bias", "mu")
                        else torch.int8
                        if int8 and k in SCALED
                        else torch.bfloat16
                    )
                    assert t.is_contiguous() and t.dtype == want, (i, k, t.dtype)
                lay.append(0 if t is None else t.data_ptr())
                self.tensors.append(t)
                if k in SCALED:
                    s = model.scales.get(f"{i}.{k}")
                    assert (s is not None) == (int8 and t is not None), (i, k)
                    lay.append(0 if s is None else s.data_ptr())
                    self.tensors.append(s)
        arr = lambda ty, xs: (ty * len(xs))(*xs)
        order = cpu_order(self.threads) if pin else None
        self.cpus, nodes = order or (None, None)
        place = place and os.environ.get("ALLIE_NUMA") != "0"
        if place and nodes and Path("/sys/devices/system/node/node1").exists():
            # weights on the NUMA node of the threads (bound), or spread over theirs (interleaved)
            ts = [t for t in self.tensors if t is not None]
            nodes = sorted(set(nodes))
            self.lib.allie_place(arr(ctypes.c_void_p, [t.data_ptr() for t in ts]),
                                 arr(ctypes.c_int64, [t.numel() * t.element_size() for t in ts]),
                                 len(ts), arr(ctypes.c_int32, nodes), len(nodes))  # fmt: skip
        self.args = (arr(ctypes.c_int64, cfg), arr(ctypes.c_double, [model.scale, model.floor]),
                     arr(ctypes.c_void_p, ptrs), arr(ctypes.c_void_p, lay),
                     arr(ctypes.c_int32, self.cpus) if self.cpus else None, spin)  # fmt: skip
        self.handle, self.pid = self.lib.allie_new(*self.args), os.getpid()

    def __del__(self):
        if getattr(self, "handle", None) and os.getpid() == self.pid:  # a fork's copy: leaked
            self.lib.allie_free(self.handle)
            self.handle = None

    def profile(self, on=True):
        """Seconds spent in each phase of step() since the last call; on: keep counting."""
        out = (ctypes.c_double * len(PHASES))()
        assert self.lib.allie_profile(self.handle, out, int(on)) == len(PHASES)
        return dict(zip(PHASES, out))

    def step(self, items):
        if os.getpid() != self.pid:  # a forked child has none of the pool's threads: new ones
            self.handle, self.pid = self.lib.allie_new(*self.args), os.getpid()
        self.lib.allie_wake(self.handle)  # workers wake while the inputs are gathered
        meta, caches, lo = [], [], 0
        for cache, ids, *_ in items:
            cache.reserve(cache.n + len(ids))
            meta += [cache.n, len(ids), cache.capacity, lo]
            caches += [cache.k.data_ptr(), cache.v.data_ptr(), cache.e.data_ptr()]
            lo += len(ids)
        cat = lambda j, dt: torch.cat([it[j] for it in items]).to(dt).contiguous()
        ids, feats, boards = (
            cat(1, torch.int64),
            cat(2, torch.float32),
            cat(3, torch.uint8),
        )
        out = torch.empty(len(items), self.model.config["vocab"])
        err = self.lib.allie_step(
            self.handle, lo, len(items), ids.data_ptr(), feats.data_ptr(), boards.data_ptr(),
            (ctypes.c_int64 * len(meta))(*meta), (ctypes.c_void_p * len(caches))(*caches),
            out.data_ptr(),
        )  # fmt: skip
        if err:
            raise ValueError(("token outside the vocabulary", "board state out of range",
                              "tokens past the cache or the context")[err - 1])  # fmt: skip
        for cache, ids, *_ in items:
            cache.n += len(ids)
        return out


def bandwidth(threads=None, gigabytes=2.0, reps=5, pin=True):
    """This machine's best streaming read bandwidth in GB/s with `threads` threads, pinned as
    Fast pins them, each reading memory it first touched."""
    threads = threads or threads_default()
    order = cpu_order(threads) if pin else None
    cpus = (ctypes.c_int32 * threads)(*order[0]) if order else None
    return library().allie_bandwidth(threads, int(gigabytes * 2**30), reps, cpus)


class Graphs:
    """CUDA graphs of model.py's forward for steps that add one token to each of up to 32 games.
    Each game's cache is a slot of one preallocated pool (Cache gets views into it, so the
    PyTorch path runs on the same memory); a step of B games replays the graph captured for
    the smallest batch >= B and attention span > every game's length, with padding rows on a
    spare slot. Launch overhead, not memory, bounds a single game's step on a GPU: one replay
    replaces about a thousand kernel launches."""

    BATCHES = (1, 2, 4, 8, 16, 32)
    SPANS = (128, 256, 512, 1025)

    def __init__(self, model, slots=None):
        m, ctx = model, model.w["cos"].shape[0]
        per = 2 * m.layers * m.heads * ctx * m.head_dim + ctx * m.width  # elements per slot
        size = per * torch.tensor([], dtype=m.dtype).element_size()
        if slots is None:  # a third of the free memory, at most 32 games
            slots = min(32, int(torch.cuda.mem_get_info(m.device)[0] / 3 // size))
        if slots < 1:
            raise RuntimeError("no GPU memory for a cache pool")
        kw = dict(dtype=m.dtype, device=m.device)
        self.k = torch.zeros(m.layers, slots + 1, m.heads, ctx, m.head_dim, **kw)  # last: padding
        self.v = torch.zeros_like(self.k)
        self.e = torch.zeros(slots + 1, ctx, m.width, **kw)
        self.model, self.slots, self.ctx = m, slots, ctx
        self.free = list(range(slots))[::-1]
        self.graphs, self.pool = {}, None

    def attach(self, cache):
        """A new cache's storage: a free slot (False when none is left)."""
        if not self.free:
            return False
        j = self.free.pop()
        cache.k, cache.v, cache.e = self.k[:, j], self.v[:, j], self.e[j]
        cache.capacity, cache.slot = self.ctx, j
        weakref.finalize(cache, self.free.append, j)
        return True

    def capture(self, b, span):
        m, dev = self.model, self.model.device
        x = dict(ids=torch.zeros(b, dtype=torch.long, device=dev),
                 feats=torch.full((b, 3), -1.0, device=dev),
                 boards=torch.zeros(b, 68, dtype=torch.uint8, device=dev),
                 pos=torch.zeros(b, dtype=torch.long, device=dev),
                 slot=torch.full((b,), self.slots, dtype=torch.long, device=dev))  # fmt: skip
        keys = torch.arange(span, device=dev)

        def previous(e):
            p = self.e[x["slot"], (x["pos"] - 1).clamp(min=0)] * (x["pos"] > 0)[:, None]
            self.e[x["slot"], x["pos"]] = e
            return p

        def attend(i, q, k, v):
            self.k[i, x["slot"], :, x["pos"]] = k
            self.v[i, x["slot"], :, x["pos"]] = v
            mask = (keys <= x["pos"][:, None])[:, None, None]
            y = F.scaled_dot_product_attention(q[:, :, None], self.k[i, x["slot"], :, :span],
                                               self.v[i, x["slot"], :, :span], attn_mask=mask,
                                               scale=m.scale)  # fmt: skip
            return y[:, :, 0]

        def run():
            return m.forward(x["ids"], x["pos"], x["feats"], x["boards"], previous, attend)

        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(2):  # warm up the libraries outside the capture
                run()
        torch.cuda.current_stream(dev).wait_stream(side)
        self.pool = self.pool or torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = run()
        return g, x, out

    def step(self, items):
        """step()'s logits, or None when the step is not one new token per pooled game."""
        n = len(items)
        if n > self.BATCHES[-1] or any(len(ids) != 1 or getattr(c, "slot", None) is None
                                       for c, ids, *_ in items):  # fmt: skip
            return None
        b = next(x for x in self.BATCHES if x >= n)
        span = next(x for x in self.SPANS if x > max(c.n for c, *_ in items))
        if (b, span) not in self.graphs:
            self.graphs[b, span] = self.capture(b, span)
        g, x, out = self.graphs[b, span]
        for j, k in enumerate(("ids", "feats", "boards"), 1):
            x[k][:n].copy_(torch.cat([it[j] for it in items]))
        x["pos"][:n].copy_(torch.tensor([c.n for c, *_ in items]))
        x["slot"][:n].copy_(torch.tensor([c.slot for c, *_ in items]))
        if b > n:  # padding rows: a spare slot's first position
            x["pos"][n:], x["slot"][n:] = 0, self.slots
        g.replay()
        for c, *_ in items:
            c.n += 1
        return out[:n].clone()
