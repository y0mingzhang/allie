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
from pathlib import Path

import torch

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
#define MB 4
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
#define MB 2
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
#define MB 2
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
  std::mutex mu;
  std::condition_variable cv;
  bool stop = false;
  void (*fn)(void*, int) = nullptr;
  void* arg = nullptr;

  Pool(int n_, int pin_, double spin_) : n(n_), pin(pin_ != 0), spin(spin_) {
#if defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0)
      for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, &set)) cpus.push_back(c);
#endif
    if ((int)cpus.size() < n) pin = false;
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
          std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&] { return epoch.load(std::memory_order_acquire) != seen; });
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

// acc[m * 4 + i] = x[m] . r[i] over K
template <class W, int M>
static inline void block(const float* const* x, const W* const* r, int K, float* acc) {
  vf a[M][4];
  for (int m = 0; m < M; m++)
    for (int i = 0; i < 4; i++) a[m][i] = vzero();
  int k = 0, kv = K - K % VL;
  if (M == 1) {  // two partial sums per row: eight independent chains
    vf b[4] = {vzero(), vzero(), vzero(), vzero()};
    for (; k + 2 * VL <= kv; k += 2 * VL) {
      vf x0 = vld(x[0] + k), x1 = vld(x[0] + k + VL);
      for (int i = 0; i < 4; i++) {
        a[0][i] = vfma(vld(r[i] + k), x0, a[0][i]);
        b[i] = vfma(vld(r[i] + k + VL), x1, b[i]);
      }
    }
    for (int i = 0; i < 4; i++) a[0][i] = vadd(a[0][i], b[i]);
  }
  for (; k < kv; k += VL) {
    vf w0 = vld(r[0] + k), w1 = vld(r[1] + k), w2 = vld(r[2] + k), w3 = vld(r[3] + k);
    for (int m = 0; m < M; m++) {
      vf xm = vld(x[m] + k);
      a[m][0] = vfma(w0, xm, a[m][0]);
      a[m][1] = vfma(w1, xm, a[m][1]);
      a[m][2] = vfma(w2, xm, a[m][2]);
      a[m][3] = vfma(w3, xm, a[m][3]);
    }
  }
  for (int m = 0; m < M; m++)
    for (int i = 0; i < 4; i++) {
      float s = vsum(a[m][i]);
      for (int j = kv; j < K; j++) s += x[m][j] * f32(r[i][j]);
      acc[m * 4 + i] = s;
    }
}

// out[m * ldo + j] = x[m] . w[j] for rows j < n of w (row stride K), tokens m < M
template <class W>
static void dots(const float* const* x, int M, const W* w, int K, int n, float* out, int ldo) {
  float acc[MB * 4];
  int tile = M <= 64 ? M : 32;
  for (int m0 = 0; m0 < M; m0 += tile) {
    int m1 = std::min(M, m0 + tile);
    for (int j = 0; j < n; j += 4) {
      const W* r[4];
      for (int i = 0; i < 4; i++) r[i] = w + (size_t)std::min(j + i, n - 1) * K;
      int nr = std::min(4, n - j);
      for (int m = m0; m < m1; m += MB) {
        int mb = std::min(MB, m1 - m);
        switch (mb) {
          case 1: block<W, 1>(x + m, r, K, acc); break;
          case 2: block<W, 2>(x + m, r, K, acc); break;
#if MB > 2
          case 3: block<W, 3>(x + m, r, K, acc); break;
          case 4: block<W, 4>(x + m, r, K, acc); break;
#endif
        }
        for (int a = 0; a < mb; a++)
          for (int i = 0; i < nr; i++) out[(size_t)(m + a) * ldo + j + i] = acc[a * 4 + i];
      }
    }
  }
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
  double cost0, cost;
};

struct Engine {
  int L, D, H, hd, V, nve, E, topk, keep, eh, sh, dh, ctx, backout_layer;
  float scale, floor_;
  int q8;
  const bf16 *embed, *embed2, *lm_head, *feat, *smear_gate, *cosv, *sinv, *bfirst, *bres[2], *bsq, *bmeta, *bout,
      *skip_gate[3];
  const float *scal, *x0l;
  std::vector<float> w1, wr, wsq;  // board CNN weights, FP32, [piece][tap][co], [conv][tap][ci][co], [o][ci]
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
  double segcost;
  std::vector<float> sgate;
  std::vector<const float*> ph, py, phf, pxf, ptok, phid, pshid;
  std::vector<Seg> segs;
  std::vector<std::vector<float>> scratch;
  int prof = 0;
  double ptime[NPHASE] = {0}, plast = 0;

  void sync(int t, int phase) {  // barrier; thread 0 charges the time since the last one to phase
    pool->barrier();
    if (prof && t == 0) {
      double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
      ptime[phase] += now - plast;
      plast = now;
    }
  }
  void embedding(int t, int nt);
  void blockstep(int i, int t, int nt);
  void ffn(int t, int nt, int dense, int i);
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

static void boardcnn(const Engine& m, const uint8_t* st, float* v) {  // channels last, zero border
  enum { C = 32 / VL, NP = 8 / C };  // NP squares at once: eight independent sums
  alignas(64) float a[100 * 32] = {0}, b[64 * 32];
  int piece[100];
  for (int q = 0; q < 100; q++) piece[q] = 13;  // off the board: a zero weight row
  for (int p = 0; p < 64; p++) piece[(p / 8 + 1) * 10 + p % 8 + 1] = st[p];
  for (int pass = 0; pass < 3; pass++) {
    for (int p = 0; p < 64; p += NP) {
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
        for (int c = 0; c < C; c++) vst(b + (p + j) * 32 + c * VL, vrb(acc[j][c]));
    }
    gelu(b, 64 * 32);
    for (int p = 0; p < 64; p++) {
      float* ap = a + ((p / 8 + 1) * 10 + p % 8 + 1) * 32;
      for (int c = 0; c < 32; c++) ap[c] = pass ? rb(ap[c] + b[p * 32 + c]) : b[p * 32 + c];
    }
  }
  for (int o = 0; o < 8; o++)
    for (int p = 0; p < 64; p++) {
      const float* ap = a + ((p / 8 + 1) * 10 + p % 8 + 1) * 32;
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
  for (int k = t; k < T; k += nt) {
    clockfeat(feats + 3 * k, &f64[k * 64]);
    boardcnn(*this, boards + 68 * k, &b544[k * 544]);
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

void Engine::blockstep(int i, int t, int nt) {
  const Layer& ly = layers[i];
  int lo, hi;
  // residual stream: skip connection in, x0 blend; h = norm(x); output and value-embedding gates
  for (int k = t; k < T; k += nt) {
    float* xk = &x[(size_t)k * D];
    const float *a = &x0[(size_t)k * D], *b = &x02[(size_t)k * D];
    if (ly.skout >= 0) {
      int j = ly.skout;
      float gs = sigm(scal[3 * L + 2 + j]) * 2;
      float gg = rb(gs * rb(sigm(rb(dot(a, skip_gate[j], 16)))));
      axpy_rb(xk, xk, gg, &skip[2 - j][(size_t)k * D], D);
    }
    float c0 = x0l[2 * i], c1 = x0l[2 * i + 1], lam = scal[i];
    int d = 0;
    if (i == 0) {
      float c = (float)((double)lam + (double)c0);
      for (; d + VL <= D; d += VL)
        vst(xk + d, vrb(vadd(vrb(vmul(vset(c), vld(xk + d))), vrb(vmul(vset(c1), vld(b + d))))));
      for (; d < D; d++) xk[d] = rb(rb(c * xk[d]) + rb(c1 * b[d]));
    } else {
      for (; d + VL <= D; d += VL) {
        vf s = vrb(vadd(vrb(vmul(vset(c0), vld(a + d))), vrb(vmul(vset(c1), vld(b + d)))));
        vst(xk + d, vrb(vfma(vset(lam), vld(xk + d), s)));
      }
      for (; d < D; d++) xk[d] = rb(rb(rb(c0 * a[d]) + rb(c1 * b[d])) + lam * xk[d]);
    }
    float* hk = &h[(size_t)k * D];
    norm(xk, hk, D);
    for (int r = 0; r < ly.G; r++) g[(size_t)k * ly.G + r] = rb(sigm(rb(dot(hk, ly.gates + r * 16, 16))));
  }
  sync(t, B_NORM);
  split(3 * D, t, nt, 4, lo, hi);
  mm(ly.qkv, D, lo, hi - lo, ph.data(), T, &qkv[lo], 3 * D);
  sync(t, B_QKV);
  // q, k: per-head norm and rotary; v (+ value embedding); k, v into the cache
  int half = hd / 2;
  for (int u = t; u < T * H; u += nt) {
    int k = u / H, hh = u % H;
    const Seq& s = seq[tseq[k]];
    int64_t p = s.n0 + (k - s.off);
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
      float g2 = rb(2 * g[(size_t)k * ly.G + H + hh]);
      const bf16* vr = ve[ly.ve] + (size_t)ids[k] * D + hh * hd;
      for (int c = 0; c < hd; c++) vc[c] = tobf(rb(v[c] + rb(g2 * f32(vr[c]))));
    } else {
      for (int c = 0; c < hd; c++) vc[c] = tobf(v[c]);
    }
  }
  sync(t, B_ROTARY);
  // attention: (sequence, head, query block) units
  float* sc = scratch[t].data();
  int nu = (int)units.size() / 3;
  for (int u = t; u < nu * H; u += nt) {
    int w = u / H, hh = u % H;
    const Seq& s = seq[units[3 * w]];
    const bf16* K = s.k + ((size_t)i * H + hh) * s.cap * hd;
    const bf16* Vv = s.v + ((size_t)i * H + hh) * s.cap * hd;
    for (int k = units[3 * w + 1]; k < units[3 * w + 2]; k++) {
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
      float og = g[(size_t)k * ly.G + hh], ov[256];
      for (int c = 0; c < hd / VL; c++) vst(ov + c * VL, o[c]);
      for (int c = 0; c < hd; c++) yo[c] = rb(rb(ov[c] / sum) * og);
    }
  }
  sync(t, B_ATTN);
  split(D, t, nt, 4, lo, hi);
  mm(ly.o, D, lo, hi - lo, py.data(), T, &tmp[lo], D);
  for (int k = 0; k < T; k++) add_rb(&x[(size_t)k * D + lo], &tmp[(size_t)k * D + lo], hi - lo);
  sync(t, B_O);
  for (int k = t; k < T; k += nt) {
    norm(&x[(size_t)k * D], &h[(size_t)k * D], D);
    if (!ly.fc.w)
      for (int d = 0; d < D; d++) hf[(size_t)k * D + d] = h[(size_t)k * D + d] - ly.mu[d];
  }
  sync(t, B_NORM2);
  ffn(t, nt, ly.fc.w != nullptr, i);
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

void Engine::ffn(int t, int nt, int dense, int i) {
  const Layer& ly = layers[i];
  int lo, hi;
  if (dense) {
    if (t == 0) segs.assign(1, Seg{ly.fc, dh, T, std::max(sh, dh), ph.data(), shid.data(), 0, 0});
  } else {
    split(E, t, nt, 4, lo, hi);
    Mat R{ly.router, nullptr, 2};
    mm(R, D, lo, hi - lo, phf.data(), T, &rs[lo], E);
    for (int k = 0; k < T; k++)
      for (int e = lo; e < hi; e++) rs[(size_t)k * E + e] = sigm(rs[(size_t)k * E + e]);
    sync(t, B_ROUTER);
    for (int k = t; k < T; k += nt) route(*this, ly, k);
    sync(t, B_TOPK);
    if (t == 0) {  // tokens grouped by expert: each expert's weights read once
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
      segs.clear();
      size_t es = (size_t)2 * eh * D * (q8 ? 1 : 2);
      for (int e = 0; e < E; e++)
        if (cnt[e]) {
          Mat A{(const char*)ly.up + e * es, q8 ? ly.ups + (size_t)e * 2 * eh : nullptr, q8};
          segs.push_back(Seg{A, eh, cnt[e], eh, &ptok[start[e]], &hid[(size_t)start[e] * eh], 0, 0});
        }
      segs.push_back(Seg{ly.sup, sh, T, std::max(sh, dh), ph.data(), shid.data(), 0, 0});
    }
  }
  if (t == 0) {
    double total = 0;
    for (auto& s : segs) s.cost0 = total, s.cost = (double)s.pairs * (2 + s.ntok), total += s.cost;
    segcost = total;
  }
  sync(t, B_GROUP);
  // up (gate and value halves), SwiGLU: each thread an equal share of the cost
  double c0 = segcost * t / nt, c1 = segcost * (t + 1) / nt;
  float* ta = scratch[t].data();
  for (const auto& s : segs) {
    if (s.cost0 + s.cost <= c0 || s.cost0 >= c1) continue;
    double per = s.cost / s.pairs;
    auto at = [&](double c) { return std::min(s.pairs, std::max(0, ((int)((c - s.cost0) / per + 0.5) + 3) / 4 * 4)); };
    int p0 = s.cost0 >= c0 ? 0 : at(c0), p1 = s.cost0 + s.cost <= c1 ? s.pairs : at(c1);
    for (int p = p0; p < p1; p += 64) {
      int n = std::min(64, p1 - p);
      float *a = ta, *b = ta + (size_t)s.ntok * n;
      mm(s.A, D, p, n, s.x, s.ntok, a, n);
      mm(s.A, D, s.pairs + p, n, s.x, s.ntok, b, n);
      for (int m = 0; m < s.ntok; m++) silu_mul(a + (size_t)m * n, b + (size_t)m * n, s.out + (size_t)m * s.ldo + p, n);
    }
  }
  sync(t, B_UP);
  // down: each thread a slice of the output features, summed over its tokens' experts
  split(D, t, nt, 4, lo, hi);
  int n = hi - lo;
  float* tb = scratch[t].data();
  if (n > 0) {
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
  }
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
  int lo, hi;
  split(V, t, nt, 4, lo, hi);
  Mat A{lm_head, nullptr, 0};
  mm(A, D, lo, hi - lo, pxf.data(), S, &z[lo], V);
  for (int s = 0; s < S; s++)
    for (int v = lo; v < hi; v++) out[(size_t)s * V + v] = 23.f * sigm((z[(size_t)s * V + v] + 5.f) / 7.5f);
  sync(t, H_HEAD);
}

extern "C" {

// cfg: L D H hd V nve E topk keep eh sh dh int8 ctx backout_layer threads pin, then per layer
// G ve skip_in skip_out. glob: embed embed2 lm_head feat_embed smear_gate scalars x0_lambdas cos sin
// board.first board.residual.0 board.residual.1 board.squeeze board.meta board.output
// skip_gate.0-2 value_embed.*. per layer (20): qkv qkv_s o o_s gates fc fc_s proj proj_s router
// moe_bias mu up up_s down down_s shared_up shared_up_s shared_down shared_down_s
void* allie_new(const int64_t* cfg, const double* fl, void* const* glob, void* const* lay, double spin) {
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
  m->pool = new Pool(cfg[15], cfg[16], spin);
  return m;
}

void allie_free(void* h) {
  Engine* m = (Engine*)h;
  delete m->pool;
  delete m;
}

int allie_threads(void* h) { return ((Engine*)h)->pool->n; }

// meta: per sequence n0 len cap off; caches: per sequence k v e; out: S x V
void allie_step(void* h, int T, int S, const int64_t* ids, const float* feats, const uint8_t* boards,
                const int64_t* meta, void* const* caches, float* out) {
  Engine& m = *(Engine*)h;
  int D = m.D, nt = m.pool->n;
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
  if (m.prof) m.plast = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  m.pool->run(trampoline, &m);
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
double allie_bandwidth(int threads, int64_t bytes, int reps, int pin) {
  Pool pool(threads, pin, 0.01);
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
    lib.allie_new.restype, lib.allie_new.argtypes = P, [P, P, P, P, D]
    lib.allie_free.argtypes = [P]
    lib.allie_threads.restype, lib.allie_threads.argtypes = I, [P]
    lib.allie_step.argtypes = [P, I, I, P, P, P, P, P, P]
    lib.allie_bandwidth.restype = D
    lib.allie_bandwidth.argtypes = [I, ctypes.c_int64, I, I]
    lib.allie_isa.restype = ctypes.c_char_p
    lib.allie_profile.restype, lib.allie_profile.argtypes = I, [P, P, I]
    return lib


def threads_default():
    return max(1, min(torch.get_num_threads(), len(os.sched_getaffinity(0))))


PHASES = ("board cnn", "embedding rows", "smear", "norm", "qkv", "rotary", "attention", "o",
          "norm2", "router", "top-k", "group", "up", "down", "head norm", "head")  # fmt: skip
LAYER = ("qkv", "o", "gates", "fc", "proj", "router", "moe_bias", "mu", "up", "down", "shared_up",
         "shared_down")  # fmt: skip
SCALED = ("qkv", "o", "fc", "proj", "up", "down", "shared_up", "shared_down")


class Fast:
    """model.py's step() for a CPU Model with BF16 activations (int8 or BF16 matrices)."""

    def __init__(self, model, threads=None, pin=True, spin=0.002):
        assert model.device.type == "cpu" and model.dtype == torch.bfloat16
        c, w, n = model.config, model.w, model.layers
        assert model.head_dim % 16 == 0 and model.head_dim <= 256 and model.topk <= 64
        assert (
            w["board.first"].shape == (32, 13, 3, 3)
            and w["board.output"].shape[0] == 544
        )
        self.lib, self.model = library(), model
        self.threads = threads or threads_default()
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
        self.handle = self.lib.allie_new(
            arr(ctypes.c_int64, cfg), arr(ctypes.c_double, [model.scale, model.floor]),
            arr(ctypes.c_void_p, ptrs), arr(ctypes.c_void_p, lay), spin,
        )  # fmt: skip

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.allie_free(self.handle)
            self.handle = None

    def profile(self, on=True):
        """Seconds spent in each phase of step() since the last call; on: keep counting."""
        out = (ctypes.c_double * len(PHASES))()
        assert self.lib.allie_profile(self.handle, out, int(on)) == len(PHASES)
        return dict(zip(PHASES, out))

    def step(self, items):
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
        self.lib.allie_step(
            self.handle, lo, len(items), ids.data_ptr(), feats.data_ptr(), boards.data_ptr(),
            (ctypes.c_int64 * len(meta))(*meta), (ctypes.c_void_p * len(caches))(*caches),
            out.data_ptr(),
        )  # fmt: skip
        for cache, ids, *_ in items:
            cache.n += len(ids)
        return out


def bandwidth(threads=None, gigabytes=2.0, reps=5, pin=True):
    """This machine's best streaming read bandwidth in GB/s with `threads` pinned threads."""
    threads = threads or threads_default()
    return library().allie_bandwidth(threads, int(gigabytes * 2**30), reps, int(pin))
