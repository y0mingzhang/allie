//! The C++ kernels' `vf` type and its operations (fast.py's SOURCE), three ways: AVX-512, AVX2 + FMA and
//! the scalar struct. Every kernel is generic over `Simd` and compiled once per variant; the Engine picks
//! one at construction. The variants reproduce the C++ builds bit for bit, including what GCC makes of the
//! source: `fma`/`fnmadd` where it contracts a multiply into an add (`exp`'s range reduction, `exp1`'s
//! final step, and the scalar `fmaf` sites), plain operations elsewhere.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::missing_safety_doc, clippy::excessive_precision)]

use std::arch::x86_64::*;

/// `f32(bf16)`: the BF16 bits widened.
#[inline(always)]
pub fn bf(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// `tobf`: round to nearest even, NaN kept quiet.
#[inline(always)]
pub fn tobf(f: f32) -> u16 {
    let u = f.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        return (u >> 16 | 0x40) as u16;
    }
    (u.wrapping_add(0x7fff + (u >> 16 & 1)) >> 16) as u16
}

/// `rb`: round to BF16.
#[inline(always)]
pub fn rb(f: f32) -> f32 {
    bf(tobf(f))
}

/// `sigm`.
#[inline(always)]
pub fn sigm(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub trait Simd: Copy + Send + Sync + 'static {
    const VL: usize;
    const MB: usize;
    const RB: usize;
    const NAME: &'static str;
    type V: Copy;
    unsafe fn zero() -> Self::V;
    unsafe fn set(a: f32) -> Self::V;
    unsafe fn ld(p: *const f32) -> Self::V;
    unsafe fn ld_i8(p: *const i8) -> Self::V;
    unsafe fn ld_bf16(p: *const u16) -> Self::V;
    unsafe fn st(p: *mut f32, a: Self::V);
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn div(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn max(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn min(a: Self::V, b: Self::V) -> Self::V;
    /// a * b + c
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    /// c - a * b, one rounding (what GCC makes of `vsub(c, vmul(a, b))`)
    unsafe fn fnmadd(a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    unsafe fn sum(a: Self::V) -> f32;
    unsafe fn floor(a: Self::V) -> Self::V;
    unsafe fn pow2(n: Self::V) -> Self::V;
    unsafe fn rb(a: Self::V) -> Self::V;
    unsafe fn exp(a: Self::V) -> Self::V;
    /// exp(a) + 1 as GCC fuses it (gelu, silu_mul): fma(y, 2^n, 1)
    unsafe fn exp1(a: Self::V) -> Self::V;
    /// a * b + c in scalar code: fused where GCC contracts it (the FMA builds)
    fn fmaf(a: f32, b: f32, c: f32) -> f32;
}

/// Cephes expf (within 2 ulp), as GCC compiles the C++: the two range-reduction steps are fnmadds.
#[inline(always)]
unsafe fn cephes<S: Simd>(x: S::V, plus1: bool) -> S::V {
    let x = S::min(S::max(x, S::set(-87.3)), S::set(88.0));
    let n = S::floor(S::fma(x, S::set(1.442_695_040_888_963_4), S::set(0.5)));
    let x = S::fnmadd(n, S::set(0.693_359_375), x);
    let x = S::fnmadd(n, S::set(-2.121_944_4e-4), x);
    let mut y = S::set(1.987_569_15e-4);
    y = S::fma(y, x, S::set(1.398_199_950_7e-3));
    y = S::fma(y, x, S::set(8.333_451_907_3e-3));
    y = S::fma(y, x, S::set(4.166_579_589_4e-2));
    y = S::fma(y, x, S::set(1.666_666_545_9e-1));
    y = S::fma(y, x, S::set(5.000_000_120_1e-1));
    y = S::fma(y, S::mul(x, x), S::add(x, S::set(1.0)));
    let p = S::pow2(n);
    if plus1 {
        S::fma(y, p, S::set(1.0))
    } else {
        S::mul(y, p)
    }
}

#[derive(Clone, Copy)]
pub struct Avx512;

impl Simd for Avx512 {
    const VL: usize = 16;
    const MB: usize = 6;
    const RB: usize = 4;
    const NAME: &'static str = "avx512";
    type V = __m512;
    #[inline(always)]
    unsafe fn zero() -> __m512 {
        _mm512_setzero_ps()
    }
    #[inline(always)]
    unsafe fn set(a: f32) -> __m512 {
        _mm512_set1_ps(a)
    }
    #[inline(always)]
    unsafe fn ld(p: *const f32) -> __m512 {
        _mm512_loadu_ps(p)
    }
    #[inline(always)]
    unsafe fn ld_i8(p: *const i8) -> __m512 {
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p as *const __m128i)))
    }
    #[inline(always)]
    unsafe fn ld_bf16(p: *const u16) -> __m512 {
        _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(_mm256_loadu_si256(p as *const __m256i))))
    }
    #[inline(always)]
    unsafe fn st(p: *mut f32, a: __m512) {
        _mm512_storeu_ps(p, a)
    }
    #[inline(always)]
    unsafe fn add(a: __m512, b: __m512) -> __m512 {
        _mm512_add_ps(a, b)
    }
    #[inline(always)]
    unsafe fn sub(a: __m512, b: __m512) -> __m512 {
        _mm512_sub_ps(a, b)
    }
    #[inline(always)]
    unsafe fn mul(a: __m512, b: __m512) -> __m512 {
        _mm512_mul_ps(a, b)
    }
    #[inline(always)]
    unsafe fn div(a: __m512, b: __m512) -> __m512 {
        _mm512_div_ps(a, b)
    }
    #[inline(always)]
    unsafe fn max(a: __m512, b: __m512) -> __m512 {
        _mm512_max_ps(a, b)
    }
    #[inline(always)]
    unsafe fn min(a: __m512, b: __m512) -> __m512 {
        _mm512_min_ps(a, b)
    }
    #[inline(always)]
    unsafe fn fma(a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmadd_ps(a, b, c)
    }
    #[inline(always)]
    unsafe fn fnmadd(a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fnmadd_ps(a, b, c)
    }
    /// `_mm512_reduce_add_ps` (GCC and clang alike): halves added, quarters added, then (s0 + s2) + (s1 + s3)
    #[inline(always)]
    unsafe fn sum(a: __m512) -> f32 {
        let d = _mm512_castps_pd(a);
        let s = _mm256_add_ps(_mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(d)), _mm256_castpd_ps(_mm512_extractf64x4_pd::<0>(d)));
        let s = _mm_add_ps(_mm256_extractf128_ps::<1>(s), _mm256_castps256_ps128(s));
        let t = _mm_add_ps(s, _mm_shuffle_ps::<0b01_00_11_10>(s, s));
        _mm_cvtss_f32(t) + _mm_cvtss_f32(_mm_movehdup_ps(t))
    }
    #[inline(always)]
    unsafe fn floor(a: __m512) -> __m512 {
        _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(a)
    }
    #[inline(always)]
    unsafe fn pow2(n: __m512) -> __m512 {
        _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127))))
    }
    #[inline(always)]
    unsafe fn rb(x: __m512) -> __m512 {
        let u = _mm512_castps_si512(x);
        let r = _mm512_add_epi32(_mm512_set1_epi32(0x7fff), _mm512_and_si512(_mm512_srli_epi32::<16>(u), _mm512_set1_epi32(1)));
        _mm512_castsi512_ps(_mm512_and_si512(_mm512_add_epi32(u, r), _mm512_set1_epi32(0xffff_0000_u32 as i32)))
    }
    #[inline(always)]
    unsafe fn exp(a: __m512) -> __m512 {
        cephes::<Self>(a, false)
    }
    #[inline(always)]
    unsafe fn exp1(a: __m512) -> __m512 {
        cephes::<Self>(a, true)
    }
    #[inline(always)]
    fn fmaf(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
}

#[derive(Clone, Copy)]
pub struct Avx2;

impl Simd for Avx2 {
    const VL: usize = 8;
    const MB: usize = 6;
    const RB: usize = 2;
    const NAME: &'static str = "avx2";
    type V = __m256;
    #[inline(always)]
    unsafe fn zero() -> __m256 {
        _mm256_setzero_ps()
    }
    #[inline(always)]
    unsafe fn set(a: f32) -> __m256 {
        _mm256_set1_ps(a)
    }
    #[inline(always)]
    unsafe fn ld(p: *const f32) -> __m256 {
        _mm256_loadu_ps(p)
    }
    #[inline(always)]
    unsafe fn ld_i8(p: *const i8) -> __m256 {
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p as *const __m128i)))
    }
    #[inline(always)]
    unsafe fn ld_bf16(p: *const u16) -> __m256 {
        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(_mm_loadu_si128(p as *const __m128i))))
    }
    #[inline(always)]
    unsafe fn st(p: *mut f32, a: __m256) {
        _mm256_storeu_ps(p, a)
    }
    #[inline(always)]
    unsafe fn add(a: __m256, b: __m256) -> __m256 {
        _mm256_add_ps(a, b)
    }
    #[inline(always)]
    unsafe fn sub(a: __m256, b: __m256) -> __m256 {
        _mm256_sub_ps(a, b)
    }
    #[inline(always)]
    unsafe fn mul(a: __m256, b: __m256) -> __m256 {
        _mm256_mul_ps(a, b)
    }
    #[inline(always)]
    unsafe fn div(a: __m256, b: __m256) -> __m256 {
        _mm256_div_ps(a, b)
    }
    #[inline(always)]
    unsafe fn max(a: __m256, b: __m256) -> __m256 {
        _mm256_max_ps(a, b)
    }
    #[inline(always)]
    unsafe fn min(a: __m256, b: __m256) -> __m256 {
        _mm256_min_ps(a, b)
    }
    #[inline(always)]
    unsafe fn fma(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fmadd_ps(a, b, c)
    }
    #[inline(always)]
    unsafe fn fnmadd(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fnmadd_ps(a, b, c)
    }
    #[inline(always)]
    unsafe fn sum(a: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
    }
    #[inline(always)]
    unsafe fn floor(a: __m256) -> __m256 {
        _mm256_floor_ps(a)
    }
    #[inline(always)]
    unsafe fn pow2(n: __m256) -> __m256 {
        _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127))))
    }
    #[inline(always)]
    unsafe fn rb(x: __m256) -> __m256 {
        let u = _mm256_castps_si256(x);
        let r = _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(_mm256_srli_epi32::<16>(u), _mm256_set1_epi32(1)));
        _mm256_castsi256_ps(_mm256_and_si256(_mm256_add_epi32(u, r), _mm256_set1_epi32(0xffff_0000_u32 as i32)))
    }
    #[inline(always)]
    unsafe fn exp(a: __m256) -> __m256 {
        cephes::<Self>(a, false)
    }
    #[inline(always)]
    unsafe fn exp1(a: __m256) -> __m256 {
        cephes::<Self>(a, true)
    }
    #[inline(always)]
    fn fmaf(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
}

/// The C++ `struct vf { float v[8]; }` path of a build without AVX: lane-wise operations, `expf` and `rb`
/// per lane, no fused multiply-adds anywhere.
#[derive(Clone, Copy)]
pub struct Scalar;

#[inline(always)]
fn lanes(f: impl Fn(usize) -> f32) -> [f32; 8] {
    let mut r = [0.0; 8];
    for (i, x) in r.iter_mut().enumerate() {
        *x = f(i);
    }
    r
}

impl Simd for Scalar {
    const VL: usize = 8;
    const MB: usize = 6;
    const RB: usize = 2;
    const NAME: &'static str = "scalar";
    type V = [f32; 8];
    #[inline(always)]
    unsafe fn zero() -> [f32; 8] {
        [0.0; 8]
    }
    #[inline(always)]
    unsafe fn set(a: f32) -> [f32; 8] {
        [a; 8]
    }
    #[inline(always)]
    unsafe fn ld(p: *const f32) -> [f32; 8] {
        lanes(|i| *p.add(i))
    }
    #[inline(always)]
    unsafe fn ld_i8(p: *const i8) -> [f32; 8] {
        lanes(|i| *p.add(i) as f32)
    }
    #[inline(always)]
    unsafe fn ld_bf16(p: *const u16) -> [f32; 8] {
        lanes(|i| bf(*p.add(i)))
    }
    #[inline(always)]
    unsafe fn st(p: *mut f32, a: [f32; 8]) {
        for (i, x) in a.iter().enumerate() {
            *p.add(i) = *x;
        }
    }
    #[inline(always)]
    unsafe fn add(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        lanes(|i| a[i] + b[i])
    }
    #[inline(always)]
    unsafe fn sub(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        lanes(|i| a[i] - b[i])
    }
    #[inline(always)]
    unsafe fn mul(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        lanes(|i| a[i] * b[i])
    }
    #[inline(always)]
    unsafe fn div(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        lanes(|i| a[i] / b[i])
    }
    #[inline(always)]
    unsafe fn max(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        lanes(|i| if a[i] < b[i] { b[i] } else { a[i] }) // std::max
    }
    #[inline(always)]
    unsafe fn min(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        lanes(|i| if b[i] < a[i] { b[i] } else { a[i] }) // std::min
    }
    #[inline(always)]
    unsafe fn fma(a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> [f32; 8] {
        lanes(|i| a[i] * b[i] + c[i])
    }
    #[inline(always)]
    unsafe fn fnmadd(a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> [f32; 8] {
        lanes(|i| c[i] - a[i] * b[i])
    }
    #[inline(always)]
    unsafe fn sum(a: [f32; 8]) -> f32 {
        a.iter().fold(0.0, |s, x| s + x)
    }
    #[inline(always)]
    unsafe fn floor(a: [f32; 8]) -> [f32; 8] {
        lanes(|i| a[i].floor())
    }
    #[inline(always)]
    unsafe fn pow2(n: [f32; 8]) -> [f32; 8] {
        lanes(|i| f32::from_bits(((n[i] as i32 + 127) as u32) << 23))
    }
    #[inline(always)]
    unsafe fn rb(a: [f32; 8]) -> [f32; 8] {
        lanes(|i| rb(a[i]))
    }
    #[inline(always)]
    unsafe fn exp(a: [f32; 8]) -> [f32; 8] {
        lanes(|i| a[i].exp())
    }
    #[inline(always)]
    unsafe fn exp1(a: [f32; 8]) -> [f32; 8] {
        lanes(|i| 1.0 + a[i].exp())
    }
    #[inline(always)]
    fn fmaf(a: f32, b: f32, c: f32) -> f32 {
        a * b + c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run<S: Simd>(f: impl Fn(f32) -> f32, g: impl Fn(S::V) -> S::V, xs: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut want = Vec::new();
        let mut got = Vec::new();
        for c in xs.chunks(S::VL) {
            let mut buf = vec![0.0f32; S::VL];
            buf[..c.len()].copy_from_slice(c);
            let mut out = vec![0.0f32; S::VL];
            unsafe { S::st(out.as_mut_ptr(), g(S::ld(buf.as_ptr()))) };
            want.extend(buf.iter().map(|&x| f(x)));
            got.extend(out);
        }
        (want, got)
    }

    fn inputs() -> Vec<f32> {
        let mut v: Vec<f32> = (-2000..2000).map(|i| i as f32 * 0.0437).collect();
        v.extend([0.0, -0.0, 1e-20, -1e-20, 87.0, -87.0, 100.0, -100.0, f32::INFINITY, -f32::INFINITY, 1.0 + f32::EPSILON]);
        v
    }

    /// GCC's Cephes expf in scalar f32 arithmetic with the same contractions: a transcription of the constants
    fn cephes_ref(x: f32) -> f32 {
        let x = x.max(-87.3).min(88.0);
        let n = x.mul_add(1.442_695_040_888_963_4, 0.5).floor();
        let x = (-n).mul_add(0.693_359_375, x);
        let x = (-n).mul_add(-2.121_944_4e-4, x);
        let mut y = 1.987_569_15e-4_f32;
        for c in [1.398_199_950_7e-3, 8.333_451_907_3e-3, 4.166_579_589_4e-2, 1.666_666_545_9e-1, 5.000_000_120_1e-1] {
            y = y.mul_add(x, c);
        }
        y = y.mul_add(x * x, x + 1.0);
        y * f32::from_bits(((n as i32 + 127) as u32) << 23)
    }

    fn ulps(a: f32, b: f32) -> i64 {
        (a.to_bits() as i64 - b.to_bits() as i64).abs()
    }

    fn check_exp<S: Simd>() {
        let xs: Vec<f32> = inputs().into_iter().filter(|x| x.abs() <= 87.0).collect();
        let (_, got) = run::<S>(|x| x, |v| unsafe { S::exp(v) }, &xs);
        for (x, g) in xs.iter().zip(&got) {
            if S::NAME == "scalar" {
                assert_eq!(g.to_bits(), x.exp().to_bits(), "exp({x})");
            } else {
                assert_eq!(g.to_bits(), cephes_ref(*x).to_bits(), "exp({x}) {g} vs {}", cephes_ref(*x));
            }
            assert!(ulps(*g, x.exp()) <= 2, "exp({x}): {g} vs expf {} ({} ulp)", x.exp(), ulps(*g, x.exp()));
        }
        let (_, got1) = run::<S>(|x| x, |v| unsafe { S::exp1(v) }, &xs);
        for (x, g) in xs.iter().zip(&got1) {
            assert!(ulps(*g, x.exp() + 1.0) <= 2, "exp1({x})");
        }
    }

    fn check_rb<S: Simd>() {
        let xs = inputs();
        let (want, got) = run::<S>(rb, |v| unsafe { S::rb(v) }, &xs);
        for ((x, w), g) in xs.iter().zip(&want).zip(&got) {
            assert_eq!(w.to_bits(), g.to_bits(), "rb({x})");
        }
    }

    fn check_sum<S: Simd>() {
        // lanes chosen so every reduction order gives another sum
        let xs: Vec<f32> = (0..S::VL).map(|i| 1.0 + (i as f32 + 1.0) * f32::EPSILON * if i % 2 == 0 { 1.0 } else { 3.0 }).collect();
        let got = unsafe { S::sum(S::ld(xs.as_ptr())) };
        let want = match S::VL {
            16 => {
                let h: Vec<f32> = (0..8).map(|i| xs[i + 8] + xs[i]).collect();
                let q: Vec<f32> = (0..4).map(|i| h[i + 4] + h[i]).collect();
                (q[0] + q[2]) + (q[1] + q[3])
            }
            8 if S::NAME == "avx2" => {
                let q: Vec<f32> = (0..4).map(|i| xs[i] + xs[i + 4]).collect();
                (q[0] + q[2]) + (q[1] + q[3])
            }
            _ => xs.iter().fold(0.0, |s, x| s + x),
        };
        assert_eq!(got.to_bits(), want.to_bits(), "{}: {got} vs {want}", S::NAME);
    }

    fn check_loads<S: Simd>() {
        let i8s: Vec<i8> = (0..16).map(|i| (i * 17 - 100) as i8).collect();
        let bfs: Vec<u16> = (0..16).map(|i| tobf(i as f32 * -1.5)).collect();
        let mut out = vec![0.0f32; 16];
        unsafe { S::st(out.as_mut_ptr(), S::ld_i8(i8s.as_ptr())) };
        assert!(out[..S::VL].iter().zip(&i8s).all(|(o, i)| *o == *i as f32));
        unsafe { S::st(out.as_mut_ptr(), S::ld_bf16(bfs.as_ptr())) };
        assert!(out[..S::VL].iter().zip(&bfs).all(|(o, b)| *o == bf(*b)));
    }

    fn all<S: Simd>() {
        check_exp::<S>();
        check_rb::<S>();
        check_sum::<S>();
        check_loads::<S>();
    }

    #[test]
    fn scalar() {
        all::<Scalar>();
        let xs = inputs();
        let (want, got) = run::<Scalar>(|x| x.exp(), |v| unsafe { Scalar::exp(v) }, &xs);
        assert!(want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn avx2() {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            all::<Avx2>();
        }
    }

    #[test]
    fn avx512() {
        if is_x86_feature_detected!("avx512f") {
            all::<Avx512>();
        }
    }

    #[test]
    fn bf16_rounding() {
        assert_eq!(tobf(1.0), 0x3f80);
        assert_eq!(rb(1.0 + 2.0f32.powi(-8)), 1.0); // ties to even
        assert_eq!(rb(1.0 + 3.0 * 2.0f32.powi(-8)), 1.0 + 2.0f32.powi(-6));
        assert!(rb(f32::NAN).is_nan());
        assert_eq!(rb(f32::INFINITY), f32::INFINITY);
    }
}
