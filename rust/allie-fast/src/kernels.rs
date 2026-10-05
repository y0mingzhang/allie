//! The matrix and vector kernels of fast.py's SOURCE, generic over the SIMD variant: rows of W (int8, BF16
//! or FP32) against FP32 activations with register tiles of partial sums, and the element-wise steps that
//! round to BF16 where model.py does. Every output element is one thread's fixed-order reduction, so the
//! results are deterministic and equal the C++ kernels' bit for bit.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::missing_safety_doc, clippy::excessive_precision)]

use crate::simd::{bf, rb, Simd};

/// A weight element: int8, BF16 (`u16`) or FP32.
pub trait Weight: Copy + 'static {
    unsafe fn ld<S: Simd>(p: *const Self) -> S::V;
    fn f32(self) -> f32;
}

impl Weight for i8 {
    #[inline(always)]
    unsafe fn ld<S: Simd>(p: *const i8) -> S::V {
        S::ld_i8(p)
    }
    #[inline(always)]
    fn f32(self) -> f32 {
        self as f32
    }
}

impl Weight for u16 {
    #[inline(always)]
    unsafe fn ld<S: Simd>(p: *const u16) -> S::V {
        S::ld_bf16(p)
    }
    #[inline(always)]
    fn f32(self) -> f32 {
        bf(self)
    }
}

impl Weight for f32 {
    #[inline(always)]
    unsafe fn ld<S: Simd>(p: *const f32) -> S::V {
        S::ld(p)
    }
    #[inline(always)]
    fn f32(self) -> f32 {
        self
    }
}

/// A matrix read in place: rows of `kind` (0 BF16, 1 int8, 2 FP32), int8 rows with BF16 scales.
#[derive(Clone, Copy)]
pub struct Mat {
    pub w: *const u8,
    pub s: *const u16,
    pub kind: u8,
}

unsafe impl Send for Mat {}
unsafe impl Sync for Mat {}

impl Mat {
    pub const NONE: Mat = Mat { w: std::ptr::null(), s: std::ptr::null(), kind: 0 };
    pub fn is_null(&self) -> bool {
        self.w.is_null()
    }
}

/// `chunk_rows`: a chunk of an n-row matrix: about three chunks a thread, a multiple of 4, at most `most`
#[inline(always)]
pub fn chunk_rows(n: usize, nt: usize, most: usize) -> usize {
    4.max(most.min(n / (3 * nt) / 4 * 4))
}

#[inline(always)]
pub fn split(n: usize, t: usize, nt: usize, align: usize) -> (usize, usize) {
    let chunks = n.div_ceil(align);
    (n.min(chunks * t / nt * align), n.min(chunks * (t + 1) / nt * align))
}

/// acc[m * R + i] = x[m] . r[i] over K: R rows of W against M tokens, R * M sums in registers
#[inline(always)]
unsafe fn block<S: Simd, W: Weight, const R: usize, const M: usize>(x: *const *const f32, r: &[*const W; R], k_: usize, acc: *mut f32) {
    let vl = S::VL;
    let mut a = [[S::zero(); R]; M];
    let mut k = 0;
    let kv = k_ - k_ % vl;
    if M == 1 {
        // two partial sums per row: 2R independent chains
        let mut b = [S::zero(); R];
        let x0p = *x;
        while k + 2 * vl <= kv {
            let x0 = S::ld(x0p.add(k));
            let x1 = S::ld(x0p.add(k + vl));
            for i in 0..R {
                a[0][i] = S::fma(W::ld::<S>(r[i].add(k)), x0, a[0][i]);
                b[i] = S::fma(W::ld::<S>(r[i].add(k + vl)), x1, b[i]);
            }
            k += 2 * vl;
        }
        for i in 0..R {
            a[0][i] = S::add(a[0][i], b[i]);
        }
    }
    while k < kv {
        let mut w = [S::zero(); R];
        for i in 0..R {
            w[i] = W::ld::<S>(r[i].add(k)); // each weight converted once for M tokens
        }
        for m in 0..M {
            let xm = S::ld((*x.add(m)).add(k));
            for i in 0..R {
                a[m][i] = S::fma(w[i], xm, a[m][i]);
            }
        }
        k += vl;
    }
    for m in 0..M {
        let xm = *x.add(m);
        for i in 0..R {
            let mut s = S::sum(a[m][i]);
            for j in kv..k_ {
                s = S::fmaf(*xm.add(j), (*r[i].add(j)).f32(), s);
            }
            *acc.add(m * R + i) = s;
        }
    }
}

/// rows [j, j + R) of w (the last row repeated past n) against tokens [m0, m1) in groups of MB
#[inline(always)]
unsafe fn rows<S: Simd, W: Weight, const R: usize>(x: *const *const f32, m0: usize, m1: usize, w: *const W, k: usize, j: usize, n: usize, out: *mut f32, ldo: usize) {
    let mut r = [w; R];
    for (i, ri) in r.iter_mut().enumerate() {
        *ri = w.add((j + i).min(n - 1) * k);
    }
    let nr = R.min(n - j);
    let mut acc = [0f32; 6 * 4];
    let mut m = m0;
    while m < m1 {
        let mb = S::MB.min(m1 - m);
        let xs = x.add(m);
        let a = acc.as_mut_ptr();
        match if R == 4 && m1 - m0 == 1 { 0 } else { mb } {
            0 | 1 => block::<S, W, R, 1>(xs, &r, k, a),
            2 => block::<S, W, R, 2>(xs, &r, k, a),
            3 => block::<S, W, R, 3>(xs, &r, k, a),
            4 => block::<S, W, R, 4>(xs, &r, k, a),
            5 => block::<S, W, R, 5>(xs, &r, k, a),
            _ => block::<S, W, R, 6>(xs, &r, k, a),
        }
        for a in 0..mb {
            for i in 0..nr {
                *out.add((m + a) * ldo + j + i) = acc[a * R + i];
            }
        }
        m += S::MB;
    }
}

/// one row against one token, eight sums along K
#[inline(always)]
unsafe fn row1<S: Simd, W: Weight>(x: *const f32, r: *const W, k_: usize) -> f32 {
    let vl = S::VL;
    let mut a = [S::zero(); 8];
    let mut k = 0;
    while k + 8 * vl <= k_ {
        for u in 0..8 {
            a[u] = S::fma(W::ld::<S>(r.add(k + u * vl)), S::ld(x.add(k + u * vl)), a[u]);
        }
        k += 8 * vl;
    }
    while k + vl <= k_ {
        a[0] = S::fma(W::ld::<S>(r.add(k)), S::ld(x.add(k)), a[0]);
        k += vl;
    }
    for u in 1..8 {
        a[0] = S::add(a[0], a[u]);
    }
    let mut s = S::sum(a[0]);
    while k < k_ {
        s = S::fmaf(*x.add(k), (*r.add(k)).f32(), s);
        k += 1;
    }
    s
}

/// out[m * ldo + j] = x[m] . w[j] for rows j < n of w (row stride K), tokens m < M
#[inline(always)]
unsafe fn dots<S: Simd, W: Weight>(x: *const *const f32, m_: usize, w: *const W, k: usize, n: usize, out: *mut f32, ldo: usize) {
    if m_ == 1 {
        for j in 0..n {
            *out.add(j) = row1::<S, W>(*x, w.add(j * k), k);
        }
        return;
    }
    let tile = if m_ <= 64 { m_ } else { 48 };
    let mut m0 = 0;
    while m0 < m_ {
        let m1 = m_.min(m0 + tile);
        let mut j = 0;
        while j < n {
            if S::RB == 4 {
                rows::<S, W, 4>(x, m0, m1, w, k, j, n, out, ldo);
            } else {
                rows::<S, W, 2>(x, m0, m1, w, k, j, n, out, ldo);
            }
            j += S::RB;
        }
        m0 += tile;
    }
}

/// y[m * ldy + j] = (row r0 + j of A) . x[m], scaled, rounded to BF16 unless FP32 (FP32 rows)
#[inline(always)]
pub unsafe fn mm<S: Simd>(a: &Mat, k: usize, r0: usize, n: usize, x: *const *const f32, m_: usize, y: *mut f32, ldy: usize) {
    if n == 0 || m_ == 0 {
        return;
    }
    match a.kind {
        1 => dots::<S, i8>(x, m_, (a.w as *const i8).add(r0 * k), k, n, y, ldy),
        0 => dots::<S, u16>(x, m_, (a.w as *const u16).add(r0 * k), k, n, y, ldy),
        _ => return dots::<S, f32>(x, m_, (a.w as *const f32).add(r0 * k), k, n, y, ldy),
    }
    let vl = S::VL;
    for m in 0..m_ {
        let ym = y.add(m * ldy);
        let mut j = 0;
        if !a.s.is_null() {
            while j + vl <= n {
                S::st(ym.add(j), S::rb(S::mul(S::ld(ym.add(j)), S::ld_bf16(a.s.add(r0 + j)))));
                j += vl;
            }
            while j < n {
                *ym.add(j) = rb(*ym.add(j) * bf(*a.s.add(r0 + j)));
                j += 1;
            }
        } else {
            while j + vl <= n {
                S::st(ym.add(j), S::rb(S::ld(ym.add(j))));
                j += vl;
            }
            while j < n {
                *ym.add(j) = rb(*ym.add(j));
                j += 1;
            }
        }
    }
}

/// y[m * N + d] = x[m] @ W[:, d] for d in [d0, d1), W: K x N BF16 (the x @ W layout)
#[inline(always)]
pub unsafe fn kn<S: Simd>(x: *const f32, k_: usize, m_: usize, w: *const u16, n: usize, d0: usize, d1: usize, y: *mut f32) {
    let vl = S::VL;
    for m in 0..m_ {
        let xm = x.add(m * k_);
        let ym = y.add(m * n);
        let mut d = d0;
        while d + 2 * vl <= d1 {
            let mut a = S::zero();
            let mut b = S::zero();
            for k in 0..k_ {
                let xk = *xm.add(k);
                if xk == 0.0 {
                    continue;
                }
                let s = S::set(xk);
                let wp = w.add(k * n + d);
                a = S::fma(S::ld_bf16(wp), s, a);
                b = S::fma(S::ld_bf16(wp.add(vl)), s, b);
            }
            S::st(ym.add(d), a);
            S::st(ym.add(d + vl), b);
            d += 2 * vl;
        }
        while d < d1 {
            let mut a = 0f32;
            for k in 0..k_ {
                a = S::fmaf(*xm.add(k), bf(*w.add(k * n + d)), a);
            }
            *ym.add(d) = a;
            d += 1;
        }
    }
}

#[inline(always)]
pub unsafe fn dot<S: Simd>(a: *const f32, b: *const u16, n: usize) -> f32 {
    let mut s = 0f32;
    for i in 0..n {
        s = S::fmaf(*a.add(i), bf(*b.add(i)), s);
    }
    s
}

/// rms_norm, eps = FP32 epsilon, to BF16 (y may be x)
#[inline(always)]
pub unsafe fn norm<S: Simd>(x: *const f32, y: *mut f32, n: usize) {
    let vl = S::VL;
    let mut a = S::zero();
    let mut i = 0;
    while i + vl <= n {
        a = S::fma(S::ld(x.add(i)), S::ld(x.add(i)), a);
        i += vl;
    }
    let mut s = S::sum(a);
    while i < n {
        s = S::fmaf(*x.add(i), *x.add(i), s);
        i += 1;
    }
    let r = 1.0 / (s / n as f32 + 1.192_092_895_507_812_5e-7).sqrt();
    let vr = S::set(r);
    i = 0;
    while i + vl <= n {
        S::st(y.add(i), S::rb(S::mul(S::ld(x.add(i)), vr)));
        i += vl;
    }
    while i < n {
        *y.add(i) = rb(*x.add(i) * r);
        i += 1;
    }
}

/// y = rb(a + rb(c * b))
#[inline(always)]
pub unsafe fn axpy_rb<S: Simd>(y: *mut f32, a: *const f32, c: f32, b: *const f32, n: usize) {
    let vl = S::VL;
    let mut i = 0;
    while i + vl <= n {
        S::st(y.add(i), S::rb(S::add(S::ld(a.add(i)), S::rb(S::mul(S::set(c), S::ld(b.add(i)))))));
        i += vl;
    }
    while i < n {
        *y.add(i) = rb(*a.add(i) + rb(c * *b.add(i)));
        i += 1;
    }
}

/// y = rb(y + b)
#[inline(always)]
pub unsafe fn add_rb<S: Simd>(y: *mut f32, b: *const f32, n: usize) {
    let vl = S::VL;
    let mut i = 0;
    while i + vl <= n {
        S::st(y.add(i), S::rb(S::add(S::ld(y.add(i)), S::ld(b.add(i)))));
        i += vl;
    }
    while i < n {
        *y.add(i) = rb(*y.add(i) + *b.add(i));
        i += 1;
    }
}

/// gelu, tanh approximation, to BF16. GCC fuses `vexp(..) + 1` into one FMA (`exp1`).
#[inline(always)]
pub unsafe fn gelu<S: Simd>(a: *mut f32, n: usize) {
    let vl = S::VL;
    let (c, k) = (0.797_884_560_802_865_4f32, 0.044_715f32);
    let mut i = 0;
    while i + vl <= n {
        let x = S::ld(a.add(i));
        let u = S::mul(S::set(c), S::fma(S::mul(S::set(k), S::mul(x, x)), x, x));
        let z = S::mul(S::set(2.0), S::min(S::max(u, S::set(-9.0)), S::set(9.0)));
        let t = S::sub(S::set(1.0), S::div(S::set(2.0), S::exp1(z)));
        S::st(a.add(i), S::mul(S::mul(S::set(0.5), x), S::add(S::set(1.0), t)));
        i += vl;
    }
    while i < n {
        let v = *a.add(i);
        *a.add(i) = 0.5 * v * (1.0 + (c * S::fmaf(k * v * v, v, v)).tanh());
        i += 1;
    }
    for i in 0..n {
        *a.add(i) = rb(*a.add(i));
    }
}

/// y = rb(rb(silu(a)) * b)
#[inline(always)]
pub unsafe fn silu_mul<S: Simd>(a: *const f32, b: *const f32, y: *mut f32, n: usize) {
    let vl = S::VL;
    let mut t = [0f32; 16];
    let mut i = 0;
    while i + vl <= n {
        let x = S::ld(a.add(i));
        S::st(t.as_mut_ptr(), S::div(x, S::exp1(S::sub(S::zero(), x))));
        for j in 0..vl {
            *y.add(i + j) = rb(rb(t[j]) * *b.add(i + j));
        }
        i += vl;
    }
    while i < n {
        let v = *a.add(i);
        *y.add(i) = rb(rb(v / (1.0 + (-v).exp())) * *b.add(i));
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{tobf, Avx2, Scalar};

    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    /// mm (BF16 rows, int8 rows) against a naive f64 reference for every tile shape, all three variants
    unsafe fn mm_vs_f64<S: Simd>() {
        let mut r = rng(7);
        for &(m, n, k) in &[(1usize, 5usize, 37usize), (2, 3, 64), (6, 7, 64), (7, 9, 80), (70, 4, 32)] {
            let x: Vec<Vec<f32>> = (0..m).map(|_| (0..k).map(|_| rb(r())).collect()).collect();
            let xp: Vec<*const f32> = x.iter().map(|v| v.as_ptr()).collect();
            let wb: Vec<u16> = (0..n * k).map(|_| tobf(r())).collect();
            let wi: Vec<i8> = (0..n * k).map(|_| (r() * 127.0) as i8).collect();
            let sc: Vec<u16> = (0..n).map(|_| tobf(r().abs() / 100.0 + 1e-3)).collect();
            let mut y = vec![0f32; m * n];
            let naive_b = |i: usize, j: usize| x[i].iter().zip(&wb[j * k..]).map(|(a, b)| *a as f64 * bf(*b) as f64).sum::<f64>();
            let naive_i = |i: usize, j: usize| x[i].iter().zip(&wi[j * k..]).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() * bf(sc[j]) as f64;
            let cases: [(Mat, &dyn Fn(usize, usize) -> f64); 2] = [
                (Mat { w: wb.as_ptr() as *const u8, s: std::ptr::null(), kind: 0 }, &naive_b),
                (Mat { w: wi.as_ptr() as *const u8, s: sc.as_ptr(), kind: 1 }, &naive_i),
            ];
            for (mat, naive) in cases {
                mm::<S>(&mat, k, 0, n, xp.as_ptr(), m, y.as_mut_ptr(), n);
                for i in 0..m {
                    for j in 0..n {
                        let (got, want) = (y[i * n + j] as f64, naive(i, j));
                        assert!((got - want).abs() <= want.abs() * 0.01 + 1e-3, "{} m{m} n{n} k{k} [{i},{j}] {got} vs {want}", S::NAME);
                        assert_eq!(rb(y[i * n + j]), y[i * n + j]); // rounded to BF16
                    }
                }
            }
        }
    }

    unsafe fn elementwise<S: Simd>() {
        let mut r = rng(3);
        let n = 29;
        let x: Vec<f32> = (0..n).map(|_| r() * 4.0).collect();
        let mut y = vec![0f32; n];
        norm::<S>(x.as_ptr(), y.as_mut_ptr(), n);
        let ms = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n as f64;
        for (a, b) in x.iter().zip(&y) {
            assert!(((*a as f64 / (ms + 1.19e-7).sqrt()) - *b as f64).abs() < 0.02);
        }
        let mut g = x.clone();
        gelu::<S>(g.as_mut_ptr(), n);
        for (a, b) in x.iter().zip(&g) {
            let t = 0.5 * a * (1.0 + (0.797_884_6 * (a + 0.044_715 * a * a * a)).tanh());
            assert!((t - b).abs() < 0.02 + t.abs() * 0.01, "gelu({a}) {b} vs {t}");
        }
        let mut s = vec![0f32; n];
        silu_mul::<S>(x.as_ptr(), x.as_ptr(), s.as_mut_ptr(), n);
        for (a, b) in x.iter().zip(&s) {
            let t = a / (1.0 + (-a).exp()) * a;
            assert!((t - b).abs() < 0.02 + t.abs() * 0.01, "silu({a}) {b} vs {t}");
        }
        let w: Vec<u16> = (0..n * 20).map(|_| tobf(r())).collect();
        let mut o = vec![0f32; 20];
        kn::<S>(x.as_ptr(), n, 1, w.as_ptr(), 20, 0, 20, o.as_mut_ptr());
        for d in 0..20 {
            let want: f64 = (0..n).map(|k| x[k] as f64 * bf(w[k * 20 + d]) as f64).sum();
            assert!((o[d] as f64 - want).abs() < 1e-4, "kn[{d}] {} vs {want}", o[d]);
        }
    }

    #[test]
    fn scalar() {
        unsafe {
            mm_vs_f64::<Scalar>();
            elementwise::<Scalar>();
        }
    }

    #[test]
    fn avx2() {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                mm_vs_f64::<Avx2>();
                elementwise::<Avx2>();
            }
        }
    }

    #[test]
    fn avx512() {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                mm_vs_f64::<crate::simd::Avx512>();
                elementwise::<crate::simd::Avx512>();
            }
        }
    }

    #[test]
    fn chunking() {
        assert_eq!(chunk_rows(1536, 3, 64), 64);
        assert_eq!(chunk_rows(1536, 16, 128), 32);
        assert_eq!(chunk_rows(64, 3, 64), 4);
        assert_eq!(split(64, 0, 3, 16), (0, 16));
        assert_eq!(split(64, 2, 3, 16), (32, 64));
    }
}
