//! The native search loop's pieces, shared by the coverage and KL searchers: the Rust-owned slot buffers a
//! tree's leaves are written to (one per view), the leaf items of one call built into a `Server` request, the
//! views' W/D/L mixing (tree.py's `mixed`) and Python's monotonic clock for deadlines.

use crate::server::{Inner, Request};

pub const NONE: u32 = u32::MAX;
const WDL: usize = 2413;

/// A game's cache as Python hands it: k, v, e data pointers, capacity, filled rows.
#[derive(Clone, Copy)]
pub struct CacheRef {
    pub k: usize,
    pub v: usize,
    pub e: usize,
    pub cap: usize,
    pub n0: usize,
}

impl From<(usize, usize, usize, usize, usize)> for CacheRef {
    fn from(t: (usize, usize, usize, usize, usize)) -> CacheRef {
        CacheRef { k: t.0, v: t.1, e: t.2, cap: t.3, n0: t.4 }
    }
}

/// A tree's keys, values and embeddings by slot: k, v [L, H, cap, hd], e [cap, D] in BF16, zeroed (pages
/// touched as slots are written).
pub struct Slots {
    l: usize,
    h: usize,
    hd: usize,
    d: usize,
    pub cap: usize,
    k: Vec<u16>,
    v: Vec<u16>,
    e: Vec<u16>,
}

impl Slots {
    pub fn new(dims: [usize; 5], cap: usize) -> Slots {
        let [l, h, hd, d, _] = dims;
        Slots { l, h, hd, d, cap, k: vec![0; l * h * cap * hd], v: vec![0; l * h * cap * hd], e: vec![0; cap * d] }
    }

    /// Room for `cap` slots: the rows re-laid at the larger capacity (slot positions are absolute).
    pub fn ensure(&mut self, cap: usize) {
        if cap <= self.cap {
            return;
        }
        let new = cap.max(2 * self.cap);
        let (l, h, hd, old) = (self.l, self.h, self.hd, self.cap);
        for buf in [&mut self.k, &mut self.v] {
            let mut grown = vec![0u16; l * h * new * hd];
            for r in 0..l * h {
                grown[r * new * hd..r * new * hd + old * hd].copy_from_slice(&buf[r * old * hd..(r + 1) * old * hd]);
            }
            *buf = grown;
        }
        self.e.resize(new * self.d, 0);
        self.cap = new;
    }

    fn ptrs(&self) -> [usize; 3] {
        [self.k.as_ptr() as usize, self.v.as_ptr() as usize, self.e.as_ptr() as usize]
    }
}

/// The leaf items of one call, one engine item each.
pub struct Batch {
    req: Request,
    n: usize,
    v: usize,
}

impl Batch {
    pub fn new(v: usize) -> Batch {
        Batch { req: Request { ids: Vec::new(), feats: Vec::new(), boards: Vec::new(), meta: Vec::new(), caches: Vec::new(), paths: Vec::new(), out: std::ptr::null_mut() }, n: 0, v }
    }

    /// A leaf: `token` after the cache's rows and the slots `path` (its ancestors below the root, in order),
    /// its keys, values and embedding to slot `dest`.
    pub fn leaf(&mut self, c: &CacheRef, s: &Slots, path: &[u32], dest: u32, token: i64, feats: [f32; 3], board: &[u8; 68]) {
        let r = &mut self.req;
        let poff = r.paths.len() as i64;
        r.paths.extend(path.iter().map(|&x| x as i64));
        r.meta.extend([c.n0 as i64, 1, c.cap as i64, self.n as i64, s.cap as i64, path.len() as i64, poff, dest as i64]);
        let p = s.ptrs();
        r.caches.extend([c.k, c.v, c.e, p[0], p[1], p[2]]);
        r.ids.push(token);
        r.feats.extend(feats);
        r.boards.extend_from_slice(board);
        self.n += 1;
    }

    /// The logits [n, V] (float32) from the server, or the engine's error.
    pub fn submit(mut self, server: &Inner) -> Result<Vec<f32>, String> {
        let mut out = vec![0f32; self.n * self.v];
        self.req.out = out.as_mut_ptr();
        match server.submit(self.req) {
            0 => Ok(out),
            1 => Err("token outside the vocabulary".into()),
            2 => Err("board state out of range".into()),
            3 => Err("tokens, paths or slots past the cache, the tree or the context".into()),
            _ => Err("the server has stopped".into()),
        }
    }
}

/// tree.py's `mixed` for one leaf: the first view's logits (as float64) with the W/D/L entries set to the
/// log of the views' mean W/D/L probabilities, in numpy's operation order.
pub fn mixed(rows: &[&[f32]]) -> Vec<f64> {
    let mut z: Vec<f64> = rows[0].iter().map(|&x| x as f64).collect();
    let mut mean = [0f64; 3];
    for (v, r) in rows.iter().enumerate() {
        let x = [r[WDL] as f64, r[WDL + 1] as f64, r[WDL + 2] as f64];
        let m = x[0].max(x[1]).max(x[2]);
        let q = x.map(|x| (x - m).exp());
        let s = ((0. + q[0]) + q[1]) + q[2];
        for k in 0..3 {
            let p = q[k] / s;
            mean[k] = if v == 0 { p } else { mean[k] + p };
        }
    }
    for k in 0..3 {
        z[WDL + k] = (mean[k] / rows.len() as f64).ln();
    }
    z
}

/// Python's time.monotonic(): CLOCK_MONOTONIC seconds.
pub fn monotonic() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_grow_in_place() {
        let mut s = Slots::new([2, 3, 4, 5, 2432], 2);
        for (i, x) in s.k.iter_mut().enumerate() {
            *x = i as u16;
        }
        s.e[7] = 9;
        s.ensure(2);
        assert_eq!(s.cap, 2);
        s.ensure(3);
        assert_eq!((s.cap, s.k.len(), s.e.len(), s.e[7]), (4, 2 * 3 * 4 * 4, 4 * 5, 9));
        for r in 0..6 {
            assert_eq!(&s.k[r * 16..r * 16 + 8], &(r as u16 * 8..r as u16 * 8 + 8).collect::<Vec<_>>()[..]);
            assert!(s.k[r * 16 + 8..r * 16 + 16].iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn mixed_logs_mean_wdl() {
        let mut a = vec![0f32; 2432];
        let mut b = vec![1f32; 2432];
        a[WDL..WDL + 3].copy_from_slice(&[1., 0., 0.]);
        b[WDL..WDL + 3].copy_from_slice(&[0., 0., 1.]);
        let z = mixed(&[&a, &b]);
        assert_eq!(z[0], 0.);
        let e = 1f64.exp();
        let (hi, lo) = (e / (e + 2.), 1. / (e + 2.));
        assert!((z[WDL] - ((hi + lo) / 2.).ln()).abs() < 1e-15 && (z[WDL + 1] - lo.ln()).abs() < 1e-15);
        let one = mixed(&[&a]);
        assert!((one[WDL].exp() - hi).abs() < 1e-15);
        assert!(monotonic() > 0.);
    }

    #[test]
    fn batch_items() {
        let mut b = Batch::new(2432);
        let s = Slots::new([1, 1, 4, 4, 2432], 3);
        let c = CacheRef { k: 1, v: 2, e: 3, cap: 64, n0: 20 };
        b.leaf(&c, &s, &[1], 2, 400, [1., 2., 3.], &[0; 68]);
        b.leaf(&c, &s, &[], 1, 401, [4., 5., 6.], &[0; 68]);
        assert_eq!(b.req.meta, [20, 1, 64, 0, 3, 1, 0, 2, 20, 1, 64, 1, 3, 0, 1, 1]);
        assert_eq!((b.req.ids.clone(), b.req.paths.clone(), b.req.feats.len(), b.req.boards.len()), (vec![400, 401], vec![1], 6, 136));
        assert_eq!(&b.req.caches[..3], &[1, 2, 3]);
    }
}
