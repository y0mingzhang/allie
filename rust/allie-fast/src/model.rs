//! The C++ `Engine` of fast.py's SOURCE: one step of Allie 2.0 (input embedding, board CNN, every block
//! with attention over per-game KV caches, the head) on a pool of threads, with barriers between a block's
//! phases and its matrices shared out in chunks. Weights and caches are read and written in place through
//! raw pointers the caller keeps alive. Generic over the SIMD variant; `run_*` are the per-ISA entry points.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::missing_safety_doc, clippy::neg_cmp_op_on_partial_ord)]

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicI32, Ordering::Relaxed};
use std::time::Instant;

use crate::kernels::{add_rb, axpy_rb, chunk_rows, dot, gelu, kn, mm, norm, silu_mul, split, Mat};
use crate::pool::{Padded, Pool};
use crate::simd::{bf, rb, sigm, tobf, Avx2, Avx512, Scalar, Simd};

pub const NPHASE: usize = 17;
const S_START: usize = 0;
const E_CNN: usize = 1;
const E_ROWS: usize = 2;
const E_SMEAR: usize = 3;
const B_NORM: usize = 4;
const B_QKV: usize = 5;
const B_ROTARY: usize = 6;
const B_ATTN: usize = 7;
const B_O: usize = 8;
const B_NORM2: usize = 9;
const B_ROUTER: usize = 10;
const B_TOPK: usize = 11;
const B_GROUP: usize = 12;
const B_UP: usize = 13;
const B_DOWN: usize = 14;
const H_NORM: usize = 15;
const H_HEAD: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    Avx512,
    Avx2,
    Scalar,
}

impl Isa {
    /// The best variant this CPU runs, or a lower one named by ALLIE_RUST_ISA (avx2 | scalar).
    pub fn detect() -> Isa {
        let best = if is_x86_feature_detected!("avx512f") {
            Isa::Avx512
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            Isa::Avx2
        } else {
            Isa::Scalar
        };
        match std::env::var("ALLIE_RUST_ISA").as_deref() {
            Ok("scalar") => Isa::Scalar,
            Ok("avx2") if best == Isa::Avx512 => Isa::Avx2,
            _ => best,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Avx512 => Avx512::NAME,
            Isa::Avx2 => Avx2::NAME,
            Isa::Scalar => Scalar::NAME,
        }
    }
}

/// A buffer the threads write through a raw pointer; grown only between steps.
pub struct Buf {
    v: Vec<f32>,
    p: *mut f32,
}

impl Buf {
    fn new() -> Buf {
        Buf { v: Vec::new(), p: std::ptr::null_mut() }
    }
    fn grow(&mut self, n: usize) {
        if self.v.len() < n {
            self.v.resize(n, 0.0);
        }
        self.p = self.v.as_mut_ptr();
    }
}

pub struct Layer {
    qkv: Mat,
    o: Mat,
    fc: Mat,
    proj: Mat,
    sup: Mat,
    sdown: Mat,
    up: *const u8,
    down: *const u8,
    ups: *const u16,
    downs: *const u16,
    gates: *const u16,
    router: *const f32,
    bias: *const f32,
    mu: *const f32,
    g: usize,
    ve: i64,
    skin: i64,
    skout: i64,
}

/// an item: `len` tokens appended to a game's cache (k, v: [L, H, cap, hd], e: [cap, D]) after its n0 rows; or,
/// a path item (scap > 0), one search leaf evaluated in place: its token at position n0 + plen attends to the
/// cache's n0 rows and then to slots paths[path..path + plen] of a slot buffer (sk, sv, se, capacity scap: the
/// leaf's ancestors in order), and its own keys, values and embedding go to slot `dest`; the cache is read only
#[derive(Clone, Copy)]
struct Seq {
    n0: usize,
    off: usize,
    cap: usize,
    k: *mut u16,
    v: *mut u16,
    e: *mut u16,
    sk: *mut u16,
    sv: *mut u16,
    se: *mut u16,
    scap: usize,
    path: usize,
    plen: usize,
    dest: usize,
}

impl Seq {
    #[inline(always)]
    fn leaf(&self) -> bool {
        self.scap > 0
    }

    /// token k's position
    #[inline(always)]
    fn pos(&self, k: usize) -> usize {
        self.n0 + if self.leaf() { self.plen } else { k - self.off }
    }
}

/// one up / fc matrix (rows: gate halves, then value halves) for some tokens
#[derive(Clone, Copy)]
struct Seg {
    a: Mat,
    pairs: usize,
    ntok: usize,
    ldo: usize,
    x: *const *const f32,
    out: *mut f32,
}

#[derive(Clone, Copy)]
struct Chunk {
    seg: usize,
    p0: usize,
    n: usize,
}

/// each token's experts and gates, the tokens grouped by expert, the up chunks
#[derive(Default)]
struct Routing {
    idx: Vec<usize>,
    cnt: Vec<usize>,
    start: Vec<usize>,
    at: Vec<usize>,
    stok: Vec<usize>,
    gate: Vec<f32>,
    sgate: Vec<f32>,
    ptok: Vec<*const f32>,
    segs: Vec<Seg>,
    upch: Vec<Chunk>,
    active: Vec<usize>,
}

impl Routing {
    fn size(&mut self, t: usize, keep: usize, e: usize) {
        let n = t * keep;
        if self.idx.len() < n {
            self.idx.resize(n, 0);
            self.stok.resize(n, 0);
            self.gate.resize(n, 0.0);
            self.sgate.resize(n, 0.0);
            self.ptok.resize(n, std::ptr::null());
        }
        self.cnt.resize(e, 0);
        self.start.resize(e, 0);
        self.at.resize(e, 0);
    }
}

/// a thread's own copy of the per-token steps (solo)
#[derive(Default)]
struct Priv {
    x: Vec<f32>,
    h: Vec<f32>,
    hf: Vec<f32>,
    g: Vec<f32>,
    ph: Vec<*const f32>,
    phf: Vec<*const f32>,
    r: Routing,
}

pub struct Engine {
    pub l: usize,
    pub d: usize,
    pub h: usize,
    pub hd: usize,
    pub v: usize,
    pub e: usize,
    pub topk: usize,
    pub keep: usize,
    pub eh: usize,
    pub sh: usize,
    pub dh: usize,
    pub ctx: usize,
    backout_layer: i64,
    scale: f32,
    floor: f32,
    q8: bool,
    embed: *const u16,
    embed2: *const u16,
    lm_head: *const u16,
    feat: *const u16,
    smear_gate: *const u16,
    cosv: *const u16,
    sinv: *const u16,
    bmeta: *const u16,
    bout: *const u16,
    skip_gate: [*const u16; 3],
    scal: *const f32,
    x0l: *const f32,
    w1: Vec<f32>,
    wr: Vec<f32>,
    wsq: Vec<f32>,
    cnn: [Buf; 2],
    ve: Vec<*const u16>,
    layers: Vec<Layer>,
    pub pool: Pool,
    pub isa: Isa,
    // one step
    t: usize,
    s: usize,
    ids: *const i64,
    feats: *const f32,
    boards: *const u8,
    out: *mut f32,
    seq: Vec<Seq>,
    paths: Vec<usize>,
    tseq: Vec<usize>,
    last: Vec<usize>,
    units: Vec<usize>,
    f64_: Buf,
    b544: Buf,
    clk: Buf,
    brd: Buf,
    e_: Buf,
    x: Buf,
    x0: Buf,
    x02: Buf,
    hb: Buf,
    hf: Buf,
    qkv: Buf,
    g: Buf,
    q: Buf,
    y: Buf,
    tmp: Buf,
    skip: [Buf; 3],
    bko: Buf,
    rs: Buf,
    acc: Buf,
    hid: Buf,
    shid: Buf,
    xf: Buf,
    z: Buf,
    ph: Vec<*const f32>,
    py: Vec<*const f32>,
    phf: Vec<*const f32>,
    pxf: Vec<*const f32>,
    phid: Vec<*const f32>,
    pshid: Vec<*const f32>,
    shared: UnsafeCell<Routing>,
    ctr: [Padded<AtomicI32>; 2],
    phase: Vec<Padded<UnsafeCell<i32>>>,
    scratch: Vec<Buf>,
    priv_: Vec<UnsafeCell<Priv>>,
    prof: bool,
    ptime: UnsafeCell<[f64; NPHASE]>,
    plast: UnsafeCell<f64>,
    start: Instant,
}

// The raw pointers address memory the Python side owns and keeps alive for the engine's lifetime, and a
// step's threads partition their writes; one step runs at a time.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

#[inline(always)]
fn padded(p: usize) -> usize {
    (p / 8 + 1) * 10 + p % 8 + 1
}

fn clockfeat(t: *const f32, f: *mut f32) {
    unsafe {
        for j in 0..64 {
            *f.add(j) = 0.0;
        }
        for j in 0..3 {
            let tj = *t.add(j);
            if !(tj >= 0.0) {
                continue;
            }
            let v = (if tj < 0.0 { 0.0 } else { tj }).ln_1p() / 10.0;
            for k in 0..8 {
                let a = v * (std::f32::consts::PI * (1u32 << k) as f32);
                *f.add(j * 18 + k) = rb(a.sin());
                *f.add(j * 18 + 8 + k) = rb(a.cos());
            }
            *f.add(j * 18 + 16) = rb(v);
            *f.add(j * 18 + 17) = 1.0;
        }
    }
}

/// q . k over the nc vectors of a BF16 row
#[inline(always)]
unsafe fn score<S: Simd>(qr: &[S::V; 32], nc: usize, kr: *const u16) -> f32 {
    let mut a = S::zero();
    for c in 0..nc {
        a = S::fma(qr[c], S::ld_bf16(kr.add(c * S::VL)), a);
    }
    S::sum(a)
}

/// a short block of weights fetched before it is read
#[inline(always)]
unsafe fn ahead(p: *const u8, bytes: usize) {
    let mut o = 0;
    while o < bytes {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p.add(o) as *const i8);
        o += 64;
    }
}

fn board_pieces(st: *const u8, piece: &mut [usize; 100]) {
    *piece = [13; 100]; // off the board: a zero weight row
    for p in 0..64 {
        piece[padded(p)] = unsafe { *st.add(p) } as usize;
    }
}

/// pass 0: the first conv on the one-hot board; passes 1, 2: residual convs. Squares [p0, p1) of
/// out = gelu(conv(a)) (+ a for the residual passes), each rounded to BF16
#[inline(always)]
unsafe fn board_pass<S: Simd>(w1: *const f32, wr: *const f32, pass: usize, piece: &[usize; 100], a: *const f32, out: *mut f32, p0: usize, p1: usize) {
    let vl = S::VL;
    let nc = 32 / vl; // C
    let np = 8 / nc; // NP squares at once: eight independent sums
    let mut y = [0f32; 128];
    let mut p = p0;
    while p < p1 {
        let mut acc = [[S::zero(); 4]; 4];
        for tap in 0..9 {
            let mut q = [0usize; 4];
            for (j, qj) in q.iter_mut().enumerate().take(np) {
                *qj = ((p + j) / 8 + tap / 3) * 10 + (p + j) % 8 + tap % 3;
            }
            if pass == 0 {
                // one-hot input: the weights of the piece on each neighbour
                for j in 0..np {
                    for c in 0..nc {
                        acc[j][c] = S::add(acc[j][c], S::ld(w1.add((piece[q[j]] * 9 + tap) * 32 + c * vl)));
                    }
                }
                continue;
            }
            let w = wr.add(((pass - 1) * 9 + tap) * 32 * 32);
            for ci in 0..32 {
                let mut wc = [S::zero(); 4];
                for (c, wcc) in wc.iter_mut().enumerate().take(nc) {
                    *wcc = S::ld(w.add(ci * 32 + c * vl));
                }
                for j in 0..np {
                    let x = S::set(*a.add(q[j] * 32 + ci));
                    for c in 0..nc {
                        acc[j][c] = S::fma(x, wc[c], acc[j][c]);
                    }
                }
            }
        }
        for j in 0..np {
            for c in 0..nc {
                S::st(y.as_mut_ptr().add(j * 32 + c * vl), S::rb(acc[j][c]));
            }
        }
        gelu::<S>(y.as_mut_ptr(), np * 32);
        for j in 0..np {
            let ap = a.add(padded(p + j) * 32);
            let o = out.add(padded(p + j) * 32);
            for c in 0..32 {
                *o.add(c) = if pass != 0 { rb(*ap.add(c) + y[j * 32 + c]) } else { y[j * 32 + c] };
            }
        }
        p += np;
    }
}

/// squeeze, the castling / en-passant / side features, layer norm: board.output's 544 inputs. GCC
/// vectorizes the products and adds them in order: no fused multiply-adds here.
unsafe fn board_final(wsq: *const f32, bmeta: *const u16, st: *const u8, a: *const f32, v: *mut f32) {
    for o in 0..8 {
        for p in 0..64 {
            let ap = a.add(padded(p) * 32);
            let mut s = 0f32;
            for c in 0..32 {
                s += *wsq.add(o * 32 + c) * *ap.add(c);
            }
            *v.add(o * 64 + p) = rb(s);
        }
    }
    let (side, rights, ep) = (*st.add(64) as usize, *st.add(65) as usize, *st.add(66) as usize);
    for j in 0..32 {
        *v.add(512 + j) = rb(bf(*bmeta.add(side * 32 + j)) + bf(*bmeta.add((2 + rights) * 32 + j)) + bf(*bmeta.add((18 + ep) * 32 + j)));
    }
    let (mut mean, mut var) = (0f32, 0f32);
    for j in 0..544 {
        mean += *v.add(j);
    }
    mean /= 544.0;
    for j in 0..544 {
        let d = *v.add(j) - mean;
        var += d * d;
    }
    let r = 1.0 / (var / 544.0 + 1e-5).sqrt();
    for j in 0..544 {
        *v.add(j) = rb((*v.add(j) - mean) * r);
    }
}

impl Engine {
    /// cfg: L D H hd V nve E topk keep eh sh dh int8 ctx backout_layer threads 0 (unused), then per layer
    /// G ve skip_in skip_out. glob: embed embed2 lm_head feat_embed smear_gate scalars x0_lambdas cos sin
    /// board.first board.residual.0 board.residual.1 board.squeeze board.meta board.output skip_gate.0-2
    /// value_embed.*. lay, per layer (20): qkv qkv_s o o_s gates fc fc_s proj proj_s router moe_bias mu
    /// up up_s down down_s shared_up shared_up_s shared_down shared_down_s. Pointers as integers.
    pub fn new(cfg: &[i64], fl: (f64, f64), glob: &[usize], lay: &[usize], cpus: Option<Vec<i32>>, groups: Option<Vec<i32>>, spin: f64) -> Result<Engine, String> {
        if cfg.len() < 17 {
            return Err("cfg: 17 values then 4 per layer".into());
        }
        let u = |i: usize| cfg[i].max(0) as usize;
        let (l, nve) = (u(0), u(5));
        if cfg.len() < 17 + 4 * l || glob.len() != 18 + nve || lay.len() != 20 * l {
            return Err(format!("cfg {} glob {} lay {}: wrong lengths for {l} layers, {nve} value embeddings", cfg.len(), glob.len(), lay.len()));
        }
        let bp = |j: usize| glob[j] as *const u16;
        let (bfirst, bres, bsq) = (bp(9), [bp(10), bp(11)], bp(12));
        let mut w1 = vec![0f32; 14 * 9 * 32];
        let mut wr = vec![0f32; 2 * 9 * 32 * 32];
        unsafe {
            for co in 0..32 {
                for tap in 0..9 {
                    for ci in 0..13 {
                        w1[(ci * 9 + tap) * 32 + co] = bf(*bfirst.add((co * 13 + ci) * 9 + tap));
                    }
                    for j in 0..2 {
                        for ci in 0..32 {
                            wr[((j * 9 + tap) * 32 + ci) * 32 + co] = bf(*bres[j].add((co * 32 + ci) * 9 + tap));
                        }
                    }
                }
            }
        }
        let wsq = (0..8 * 32).map(|i| bf(unsafe { *bsq.add(i) })).collect();
        let q8 = cfg[12] != 0;
        let kind = q8 as u8;
        let layers = (0..l)
            .map(|i| {
                let p = &lay[20 * i..20 * i + 20];
                let c = &cfg[17 + 4 * i..21 + 4 * i];
                let mat = |a: usize| Mat { w: p[a] as *const u8, s: p[a + 1] as *const u16, kind };
                Layer {
                    qkv: mat(0),
                    o: mat(2),
                    gates: p[4] as *const u16,
                    fc: mat(5),
                    proj: mat(7),
                    router: p[9] as *const f32,
                    bias: p[10] as *const f32,
                    mu: p[11] as *const f32,
                    up: p[12] as *const u8,
                    ups: p[13] as *const u16,
                    down: p[14] as *const u8,
                    downs: p[15] as *const u16,
                    sup: mat(16),
                    sdown: mat(18),
                    g: c[0].max(0) as usize,
                    ve: c[1],
                    skin: c[2],
                    skout: c[3],
                }
            })
            .collect();
        let mut cnn = [Buf::new(), Buf::new()];
        cnn[0].grow(100 * 32);
        cnn[1].grow(100 * 32);
        Ok(Engine {
            l,
            d: u(1),
            h: u(2),
            hd: u(3),
            v: u(4),
            e: u(6),
            topk: u(7),
            keep: u(8),
            eh: u(9),
            sh: u(10),
            dh: u(11),
            ctx: u(13),
            backout_layer: cfg[14],
            scale: fl.0 as f32,
            floor: fl.1 as f32,
            q8,
            embed: bp(0),
            embed2: bp(1),
            lm_head: bp(2),
            feat: bp(3),
            smear_gate: bp(4),
            scal: glob[5] as *const f32,
            x0l: glob[6] as *const f32,
            cosv: bp(7),
            sinv: bp(8),
            bmeta: bp(13),
            bout: bp(14),
            skip_gate: [bp(15), bp(16), bp(17)],
            w1,
            wr,
            wsq,
            cnn,
            ve: (0..nve).map(|j| bp(18 + j)).collect(),
            layers,
            pool: Pool::new(u(15), cpus, groups, spin),
            isa: Isa::detect(),
            t: 0,
            s: 0,
            ids: std::ptr::null(),
            feats: std::ptr::null(),
            boards: std::ptr::null(),
            out: std::ptr::null_mut(),
            seq: Vec::new(),
            paths: Vec::new(),
            tseq: Vec::new(),
            last: Vec::new(),
            units: Vec::new(),
            f64_: Buf::new(),
            b544: Buf::new(),
            clk: Buf::new(),
            brd: Buf::new(),
            e_: Buf::new(),
            x: Buf::new(),
            x0: Buf::new(),
            x02: Buf::new(),
            hb: Buf::new(),
            hf: Buf::new(),
            qkv: Buf::new(),
            g: Buf::new(),
            q: Buf::new(),
            y: Buf::new(),
            tmp: Buf::new(),
            skip: [Buf::new(), Buf::new(), Buf::new()],
            bko: Buf::new(),
            rs: Buf::new(),
            acc: Buf::new(),
            hid: Buf::new(),
            shid: Buf::new(),
            xf: Buf::new(),
            z: Buf::new(),
            ph: Vec::new(),
            py: Vec::new(),
            phf: Vec::new(),
            pxf: Vec::new(),
            phid: Vec::new(),
            pshid: Vec::new(),
            shared: UnsafeCell::new(Routing::default()),
            ctr: [Padded(AtomicI32::new(0)), Padded(AtomicI32::new(0))],
            phase: Vec::new(),
            scratch: Vec::new(),
            priv_: Vec::new(),
            prof: false,
            ptime: UnsafeCell::new([0.0; NPHASE]),
            plast: UnsafeCell::new(0.0),
            start: Instant::now(),
        })
    }

    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Seconds spent in each phase since the last call; on: keep profiling.
    pub fn profile(&mut self, on: bool) -> [f64; NPHASE] {
        let t = std::mem::replace(self.ptime.get_mut(), [0.0; NPHASE]);
        self.prof = on;
        t
    }

    /// One step: T tokens of S items. meta: per item n0 len cap off scap plen poff dest (scap 0: a plain item,
    /// the rest ignored; else a path item: len 1, its slots paths[poff..poff + plen], all slots < scap);
    /// caches: per item the cache's k v e then the slot buffer's k v e (0 for a plain item), pointers as
    /// integers; out: S x V. Returns 0, or 1 for a token outside the vocabulary, 2 for a board state outside
    /// its ranges, 3 for spans, paths or slots that do not fit.
    pub fn step(&mut self, t: usize, s: usize, ids: *const i64, feats: *const f32, boards: *const u8, meta: &[i64], caches: &[usize], paths: &[i64], out: *mut f32) -> i32 {
        if meta.len() < 8 * s || caches.len() < 6 * s {
            return 3;
        }
        let d = self.d;
        let nt = self.pool.n();
        unsafe {
            for k in 0..t {
                let b = boards.add(68 * k);
                let id = *ids.add(k);
                if id < 0 || id as usize >= self.v {
                    return 1;
                }
                if (0..64).any(|i| *b.add(i) > 12) || *b.add(64) > 1 || *b.add(65) > 15 || *b.add(66) > 8 {
                    return 2;
                }
            }
        }
        let (mut off, ctx) = (0i64, self.ctx as i64);
        for q in meta[..8 * s].chunks(8) {
            let (n0, len, cap, scap, plen, poff, dest) = (q[0], q[1], q[2], q[4], q[5], q[6], q[7]);
            if len < 1 || n0 < 0 || q[3] != off {
                return 3;
            }
            if scap == 0 {
                if n0 + len > cap.min(ctx) {
                    return 3;
                }
            } else if len != 1 || n0 > cap || plen < 0 || n0 + plen + 1 > ctx || poff < 0 || poff + plen > paths.len() as i64 || !(0..scap).contains(&dest) || paths[poff as usize..(poff + plen) as usize].iter().any(|r| !(0..scap).contains(r)) {
                return 3;
            }
            off += len;
        }
        if s > 0 && off != t as i64 || s == 0 && t != 0 {
            return 3;
        }
        self.t = t;
        self.s = s;
        self.ids = ids;
        self.feats = feats;
        self.boards = boards;
        self.out = out;
        self.seq.clear();
        self.paths.clear();
        self.paths.extend(paths.iter().map(|&r| r as usize));
        self.tseq.resize(t, 0);
        self.last.resize(s, 0);
        self.units.clear();
        let qc = 1.max(32.min(t * self.h / (4 * nt) + 1));
        for si in 0..s {
            let q = &meta[8 * si..8 * si + 8];
            let c = &caches[6 * si..6 * si + 6];
            let (n0, len, cap, off) = (q[0] as usize, q[1] as usize, q[2] as usize, q[3] as usize);
            self.seq.push(Seq {
                n0,
                off,
                cap,
                k: c[0] as *mut u16,
                v: c[1] as *mut u16,
                e: c[2] as *mut u16,
                sk: c[3] as *mut u16,
                sv: c[4] as *mut u16,
                se: c[5] as *mut u16,
                scap: q[4] as usize,
                path: q[6] as usize,
                plen: q[5] as usize,
                dest: q[7] as usize,
            });
            for j in 0..len {
                self.tseq[off + j] = si;
            }
            self.last[si] = off + len - 1;
            let mut a = 0;
            while a < len {
                self.units.extend([si, off + a, off + len.min(a + qc)]);
                a += qc;
            }
        }
        let td = t * d;
        for b in [&mut self.clk, &mut self.brd, &mut self.e_, &mut self.x, &mut self.x0, &mut self.x02, &mut self.hb, &mut self.hf, &mut self.q, &mut self.y, &mut self.tmp, &mut self.bko, &mut self.acc] {
            b.grow(td);
        }
        for b in &mut self.skip {
            b.grow(td);
        }
        self.f64_.grow(t * 64);
        self.b544.grow(t * 544);
        self.qkv.grow(3 * td);
        self.g.grow(t * 2 * self.h);
        self.rs.grow(t * self.e);
        self.hid.grow(t * self.keep * self.eh);
        let w = self.sh.max(self.dh);
        self.shid.grow(t * w);
        self.xf.grow(s * d);
        self.z.grow(s * self.v);
        self.shared.get_mut().size(t, self.keep, self.e);
        let rows = |p: *mut f32, n: usize, stride: usize| -> Vec<*const f32> { (0..n).map(|k| p.wrapping_add(k * stride) as *const f32).collect() };
        self.ph = rows(self.hb.p, t, d);
        self.py = rows(self.y.p, t, d);
        self.phf = rows(self.hf.p, t, d);
        self.pshid = rows(self.shid.p, t, w);
        self.pxf = rows(self.xf.p, s, d);
        self.phid = rows(self.hid.p, t * self.keep, self.eh);
        let need = (self.ctx + 64).max(d).max(t * 128); // scores, previous, up / down blocks
        self.scratch.resize_with(nt, Buf::new);
        for b in &mut self.scratch {
            b.grow(need);
        }
        self.priv_.resize_with(nt, || UnsafeCell::new(Priv::default()));
        if t <= 2 {
            for p in &mut self.priv_ {
                let p = p.get_mut();
                for v in [&mut p.x, &mut p.h, &mut p.hf] {
                    if v.len() < td {
                        v.resize(td, 0.0);
                    }
                }
                if p.g.len() < t * 2 * self.h {
                    p.g.resize(t * 2 * self.h, 0.0);
                }
                p.ph = (0..t).map(|k| p.h.as_ptr().wrapping_add(k * d)).collect();
                p.phf = (0..t).map(|k| p.hf.as_ptr().wrapping_add(k * d)).collect();
                p.r.size(t, self.keep, self.e);
            }
        }
        if self.prof {
            *self.plast.get_mut() = self.now();
        }
        self.phase.resize_with(nt, || Padded(UnsafeCell::new(0)));
        for p in &mut self.phase {
            *p.0.get_mut() = 0;
        }
        self.ctr[0].0.store(0, Relaxed);
        self.ctr[1].0.store(0, Relaxed);
        let me: &Engine = self;
        me.pool.run(&|t| me.dispatch(t));
        0
    }

    fn dispatch(&self, t: usize) {
        unsafe {
            match self.isa {
                Isa::Avx512 => run_avx512(self, t),
                Isa::Avx2 => run_avx2(self, t),
                Isa::Scalar => self.run::<Scalar>(t),
            }
        }
    }

    #[inline(always)]
    unsafe fn run<S: Simd>(&self, t: usize) {
        let nt = self.pool.n();
        if self.prof && t == 0 {
            self.mark(S_START); // from the call: pinning, waking the pool
        }
        self.embedding::<S>(t, nt);
        for i in 0..self.l {
            self.blockstep::<S>(i, t, nt);
        }
        self.head::<S>(t, nt);
    }

    /// thread 0, profiling: the time since the last mark is phase p's
    #[inline(always)]
    unsafe fn mark(&self, p: usize) {
        let now = self.now();
        (*self.ptime.get())[p] += now - *self.plast.get();
        *self.plast.get() = now;
    }

    /// barrier; thread 0 charges the time since the last one to phase p, and clears the work counter of
    /// phase p for the phase after next
    #[inline(always)]
    unsafe fn sync(&self, t: usize, p: usize) {
        self.pool.barrier(t);
        let ph = self.phase[t].0.get();
        if t == 0 {
            self.ctr[(*ph & 1) as usize].0.store(0, Relaxed);
            if self.prof {
                self.mark(p);
            }
        }
        *ph += 1;
    }

    /// the next of n chunks of this phase's work, shared out as threads come free. (A loop body, not a
    /// closure: a closure would be compiled without the entry point's target features.)
    #[inline(always)]
    unsafe fn next_chunk(&self, t: usize, n: usize) -> Option<usize> {
        let i = self.ctr[(*self.phase[t].0.get() & 1) as usize].0.fetch_add(1, Relaxed) as usize;
        (i < n).then_some(i)
    }

    #[inline(always)]
    unsafe fn embedding<S: Simd>(&self, t: usize, nt: usize) {
        let (tt, d) = (self.t, self.d);
        let (w1, wr) = (self.w1.as_ptr(), self.wr.as_ptr());
        let mut piece = [0usize; 100];
        if tt >= nt {
            // a token per thread
            let mut a = [[0f32; 100 * 32]; 2];
            let ap = a.as_mut_ptr() as *mut f32;
            let mut k = t;
            while k < tt {
                clockfeat(self.feats.add(3 * k), self.f64_.p.add(k * 64));
                board_pieces(self.boards.add(68 * k), &mut piece);
                for pass in 0..3 {
                    board_pass::<S>(w1, wr, pass, &piece, ap.add((pass & 1) * 3200), ap.add((!pass & 1) * 3200), 0, 64);
                }
                board_final(self.wsq.as_ptr(), self.bmeta, self.boards.add(68 * k), ap.add(3200), self.b544.p.add(k * 544));
                k += nt;
            }
        } else {
            // all threads on each token: a share of the squares, a barrier between passes
            let (p0, p1) = split(64, t, nt, 8 / (32 / S::VL));
            for k in 0..tt {
                if t == 0 {
                    clockfeat(self.feats.add(3 * k), self.f64_.p.add(k * 64));
                }
                board_pieces(self.boards.add(68 * k), &mut piece);
                for pass in 0..3 {
                    board_pass::<S>(w1, wr, pass, &piece, self.cnn[pass & 1].p, self.cnn[!pass & 1].p, p0, p1);
                    self.pool.barrier(t);
                }
                if t == 0 {
                    board_final(self.wsq.as_ptr(), self.bmeta, self.boards.add(68 * k), self.cnn[1].p, self.b544.p.add(k * 544));
                }
                if k + 1 < tt {
                    self.pool.barrier(t); // the next token's first pass overwrites cnn[1]
                }
            }
        }
        self.sync(t, E_CNN);
        let (d0, d1) = split(d, t, nt, 2 * S::VL);
        kn::<S>(self.f64_.p, 64, tt, self.feat, d, d0, d1, self.clk.p);
        kn::<S>(self.b544.p, 544, tt, self.bout, d, d0, d1, self.brd.p);
        for k in 0..tt {
            let em = self.embed.add(*self.ids.add(k) as usize * d);
            let flag = *self.boards.add(68 * k + 67) as f32;
            for dd in d0..d1 {
                let i = k * d + dd;
                *self.e_.p.add(i) = rb(rb(bf(*em.add(dd)) + rb(*self.clk.p.add(i))) + rb(rb(*self.brd.p.add(i)) * flag));
            }
        }
        self.sync(t, E_ROWS);
        let smear = *self.scal.add(3 * self.l);
        let prev = self.scratch[t].p;
        let mut k = t;
        while k < tt {
            let s = &self.seq[self.tseq[k]];
            let j = k - s.off;
            let ek = self.e_.p.add(k * d);
            if j > 0 {
                std::ptr::copy_nonoverlapping(ek.sub(d), prev, d);
            } else {
                // the row before: a leaf's last ancestor slot, else the cache's last row
                let before = if s.leaf() && s.plen > 0 {
                    Some(s.se.add(self.paths[s.path + s.plen - 1] * d))
                } else if s.n0 > 0 {
                    Some(s.e.add((s.n0 - 1) * d))
                } else {
                    None
                };
                match before {
                    Some(b) => {
                        for dd in 0..d {
                            *prev.add(dd) = bf(*b.add(dd));
                        }
                    }
                    None => std::ptr::write_bytes(prev, 0, d),
                }
            }
            let c = rb(smear * rb(sigm(rb(dot::<S>(ek, self.smear_gate, 16)))));
            let xk = self.x.p.add(k * d);
            axpy_rb::<S>(xk, ek, c, prev, d);
            norm::<S>(xk, xk, d);
            std::ptr::copy_nonoverlapping(xk, self.x0.p.add(k * d), d);
            let e2 = self.embed2.add(*self.ids.add(k) as usize * d);
            let x2 = self.x02.p.add(k * d);
            for dd in 0..d {
                *x2.add(dd) = bf(*e2.add(dd));
            }
            norm::<S>(x2, x2, d);
            let ce = if s.leaf() { s.se.add(s.dest * d) } else { s.e.add((s.n0 + j) * d) };
            for dd in 0..d {
                *ce.add(dd) = tobf(*ek.add(dd));
            }
            k += nt;
        }
        self.sync(t, E_SMEAR);
    }

    /// token k's residual stream from src into dst: skip connection, x0 blend; hk = norm(dst); its gates
    #[inline(always)]
    unsafe fn resid<S: Simd>(&self, i: usize, k: usize, src: *const f32, dst: *mut f32, hk: *mut f32, gk: *mut f32) {
        let ly = &self.layers[i];
        let (d, vl) = (self.d, S::VL);
        let a = self.x0.p.add(k * d) as *const f32;
        let b = self.x02.p.add(k * d) as *const f32;
        let mut src = src;
        if ly.skout >= 0 {
            let j = ly.skout as usize;
            let gs = sigm(*self.scal.add(3 * self.l + 2 + j)) * 2.0;
            let gg = rb(gs * rb(sigm(rb(dot::<S>(a, self.skip_gate[j], 16)))));
            axpy_rb::<S>(dst, src, gg, self.skip[2 - j].p.add(k * d), d);
            src = dst;
        }
        let (c0, c1, lam) = (*self.x0l.add(2 * i), *self.x0l.add(2 * i + 1), *self.scal.add(i));
        let mut dd = 0;
        if i == 0 {
            let c = (lam as f64 + c0 as f64) as f32;
            while dd + vl <= d {
                S::st(dst.add(dd), S::rb(S::add(S::rb(S::mul(S::set(c), S::ld(src.add(dd)))), S::rb(S::mul(S::set(c1), S::ld(b.add(dd)))))));
                dd += vl;
            }
            while dd < d {
                *dst.add(dd) = rb(rb(c * *src.add(dd)) + rb(c1 * *b.add(dd)));
                dd += 1;
            }
        } else {
            while dd + vl <= d {
                let s = S::rb(S::add(S::rb(S::mul(S::set(c0), S::ld(a.add(dd)))), S::rb(S::mul(S::set(c1), S::ld(b.add(dd))))));
                S::st(dst.add(dd), S::rb(S::fma(S::set(lam), S::ld(src.add(dd)), s)));
                dd += vl;
            }
            while dd < d {
                *dst.add(dd) = rb(S::fmaf(lam, *src.add(dd), rb(rb(c0 * *a.add(dd)) + rb(c1 * *b.add(dd)))));
                dd += 1;
            }
        }
        norm::<S>(dst, hk, d);
        for r in 0..ly.g {
            *gk.add(r) = rb(sigm(rb(dot::<S>(hk, ly.gates.add(r * 16), 16))));
        }
    }

    /// token k, head hh: q and k normed and rotated, v (+ value embedding); k and v into the cache (a leaf's:
    /// into its slot)
    #[inline(always)]
    unsafe fn rotary<S: Simd>(&self, i: usize, k: usize, hh: usize, gk: *const f32) {
        let ly = &self.layers[i];
        let s = &self.seq[self.tseq[k]];
        let (d, h, hd) = (self.d, self.h, self.hd);
        let p = s.pos(k);
        let half = hd / 2;
        let row = self.qkv.p.add(k * 3 * d) as *const f32;
        let mut qn = [0f32; 256];
        let mut kk = [0f32; 256];
        norm::<S>(row.add(hh * hd), qn.as_mut_ptr(), hd);
        norm::<S>(row.add((h + hh) * hd), kk.as_mut_ptr(), hd);
        let cs = self.cosv.add(p * half);
        let sn = self.sinv.add(p * half);
        let qo = self.q.p.add(k * d + hh * hd);
        let (kb, vb, stride, at) = if s.leaf() { (s.sk, s.sv, s.scap, s.dest) } else { (s.k, s.v, s.cap, p) };
        let kc = kb.add(((i * h + hh) * stride + at) * hd);
        let vc = vb.add(((i * h + hh) * stride + at) * hd);
        for c in 0..half {
            let (co, si) = (bf(*cs.add(c)), bf(*sn.add(c)));
            *qo.add(c) = rb(rb(qn[c] * co) + rb(qn[c + half] * si));
            *qo.add(c + half) = rb(rb(qn[c + half] * co) - rb(qn[c] * si));
            *kc.add(c) = tobf(rb(rb(kk[c] * co) + rb(kk[c + half] * si)));
            *kc.add(c + half) = tobf(rb(rb(kk[c + half] * co) - rb(kk[c] * si)));
        }
        let v = row.add((2 * h + hh) * hd);
        if ly.ve >= 0 {
            let g2 = rb(2.0 * *gk.add(h + hh));
            let vr = self.ve[ly.ve as usize].add(*self.ids.add(k) as usize * d + hh * hd);
            for c in 0..hd {
                *vc.add(c) = tobf(rb(*v.add(c) + rb(g2 * bf(*vr.add(c)))));
            }
        } else {
            for c in 0..hd {
                *vc.add(c) = tobf(*v.add(c));
            }
        }
    }

    /// a leaf's row j after the cache's: its ancestors' slots in order, then its own
    #[inline(always)]
    fn slot(&self, s: &Seq, j: usize) -> usize {
        if j < s.plen {
            self.paths[s.path + j]
        } else {
            s.dest
        }
    }

    /// token k's attention in head hh over its game's cache (a leaf's: the cache's rows, then its slots), times
    /// the output gate; sc: scores
    #[inline(always)]
    unsafe fn attend<S: Simd>(&self, i: usize, k: usize, hh: usize, gk: *const f32, sc: *mut f32) {
        let s = &self.seq[self.tseq[k]];
        let (d, h, hd, vl) = (self.d, self.h, self.hd, S::VL);
        let kp = s.k.add((i * h + hh) * s.cap * hd) as *const u16;
        let vp = s.v.add((i * h + hh) * s.cap * hd) as *const u16;
        let p = s.pos(k);
        let qv = self.q.p.add(k * d + hh * hd) as *const f32;
        let nc = hd / vl;
        let mut qr = [S::zero(); 32];
        for (c, q) in qr.iter_mut().enumerate().take(nc) {
            *q = S::ld(qv.add(c * vl));
        }
        let n = p + 1;
        let nk = if s.leaf() { s.n0 } else { n }; // rows read from the cache
        let mut mx = f32::NEG_INFINITY;
        for r in 0..nk {
            let v = score::<S>(&qr, nc, kp.add(r * hd)) * self.scale;
            *sc.add(r) = v;
            if mx < v {
                mx = v;
            }
        }
        if s.leaf() {
            let sk = s.sk.add((i * h + hh) * s.scap * hd) as *const u16;
            for r in nk..n {
                let v = score::<S>(&qr, nc, sk.add(self.slot(s, r - nk) * hd)) * self.scale;
                *sc.add(r) = v;
                if mx < v {
                    mx = v;
                }
            }
        }
        let mut r = 0;
        let mut vs = S::zero();
        while r + vl <= n {
            let ex = S::exp(S::sub(S::ld(sc.add(r)), S::set(mx)));
            S::st(sc.add(r), ex);
            vs = S::add(vs, ex);
            r += vl;
        }
        let mut sum = S::sum(vs);
        while r < n {
            let ex = (*sc.add(r) - mx).exp();
            *sc.add(r) = ex;
            sum += ex;
            r += 1;
        }
        let mut o = [S::zero(); 32];
        for r in 0..nk {
            let pr = S::set(*sc.add(r));
            for c in 0..nc {
                o[c] = S::fma(pr, S::ld_bf16(vp.add(r * hd + c * vl)), o[c]);
            }
        }
        if s.leaf() {
            let sv = s.sv.add((i * h + hh) * s.scap * hd) as *const u16;
            for r in nk..n {
                let pr = S::set(*sc.add(r));
                let row = sv.add(self.slot(s, r - nk) * hd);
                for c in 0..nc {
                    o[c] = S::fma(pr, S::ld_bf16(row.add(c * vl)), o[c]);
                }
            }
        }
        let yo = self.y.p.add(k * d + hh * hd);
        let og = *gk.add(hh);
        let mut ov = [0f32; 256];
        for c in 0..nc {
            S::st(ov.as_mut_ptr().add(c * vl), o[c]);
        }
        for c in 0..hd {
            *yo.add(c) = rb(rb(ov[c] / sum) * og);
        }
    }

    #[inline(always)]
    unsafe fn blockstep<S: Simd>(&self, i: usize, t: usize, nt: usize) {
        let ly = &self.layers[i];
        let (tt, d, h, vl) = (self.t, self.d, self.h, S::VL);
        let g = ly.g;
        // solo: one or two tokens; every thread does the per-token steps itself, into its own buffers,
        // rather than wait at a barrier for one thread
        let solo = tt <= 2;
        let dense = !ly.fc.is_null();
        let pv = self.priv_[t].get();
        let xs = if solo { (*pv).x.as_mut_ptr() } else { self.x.p };
        let gs = if solo { (*pv).g.as_mut_ptr() } else { self.g.p };
        let hs = if solo { (*pv).h.as_mut_ptr() } else { self.hb.p };
        let (k0, kstep) = if solo { (0, 1) } else { (t, nt) };
        let mut k = k0;
        while k < tt {
            self.resid::<S>(i, k, self.x.p.add(k * d), xs.add(k * d), hs.add(k * d), gs.add(k * g));
            k += kstep;
        }
        if !solo {
            self.sync(t, B_NORM);
        }
        let hp: *const *const f32 = if solo { (*pv).ph.as_ptr() } else { self.ph.as_ptr() };
        let cq = chunk_rows(3 * d, nt, 64);
        while let Some(c) = self.next_chunk(t, (3 * d).div_ceil(cq)) {
            mm::<S>(&ly.qkv, d, cq * c, cq.min(3 * d - cq * c), hp, tt, self.qkv.p.add(cq * c), 3 * d);
        }
        self.sync(t, B_QKV);
        let sc = self.scratch[t].p;
        if tt == self.s {
            // one new token a game: rotary and attention in one pass per (game, head)
            let mut u = t;
            while u < tt * h {
                self.rotary::<S>(i, u / h, u % h, gs.add((u / h) * g));
                self.attend::<S>(i, u / h, u % h, gs.add((u / h) * g), sc);
                u += nt;
            }
        } else {
            let mut u = t;
            while u < tt * h {
                self.rotary::<S>(i, u / h, u % h, gs.add((u / h) * g));
                u += nt;
            }
            self.sync(t, B_ROTARY);
            let nu = self.units.len() / 3; // (game, head, query block)
            let mut u = t;
            while u < nu * h {
                for k in self.units[3 * (u / h) + 1]..self.units[3 * (u / h) + 2] {
                    self.attend::<S>(i, k, u % h, gs.add(k * g), sc);
                }
                u += nt;
            }
        }
        self.sync(t, B_ATTN);
        let co = chunk_rows(d, nt, 32);
        while let Some(c) = self.next_chunk(t, d.div_ceil(co)) {
            let lo = co * c;
            let n = co.min(d - lo);
            mm::<S>(&ly.o, d, lo, n, self.py.as_ptr(), tt, self.tmp.p.add(lo), d);
            for k in 0..tt {
                let a = k * d + lo;
                let mut j = 0;
                while j + vl <= n {
                    S::st(self.x.p.add(a + j), S::rb(S::add(S::ld(xs.add(a + j)), S::ld(self.tmp.p.add(a + j)))));
                    j += vl;
                }
                while j < n {
                    *self.x.p.add(a + j) = rb(*xs.add(a + j) + *self.tmp.p.add(a + j));
                    j += 1;
                }
            }
        }
        self.sync(t, B_O);
        let mut k = k0;
        while k < tt {
            let hk = hs.add(k * d);
            norm::<S>(self.x.p.add(k * d), hk, d);
            if !dense {
                let fk = if solo { (*pv).hf.as_mut_ptr() } else { self.hf.p }.add(k * d);
                for dd in 0..d {
                    *fk.add(dd) = *hk.add(dd) - *ly.mu.add(dd);
                }
            }
            if solo && t == 0 {
                std::ptr::copy_nonoverlapping(hk, self.hb.p.add(k * d), d); // for the experts
            }
            k += kstep;
        }
        if !solo || dense {
            self.sync(t, B_NORM2); // solo: the router reads each thread's own copy
        }
        self.ffn::<S>(t, nt, dense, i, solo);
    }

    /// top-k of one token and its gates. Exact ties in the biased scores go to the lower expert id
    unsafe fn route(&self, ly: &Layer, k: usize, r: &mut Routing) {
        let (e, topk, keep) = (self.e, self.topk, self.keep);
        let s = self.rs.p.add(k * e) as *const f32;
        let mut best = [0f32; 64];
        let mut bi = [0usize; 64];
        let mut n = 0;
        for ex in 0..e {
            let v = *s.add(ex) + *ly.bias.add(ex);
            if n == topk && v <= best[n - 1] {
                continue;
            }
            let mut j = if n < topk {
                n += 1;
                n - 1
            } else {
                n - 1
            };
            while j > 0 && best[j - 1] < v {
                best[j] = best[j - 1];
                bi[j] = bi[j - 1];
                j -= 1;
            }
            best[j] = v;
            bi[j] = ex;
        }
        let mut sum = 0f32;
        for &b in &bi[..topk] {
            sum += *s.add(b);
        }
        let f = (topk as f64).sqrt() as f32 / if sum < self.floor { self.floor } else { sum };
        for j in 0..keep {
            r.idx[k * keep + j] = bi[j];
            r.gate[k * keep + j] = rb(*s.add(bi[j]) * f);
        }
    }

    /// tokens grouped by expert (each expert's weights read once), and the up chunks: 32 pairs (64 rows) a
    /// chunk, the shared expert's (most tokens) first
    unsafe fn group(&self, i: usize, r: &mut Routing, dense: bool) {
        let ly = &self.layers[i];
        let (tt, d, keep, eh, sh, dh, e, q8) = (self.t, self.d, self.keep, self.eh, self.sh, self.dh, self.e, self.q8);
        r.segs.clear();
        if dense {
            r.segs.push(Seg { a: ly.fc, pairs: dh, ntok: tt, ldo: sh.max(dh), x: self.ph.as_ptr(), out: self.shid.p });
        } else {
            r.cnt.fill(0);
            let na = tt * keep;
            for a in 0..na {
                r.cnt[r.idx[a]] += 1;
            }
            let mut s = 0;
            for ex in 0..e {
                r.start[ex] = s;
                r.at[ex] = s;
                s += r.cnt[ex];
            }
            for a in 0..na {
                let ex = r.idx[a];
                let sl = r.at[ex];
                r.at[ex] += 1;
                r.stok[sl] = a / keep;
                r.sgate[sl] = r.gate[a];
                r.ptok[sl] = self.hb.p.add((a / keep) * d);
            }
            r.segs.push(Seg { a: ly.sup, pairs: sh, ntok: tt, ldo: sh.max(dh), x: self.ph.as_ptr(), out: self.shid.p });
            let es = 2 * eh * d * if q8 { 1 } else { 2 };
            r.active.clear();
            for ex in 0..e {
                if r.cnt[ex] > 0 {
                    r.active.push(ex);
                    let a = Mat { w: ly.up.add(ex * es), s: if q8 { ly.ups.add(ex * 2 * eh) } else { std::ptr::null() }, kind: q8 as u8 };
                    r.segs.push(Seg { a, pairs: eh, ntok: r.cnt[ex], ldo: eh, x: r.ptok.as_ptr().add(r.start[ex]), out: self.hid.p.add(r.start[ex] * eh) });
                }
            }
        }
        r.upch.clear();
        for (j, seg) in r.segs.iter().enumerate() {
            let mut p = 0;
            while p < seg.pairs {
                r.upch.push(Chunk { seg: j, p0: p, n: 32.min(seg.pairs - p) });
                p += 32;
            }
        }
    }

    #[inline(always)]
    unsafe fn ffn<S: Simd>(&self, t: usize, nt: usize, dense: bool, i: usize, solo: bool) {
        let ly = &self.layers[i];
        let (tt, d, e, eh, sh, dh, q8, vl) = (self.t, self.d, self.e, self.eh, self.sh, self.dh, self.q8, S::VL);
        // solo: each thread routes and groups the token itself (identical copies): no barrier
        let pv = self.priv_[t].get();
        let rp: *mut Routing = if solo { std::ptr::addr_of_mut!((*pv).r) } else { self.shared.get() };
        if !dense {
            let fp: *const *const f32 = if solo { (*pv).phf.as_ptr() } else { self.phf.as_ptr() };
            let cr = chunk_rows(e, nt, 16);
            while let Some(c) = self.next_chunk(t, e.div_ceil(cr)) {
                let lo = cr * c;
                let n = cr.min(e - lo);
                mm::<S>(&Mat { w: ly.router as *const u8, s: std::ptr::null(), kind: 2 }, d, lo, n, fp, tt, self.rs.p.add(lo), e);
                for k in 0..tt {
                    for ex in lo..lo + n {
                        let p = self.rs.p.add(k * e + ex);
                        *p = sigm(*p);
                    }
                }
            }
            self.sync(t, B_ROUTER);
            let (k0, kstep) = if solo { (0, 1) } else { (t, nt) };
            let mut k = k0;
            while k < tt {
                self.route(ly, k, &mut *rp);
                k += kstep;
            }
            if !solo {
                self.sync(t, B_TOPK);
            }
        }
        if solo {
            self.group(i, &mut *rp, dense);
        } else {
            if t == 0 {
                self.group(i, &mut *rp, dense);
            }
            self.sync(t, B_GROUP);
        }
        let r: &Routing = &*rp;
        // up (gate and value halves), SwiGLU
        let ta = self.scratch[t].p;
        while let Some(c) = self.next_chunk(t, r.upch.len()) {
            let u = &r.upch[c];
            let s = &r.segs[u.seg];
            let a = ta;
            let b = ta.add(s.ntok * u.n);
            mm::<S>(&s.a, d, u.p0, u.n, s.x, s.ntok, a, u.n);
            mm::<S>(&s.a, d, s.pairs + u.p0, u.n, s.x, s.ntok, b, u.n);
            for m in 0..s.ntok {
                silu_mul::<S>(a.add(m * u.n), b.add(m * u.n), s.out.add(m * s.ldo + u.p0), u.n);
            }
        }
        self.sync(t, B_UP);
        // down: a share of the output features a chunk (about three a thread: each expert's rows of a chunk
        // are then a longer run), summed over the tokens' experts
        let tb = self.scratch[t].p;
        let dr = 16.max(chunk_rows(d, nt, 128));
        while let Some(c) = self.next_chunk(t, d.div_ceil(dr)) {
            let lo = dr * c;
            let n = dr.min(d - lo);
            if dense {
                mm::<S>(&ly.proj, dh, lo, n, self.pshid.as_ptr(), tt, tb, n);
                for k in 0..tt {
                    add_rb::<S>(self.x.p.add(k * d + lo), tb.add(k * n), n);
                }
            } else {
                for k in 0..tt {
                    std::ptr::write_bytes(self.acc.p.add(k * d + lo), 0, n);
                }
                let bytes = if q8 { 1 } else { 2 };
                let es = d * eh * bytes;
                let rs = n * eh * bytes;
                let act = &r.active;
                ahead(ly.sdown.w.add(lo * sh * bytes), n * sh * bytes);
                if let Some(&e0) = act.first() {
                    ahead(ly.down.add(e0 * es + lo * (rs / n)), rs);
                }
                for (a, &ex) in act.iter().enumerate() {
                    if let Some(&e1) = act.get(a + 1) {
                        ahead(ly.down.add(e1 * es + lo * (rs / n)), rs);
                    }
                    let am = Mat { w: ly.down.add(ex * es), s: if q8 { ly.downs.add(ex * d) } else { std::ptr::null() }, kind: q8 as u8 };
                    let s0 = r.start[ex];
                    mm::<S>(&am, eh, lo, n, self.phid.as_ptr().add(s0), r.cnt[ex], tb, n);
                    for m in 0..r.cnt[ex] {
                        let gt = r.sgate[s0 + m];
                        let ak = self.acc.p.add(r.stok[s0 + m] * d + lo);
                        let bm = tb.add(m * n) as *const f32;
                        let mut j = 0;
                        while j + vl <= n {
                            S::st(ak.add(j), S::fma(S::set(gt), S::ld(bm.add(j)), S::ld(ak.add(j))));
                            j += vl;
                        }
                        while j < n {
                            *ak.add(j) = S::fmaf(gt, *bm.add(j), *ak.add(j));
                            j += 1;
                        }
                    }
                }
                mm::<S>(&ly.sdown, sh, lo, n, self.pshid.as_ptr(), tt, tb, n);
                for k in 0..tt {
                    let xa = self.x.p.add(k * d + lo);
                    let aa = self.acc.p.add(k * d + lo) as *const f32;
                    let bb = tb.add(k * n) as *const f32;
                    let mut j = 0;
                    while j + vl <= n {
                        S::st(xa.add(j), S::rb(S::add(S::ld(xa.add(j)), S::rb(S::add(S::rb(S::ld(aa.add(j))), S::ld(bb.add(j)))))));
                        j += vl;
                    }
                    while j < n {
                        *xa.add(j) = rb(*xa.add(j) + rb(rb(*aa.add(j)) + *bb.add(j)));
                        j += 1;
                    }
                }
            }
            for k in 0..tt {
                let a = k * d + lo;
                if ly.skin >= 0 {
                    std::ptr::copy_nonoverlapping(self.x.p.add(a), self.skip[ly.skin as usize].p.add(a), n);
                }
                if i as i64 == self.backout_layer {
                    std::ptr::copy_nonoverlapping(self.x.p.add(a), self.bko.p.add(a), n);
                }
            }
        }
        self.sync(t, B_DOWN);
    }

    #[inline(always)]
    unsafe fn head<S: Simd>(&self, t: usize, nt: usize) {
        let (ss, d, v) = (self.s, self.d, self.v);
        let b = *self.scal.add(3 * self.l + 1);
        let mut s = t;
        while s < ss {
            let k = self.last[s];
            let xs = self.xf.p.add(s * d);
            for dd in 0..d {
                *xs.add(dd) = rb(*self.x.p.add(k * d + dd) - rb(b * *self.bko.p.add(k * d + dd)));
            }
            norm::<S>(xs, xs, d);
            s += nt;
        }
        self.sync(t, H_NORM);
        let cv = chunk_rows(v, nt, 64);
        while let Some(c) = self.next_chunk(t, v.div_ceil(cv)) {
            let lo = cv * c;
            let hi = v.min(lo + cv);
            mm::<S>(&Mat { w: self.lm_head as *const u8, s: std::ptr::null(), kind: 0 }, d, lo, hi - lo, self.pxf.as_ptr(), ss, self.z.p.add(lo), v);
            for s in 0..ss {
                for vv in lo..hi {
                    *self.out.add(s * v + vv) = 23.0 * sigm((*self.z.p.add(s * v + vv) + 5.0) / 7.5);
                }
            }
        }
        self.sync(t, H_HEAD);
    }
}

#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn run_avx512(e: &Engine, t: usize) {
    e.run::<Avx512>(t)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn run_avx2(e: &Engine, t: usize) {
    e.run::<Avx2>(t)
}
