//! `allie_fast.Server`: one OS thread owning an `Engine`, serving step requests from Python threads (games'
//! moves, the handle-driven searches) and from the Rust searchers' native loops. What is queued is merged into
//! one step of up to `max_items` items; a short gather window after the first request lets the games served
//! by the last step submit their next leaves, so concurrent games' work lands in one step. Callers block for
//! their reply with the GIL released.
//!
//! Requests of plain items only (games' moves, a view's prefill) go first, in a step of their own, so a move
//! waits for the step in flight but never behind other games' leaves (`moves_first`).
//!
//! Pointer contract: a request's ids, feats and boards are copied at submission; its caches, slot buffers and
//! `out` are written in place during the step, so the caller keeps them alive and untouched until its reply
//! (the blocking call returns), as the Python side does for `Engine.step`.

#![allow(clippy::too_many_arguments)]

use std::collections::VecDeque;
use std::ops::Range;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::model::Engine;

/// A batch of items as `Engine::step` takes them: meta per item (n0 len cap off scap plen poff dest, off and
/// poff relative to this request), caches 6 per item, the tokens' ids [t], feats [t, 3], boards [t, 68], and
/// `out` [items, V] for the logits.
pub struct Request {
    pub ids: Vec<i64>,
    pub feats: Vec<f32>,
    pub boards: Vec<u8>,
    pub meta: Vec<i64>,
    pub caches: Vec<usize>,
    pub paths: Vec<i64>,
    pub out: *mut f32,
}

unsafe impl Send for Request {}

impl Request {
    pub fn items(&self) -> usize {
        self.meta.len() / 8
    }

    /// Plain items only: a game's move or prefill, no search leaves.
    pub fn plain(&self) -> bool {
        self.meta.chunks(8).all(|q| q[4] == 0)
    }

    /// The spans the server slices by fit (the engine checks the rest): items of at least one token at
    /// consecutive offsets covering the tokens, path items' slots within the paths; in checked arithmetic.
    pub fn valid(&self) -> bool {
        let mut off = 0i64;
        for q in self.meta.chunks(8) {
            if q.len() < 8 || q[1] < 1 || q[3] != off || q[5] < 0 || q[6] < 0 {
                return false;
            }
            if q[4] > 0 && q[6].checked_add(q[5]).is_none_or(|e| e > self.paths.len() as i64) {
                return false;
            }
            match off.checked_add(q[1]) {
                Some(x) => off = x,
                None => return false,
            }
        }
        off == self.ids.len() as i64 && self.caches.len() == 6 * self.items()
    }
}

struct Reply {
    done: Mutex<Option<i32>>,
    cv: Condvar,
}

impl Reply {
    fn set(&self, e: i32) {
        *self.done.lock().unwrap() = Some(e);
        self.cv.notify_all();
    }

    fn wait(&self) -> i32 {
        let mut g = self.done.lock().unwrap();
        while g.is_none() {
            g = self.cv.wait(g).unwrap();
        }
        g.unwrap()
    }
}

struct Pending {
    req: Request,
    reply: Arc<Reply>,
}

#[derive(Default)]
struct State {
    queue: VecDeque<Pending>,
    paused: usize,
    stop: bool,
    /// requests in the last step: the gather window waits for as many
    expected: usize,
}

/// max_items: the most items merged into a step; min_items: a batch this large runs at once; gather: the
/// window after a step's first request, waited while fewer requests than the last step's have arrived;
/// chunk > 0: every request runs alone, in steps of at most `chunk` items (the equality gate's batching);
/// moves_first: queued plain-only requests run first, in a step of their own.
#[derive(Clone, Copy)]
pub struct Policy {
    pub max_items: usize,
    pub min_items: usize,
    pub gather: Duration,
    pub chunk: usize,
    pub moves_first: bool,
}

#[derive(Default, Clone)]
pub struct Stats {
    pub steps: u64,
    pub items: u64,
    pub tokens: u64,
    pub requests: u64,
    pub widest: usize,
    /// steps whose gather window waited, the seconds waited, the requests that arrived during a wait
    pub waited: u64,
    pub wait_s: f64,
    pub joined: u64,
    /// items of every step since the last `sizes()`
    pub sizes: Vec<u32>,
}

pub struct Inner {
    engine: Mutex<Engine>,
    state: Mutex<State>,
    cv: Condvar,
    policy: Mutex<Policy>,
    stats: Mutex<Stats>,
    /// L, H, hd, D, V
    pub dims: [usize; 5],
    pub threads: usize,
    pub isa: &'static str,
}

/// Items `range` of a request, placed in a merged step.
type Part<'a> = (&'a Request, Range<usize>);

/// The arrays of one step over `parts`: ids, feats, boards, meta (offsets rebased), caches, paths.
pub fn merge(parts: &[Part]) -> (Vec<i64>, Vec<f32>, Vec<u8>, Vec<i64>, Vec<usize>, Vec<i64>) {
    let (mut ids, mut feats, mut boards) = (Vec::new(), Vec::new(), Vec::new());
    let (mut meta, mut caches, mut paths) = (Vec::new(), Vec::new(), Vec::new());
    for (r, range) in parts {
        if range.is_empty() {
            continue;
        }
        let m = |i: usize| &r.meta[8 * i..8 * i + 8];
        let (t0, last) = (m(range.start)[3] as usize, m(range.end - 1));
        let t1 = (last[3] + last[1]) as usize;
        let base = ids.len() as i64 - t0 as i64;
        ids.extend_from_slice(&r.ids[t0..t1]);
        feats.extend_from_slice(&r.feats[3 * t0..3 * t1]);
        boards.extend_from_slice(&r.boards[68 * t0..68 * t1]);
        for i in range.clone() {
            let q = m(i);
            let poff = if q[4] > 0 {
                let p = paths.len() as i64;
                paths.extend_from_slice(&r.paths[q[6] as usize..(q[6] + q[5]) as usize]);
                p
            } else {
                0
            };
            meta.extend([q[0], q[1], q[2], q[3] + base, q[4], q[5], poff, q[7]]);
            caches.extend_from_slice(&r.caches[6 * i..6 * i + 6]);
        }
    }
    (ids, feats, boards, meta, caches, paths)
}

impl Inner {
    fn new(engine: Engine, policy: Policy) -> Inner {
        let dims = [engine.l, engine.h, engine.hd, engine.d, engine.v];
        let (threads, isa) = (engine.pool.n(), engine.isa.name());
        Inner { engine: Mutex::new(engine), state: Mutex::new(State::default()), cv: Condvar::new(), policy: Mutex::new(policy), stats: Mutex::new(Stats::default()), dims, threads, isa }
    }

    pub fn policy(&self) -> Policy {
        *self.policy.lock().unwrap()
    }

    /// Queue a request and block for its reply: the engine's error code (0: the logits are in `out`; 3 for
    /// spans that do not fit, refused here; 5 when the step panicked), or -1 when the server has stopped.
    pub fn submit(&self, req: Request) -> i32 {
        if !req.valid() {
            return 3;
        }
        let reply = Arc::new(Reply { done: Mutex::new(None), cv: Condvar::new() });
        {
            let mut st = self.state.lock().unwrap();
            if st.stop {
                return -1;
            }
            st.queue.push_back(Pending { req, reply: reply.clone() });
        }
        self.cv.notify_all();
        reply.wait()
    }

    /// The next batch: the first queued request, then what fits within the gather policy. None: stopped.
    fn take(&self) -> Option<Vec<Pending>> {
        let pol = self.policy();
        let mut st = self.state.lock().unwrap();
        loop {
            if st.stop {
                return None;
            }
            if st.paused == 0 && !st.queue.is_empty() {
                break;
            }
            st = self.cv.wait(st).unwrap();
        }
        self.engine.lock().unwrap_or_else(PoisonError::into_inner).pool.wake(); // the workers spin while the batch gathers
        if pol.moves_first && pol.chunk == 0 && st.queue.iter().any(|p| p.req.plain()) {
            let (mut batch, mut items, mut rest) = (Vec::new(), 0, VecDeque::new());
            while let Some(p) = st.queue.pop_front() {
                if p.req.plain() && (batch.is_empty() || items + p.req.items() <= pol.max_items) {
                    items += p.req.items();
                    batch.push(p);
                } else {
                    rest.push_back(p);
                }
            }
            st.queue = rest;
            return Some(batch);
        }
        let mut batch = vec![st.queue.pop_front().unwrap()];
        let mut items = batch[0].req.items();
        if pol.chunk == 0 && items < pol.max_items {
            let (start, before) = (Instant::now(), batch.len());
            let mut waited = false;
            loop {
                while let Some(f) = st.queue.front() {
                    if items + f.req.items() > pol.max_items {
                        break;
                    }
                    items += f.req.items();
                    batch.push(st.queue.pop_front().unwrap());
                }
                if items >= pol.min_items || items >= pol.max_items || batch.len() >= st.expected || st.paused > 0 || st.stop {
                    break;
                }
                let gone = start.elapsed();
                if gone >= pol.gather {
                    break;
                }
                waited = true;
                st = self.cv.wait_timeout(st, pol.gather - gone).unwrap().0;
            }
            if waited {
                let mut s = self.stats.lock().unwrap();
                s.waited += 1;
                s.wait_s += start.elapsed().as_secs_f64();
                s.joined += (batch.len() - before) as u64;
            }
        }
        st.expected = batch.len();
        Some(batch)
    }

    /// One step over `parts`, each request's rows copied to its `out`; the engine's error code, 5 if it panicked
    /// (its callers get the error; the server keeps serving).
    fn step(&self, parts: &[Part], out: &mut Vec<f32>) -> i32 {
        catch_unwind(AssertUnwindSafe(|| self.run(parts, out))).unwrap_or(5)
    }

    fn run(&self, parts: &[Part], out: &mut Vec<f32>) -> i32 {
        let v = self.dims[4];
        let (ids, feats, boards, meta, caches, paths) = merge(parts);
        let (t, s) = (ids.len(), meta.len() / 8);
        if out.len() < s * v {
            out.resize(s * v, 0.0);
        }
        let err = self.engine.lock().unwrap_or_else(PoisonError::into_inner).step(t, s, ids.as_ptr(), feats.as_ptr(), boards.as_ptr(), &meta, &caches, &paths, out.as_mut_ptr());
        if err == 0 {
            let mut row = 0;
            for (r, range) in parts {
                let n = range.len();
                unsafe { std::ptr::copy_nonoverlapping(out.as_ptr().add(row * v), r.out.add(range.start * v), n * v) };
                row += n;
            }
            let mut st = self.stats.lock().unwrap();
            st.steps += 1;
            st.items += s as u64;
            st.tokens += t as u64;
            st.requests += parts.len() as u64;
            st.widest = st.widest.max(s);
            if st.sizes.len() < 1 << 20 {
                st.sizes.push(s as u32);
            }
        }
        err
    }

    fn serve(&self) {
        let mut out = Vec::new();
        while let Some(batch) = self.take() {
            let chunk = self.policy().chunk;
            if chunk > 0 {
                for p in &batch {
                    let n = p.req.items();
                    let err = (0..n).step_by(chunk).map(|a| self.step(&[(&p.req, a..n.min(a + chunk))], &mut out)).find(|&e| e != 0);
                    p.reply.set(err.unwrap_or(0));
                }
                continue;
            }
            let parts: Vec<Part> = batch.iter().map(|p| (&p.req, 0..p.req.items())).collect();
            let err = self.step(&parts, &mut out);
            for (p, part) in batch.iter().zip(parts) {
                // a bad request fails only its own caller: the rest run alone
                p.reply.set(if err != 0 && batch.len() > 1 { self.step(&[part], &mut out) } else { err });
            }
        }
    }
}

#[pyclass(name = "Server", module = "allie_fast")]
pub struct PyServer {
    pub inner: Arc<Inner>,
    thread: Option<JoinHandle<()>>,
    pid: libc::pid_t,
}

#[pymethods]
impl PyServer {
    /// `Engine`'s arguments, then the gather policy: max_items (the most items in a step), min_items (a batch
    /// this large runs at once), gather (seconds waited after a step's first request for the other requests
    /// of the last step), chunk (> 0: every request alone, in steps of at most chunk items), moves_first (plain-only
    /// requests first, in a step of their own).
    #[new]
    #[pyo3(signature = (cfg, scale, floor, globals, layers, cpus=None, groups=None, spin=0.002, max_items=128, min_items=48, gather=3e-4, chunk=0, moves_first=true))]
    fn new(cfg: Vec<i64>, scale: f64, floor: f64, globals: Vec<usize>, layers: Vec<usize>, cpus: Option<Vec<i32>>, groups: Option<Vec<i32>>, spin: f64, max_items: usize, min_items: usize, gather: f64, chunk: usize, moves_first: bool) -> PyResult<Self> {
        let engine = Engine::new(&cfg, (scale, floor), &globals, &layers, cpus, groups, spin).map_err(PyValueError::new_err)?;
        let policy = Policy { max_items: max_items.max(1), min_items, gather: Duration::from_secs_f64(gather.max(0.0)), chunk, moves_first };
        let inner = Arc::new(Inner::new(engine, policy));
        let me = inner.clone();
        let thread = std::thread::Builder::new().name("allie-server".into()).spawn(move || me.serve())?;
        Ok(PyServer { inner, thread: Some(thread), pid: unsafe { libc::getpid() } })
    }

    /// `Engine.step`'s call, queued and merged with the other callers' and run on the server's thread; blocks
    /// for the reply with the GIL released. ids, feats and boards are copied here; the caches and `out` are
    /// written during the step.
    fn step(&self, py: Python<'_>, t: usize, s: usize, ids: usize, feats: usize, boards: usize, meta: Vec<i64>, caches: Vec<usize>, paths: Vec<i64>, out: usize) -> i32 {
        if meta.len() < 8 * s || caches.len() < 6 * s {
            return 3;
        }
        let (mut meta, mut caches) = (meta, caches);
        meta.truncate(8 * s);
        caches.truncate(6 * s);
        let n = meta.chunks(8).try_fold(0i64, |a, q| a.checked_add(q[1]));
        if n != Some(t as i64) || t > 1 << 24 {
            return 3; // the tokens to copy are not the items' spans
        }
        let req = unsafe {
            Request {
                ids: std::slice::from_raw_parts(ids as *const i64, t).to_vec(),
                feats: std::slice::from_raw_parts(feats as *const f32, 3 * t).to_vec(),
                boards: std::slice::from_raw_parts(boards as *const u8, 68 * t).to_vec(),
                meta,
                caches,
                paths,
                out: out as *mut f32,
            }
        };
        let inner = self.inner.clone();
        py.allow_threads(move || inner.submit(req))
    }

    /// Seconds spent in each phase since the last call (fast.py's PHASES order); on: keep profiling. Waits for
    /// a running step.
    #[pyo3(signature = (on=true))]
    fn profile(&self, py: Python<'_>, on: bool) -> Vec<f64> {
        let inner = self.inner.clone();
        py.allow_threads(move || inner.engine.lock().unwrap_or_else(PoisonError::into_inner).profile(on).to_vec())
    }

    /// Sleeping workers spin again, ready for a step.
    fn wake(&self, py: Python<'_>) {
        let inner = self.inner.clone();
        py.allow_threads(move || inner.engine.lock().unwrap_or_else(PoisonError::into_inner).pool.wake())
    }

    /// on: no step starts until every pause is lifted; requests queue up and merge.
    fn pause(&self, on: bool) {
        {
            let mut st = self.inner.state.lock().unwrap();
            st.paused = if on { st.paused + 1 } else { st.paused.saturating_sub(1) };
        }
        self.inner.cv.notify_all();
    }

    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads
    }

    #[getter]
    fn isa(&self) -> &'static str {
        self.inner.isa
    }

    /// (L, H, head_dim, D): the slot buffers' layout.
    fn dims(&self) -> (usize, usize, usize, usize) {
        let d = self.inner.dims;
        (d[0], d[1], d[2], d[3])
    }

    /// steps, items, tokens, requests, widest (most items in a step), waited (steps whose gather window
    /// waited), wait_s, joined (requests that arrived during a wait), queued (requests waiting now).
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats.lock().unwrap().clone();
        let d = PyDict::new_bound(py);
        d.set_item("steps", s.steps)?;
        d.set_item("items", s.items)?;
        d.set_item("tokens", s.tokens)?;
        d.set_item("requests", s.requests)?;
        d.set_item("widest", s.widest)?;
        d.set_item("waited", s.waited)?;
        d.set_item("wait_s", s.wait_s)?;
        d.set_item("joined", s.joined)?;
        d.set_item("queued", self.inner.state.lock().unwrap().queue.len())?;
        Ok(d)
    }

    /// The items of every step since the last call.
    fn sizes(&self) -> Vec<u32> {
        std::mem::take(&mut self.inner.stats.lock().unwrap().sizes)
    }

    #[getter]
    fn get_max_items(&self) -> usize {
        self.inner.policy().max_items
    }

    #[setter]
    fn set_max_items(&self, n: usize) {
        self.inner.policy.lock().unwrap().max_items = n.max(1);
    }

    #[getter]
    fn get_min_items(&self) -> usize {
        self.inner.policy().min_items
    }

    #[setter]
    fn set_min_items(&self, n: usize) {
        self.inner.policy.lock().unwrap().min_items = n;
    }

    #[getter]
    fn get_gather(&self) -> f64 {
        self.inner.policy().gather.as_secs_f64()
    }

    #[setter]
    fn set_gather(&self, s: f64) {
        self.inner.policy.lock().unwrap().gather = Duration::from_secs_f64(s.max(0.0));
    }

    #[getter]
    fn get_chunk(&self) -> usize {
        self.inner.policy().chunk
    }

    #[setter]
    fn set_chunk(&self, n: usize) {
        self.inner.policy.lock().unwrap().chunk = n;
    }

    #[getter]
    fn get_moves_first(&self) -> bool {
        self.inner.policy().moves_first
    }

    #[setter]
    fn set_moves_first(&self, on: bool) {
        self.inner.policy.lock().unwrap().moves_first = on;
    }
}

impl Drop for PyServer {
    fn drop(&mut self) {
        if unsafe { libc::getpid() } != self.pid {
            // a fork's copy: the thread exists only in the parent (the pool's drop knows the same)
            self.thread.take().map(std::mem::forget);
            return;
        }
        {
            let mut st = self.inner.state.lock().unwrap();
            st.stop = true;
            for p in st.queue.drain(..) {
                p.reply.set(-1);
            }
        }
        self.inner.cv.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyServer>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(items: &[(usize, usize, bool)]) -> Request {
        // items: (n0, len, path item); ids run 100, 101, ...; paths 7, 8 for path items
        let (mut ids, mut meta, mut caches, mut paths, mut off) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0i64);
        for (k, &(n0, len, leaf)) in items.iter().enumerate() {
            let (scap, plen, poff) = if leaf { (16, 2, paths.len() as i64) } else { (0, 0, 0) };
            if leaf {
                paths.extend([7 + k as i64, 8 + k as i64]);
            }
            meta.extend([n0 as i64, len as i64, 64, off, scap, plen, poff, k as i64 + 1]);
            caches.extend([10 * k + 1, 10 * k + 2, 10 * k + 3, 0, 0, 0]);
            ids.extend((0..len).map(|j| 100 + off + j as i64));
            off += len as i64;
        }
        let t = off as usize;
        Request { ids, feats: (0..3 * t).map(|x| x as f32).collect(), boards: (0..68 * t).map(|x| x as u8).collect(), meta, caches, paths, out: std::ptr::null_mut() }
    }

    #[test]
    fn plain_requests() {
        assert!(request(&[(5, 3, false), (2, 1, false)]).plain());
        assert!(!request(&[(5, 3, false), (9, 1, true)]).plain());
    }

    #[test]
    fn invalid_spans() {
        let ok = request(&[(5, 3, false), (9, 1, true)]);
        assert!(ok.valid());
        for (i, v) in [(1, i64::MAX), (1, 0), (3, 7), (5, -1), (6, i64::MAX), (5, 3)] {
            let mut r = request(&[(5, 3, false), (9, 1, true)]);
            r.meta[8 + i] = v; // the path item's
            if i == 1 && v == i64::MAX {
                r.meta[1] = 1;
            }
            assert!(!r.valid(), "meta[{}] = {v}", 8 + i);
        }
        let mut r = request(&[(5, 3, false)]);
        r.ids.pop();
        assert!(!r.valid());
    }

    #[test]
    fn merges_and_rebases() {
        let a = request(&[(5, 3, false), (9, 1, true)]);
        let b = request(&[(2, 1, true), (4, 2, false)]);
        let (ids, feats, boards, meta, caches, paths) = merge(&[(&a, 0..2), (&b, 0..2)]);
        assert_eq!(ids, [100, 101, 102, 103, 100, 101, 102]);
        assert_eq!((feats.len(), boards.len()), (21, 7 * 68));
        assert_eq!(&meta[..8], [5, 3, 64, 0, 0, 0, 0, 1]);
        assert_eq!(&meta[8..16], [9, 1, 64, 3, 16, 2, 0, 2]);
        assert_eq!(&meta[16..24], [2, 1, 64, 4, 16, 2, 2, 1]);
        assert_eq!(&meta[24..], [4, 2, 64, 5, 0, 0, 0, 2]);
        assert_eq!(paths, [8, 9, 7, 8]);
        assert_eq!(caches.len(), 24);
        // a chunk of one request: its second item alone, offsets from zero
        let (ids, _, _, meta, _, paths) = merge(&[(&a, 1..2)]);
        assert_eq!((ids, &meta[..], paths), (vec![103], &[9, 1, 64, 0, 16, 2, 0, 2][..], vec![8, 9]));
    }
}
