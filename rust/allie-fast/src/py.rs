//! The Python face of the engine: `allie_fast.Engine`, taking exactly what fast.py's `Fast.__init__` passes
//! to the C++ `allie_new`, with `step` as `allie_step`. Raw pointers (as integers) are the contract: the
//! Python side keeps every tensor behind them alive for the engine's lifetime, as it does for the C++.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::missing_safety_doc)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::model::{Engine, NPHASE};

#[pyclass(name = "Engine", module = "allie_fast")]
pub struct PyEngine {
    inner: Box<Engine>,
}

#[pymethods]
impl PyEngine {
    /// cfg: L D H hd V nve E topk keep eh sh dh int8 ctx backout_layer threads 0, then per layer G ve
    /// skip_in skip_out; scale, floor: the attention scale and gate floor; globals: the 18 + nve global
    /// tensors' data pointers in fast.py's `glob` order; layers: 20 pointers a layer (fast.py's `lay`, 0
    /// for a missing matrix); cpus / groups: thread t's CPU and cache group; spin: seconds a worker spins
    /// before sleeping.
    #[new]
    #[pyo3(signature = (cfg, scale, floor, globals, layers, cpus=None, groups=None, spin=0.002))]
    #[allow(clippy::too_many_arguments)]
    fn new(cfg: Vec<i64>, scale: f64, floor: f64, globals: Vec<usize>, layers: Vec<usize>, cpus: Option<Vec<i32>>, groups: Option<Vec<i32>>, spin: f64) -> PyResult<Self> {
        let inner = Engine::new(&cfg, (scale, floor), &globals, &layers, cpus, groups, spin).map_err(PyValueError::new_err)?;
        Ok(PyEngine { inner: Box::new(inner) })
    }

    /// One step of T tokens over S items: ids (int64 [T]), feats (float32 [T, 3]), boards (uint8 [T, 68]) and
    /// out (float32 [S, V]) as data pointers; meta: per item n0 len cap off scap plen poff dest (scap 0: a plain
    /// item appending len tokens to its cache; else a path item, a search leaf whose one token attends to the
    /// cache's n0 rows then to slots paths[poff..poff + plen] of a slot buffer of capacity scap, its keys,
    /// values and embedding written to slot dest); caches: per item the cache's k, v, e pointers then the slot
    /// buffer's (0 for a plain item). Returns 0, or 1 for a token outside the vocabulary, 2 for a board state
    /// out of range, 3 for spans, paths or slots that do not fit. The GIL is released while the threads compute.
    #[allow(clippy::too_many_arguments)]
    fn step(&mut self, py: Python<'_>, t: usize, s: usize, ids: usize, feats: usize, boards: usize, meta: Vec<i64>, caches: Vec<usize>, paths: Vec<i64>, out: usize) -> i32 {
        let e = &mut *self.inner;
        py.allow_threads(move || e.step(t, s, ids as *const i64, feats as *const f32, boards as *const u8, &meta, &caches, &paths, out as *mut f32))
    }

    /// Seconds spent in each of the 17 phases since the last call (fast.py's PHASES order); on: keep
    /// profiling.
    #[pyo3(signature = (on=true))]
    fn profile(&mut self, on: bool) -> Vec<f64> {
        debug_assert_eq!(NPHASE, 17);
        self.inner.profile(on).to_vec()
    }

    /// Sleeping workers spin again, ready for a step.
    fn wake(&self) {
        self.inner.pool.wake()
    }

    #[getter]
    fn threads(&self) -> usize {
        self.inner.pool.n()
    }

    /// "avx512", "avx2" or "scalar": the kernels this engine runs.
    #[getter]
    fn isa(&self) -> &'static str {
        self.inner.isa.name()
    }
}

/// fast.py's allie_place: the pages of the given ranges (pointer, bytes) moved to NUMA nodes[0] (one node)
/// or interleaved over `nodes`; returns the number of ranges the kernel refused.
#[pyfunction]
fn place(ptrs: Vec<usize>, bytes: Vec<usize>, nodes: Vec<i32>) -> PyResult<i32> {
    if ptrs.len() != bytes.len() {
        return Err(PyValueError::new_err("ptrs and bytes differ in length"));
    }
    let mut mask = [0u64; 16];
    for &n in &nodes {
        if (0..1024).contains(&n) {
            mask[n as usize / 64] |= 1 << (n % 64);
        }
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let mode = if nodes.len() == 1 { libc::MPOL_BIND } else { libc::MPOL_INTERLEAVE };
    const MPOL_MF_MOVE: i32 = 2;
    let mut bad = 0;
    for (&p, &b) in ptrs.iter().zip(&bytes) {
        let (a, z) = (p / page * page, (p + b).div_ceil(page) * page);
        if z > a && unsafe { libc::syscall(libc::SYS_mbind, a, z - a, mode, mask.as_ptr(), 1025usize, MPOL_MF_MOVE) } != 0 {
            bad += 1;
        }
    }
    Ok(bad)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_function(wrap_pyfunction!(place, m)?)
}
