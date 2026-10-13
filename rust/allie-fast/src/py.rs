//! `allie_fast.place`: the NUMA placement of the weights' pages (fastrs.py keeps the tensors alive).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The pages of the given ranges (pointer, bytes) moved to NUMA nodes[0] (one node)
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
    m.add_function(wrap_pyfunction!(place, m)?)
}
