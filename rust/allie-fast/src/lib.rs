//! Allie's CPU engine: the int8 forward pass, KV caches with shared prefixes, and the tree searches that
//! allie.lichess runs on it. Python sees one extension module, `allie_fast`.

use pyo3::prelude::*;

pub mod chess;
pub mod kernels;
pub mod model;
pub mod pool;
pub mod py;
pub mod search;
pub mod server;
pub mod simd;

/// The Python-visible contract (the classes' arguments, the step's cfg and meta layouts): fastrs.py refuses an
/// engine whose INTERFACE is not its own. Bump both on any change to it.
pub const INTERFACE: u32 = 2;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn allie_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add("INTERFACE", INTERFACE)?;
    chess::register(m)?;
    py::register(m)?;
    search::register(m)?;
    server::register(m)?;
    Ok(())
}
