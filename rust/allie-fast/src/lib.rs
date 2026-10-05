//! Allie's CPU engine: the int8 forward pass, KV caches with shared prefixes, and the tree searches that
//! allie.lichess runs on it. Python sees one extension module, `allie_fast`.

use pyo3::prelude::*;

pub mod chess;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn allie_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    chess::register(m)?;
    Ok(())
}
