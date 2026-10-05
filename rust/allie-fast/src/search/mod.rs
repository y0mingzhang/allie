//! The tree searches allie.lichess runs: the coverage search (tree.cpp's forest, value.cpp's backup, tree.py's
//! node bookkeeping) as `allie_fast.Coverage`, the KL-regularized search (kl.py) as `allie_fast.KL`; both
//! driven by handles from Python or natively through the `Server` (`native`).

use pyo3::prelude::*;

pub mod backup;
pub mod clocks;
pub mod coverage;
pub mod kl;
pub mod native;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<coverage::Coverage>()?;
    m.add_class::<kl::Forest>()
}
