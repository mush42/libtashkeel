use libtashkeel_base::do_tashkeel;
use pyo3::prelude::*;

/// Add diacritics to Arabic text using Shakala.
#[pyfunction]
fn tashkeel(text: String) -> PyResult<String> {
    Ok(do_tashkeel(text))
}

/// A wrapper for libtashkeel.
#[pymodule]
fn pylibtashkeel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tashkeel, m)?)?;
    Ok(())
}
