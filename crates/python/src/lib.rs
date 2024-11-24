use libtashkeel_core::{create_inference_engine, do_tashkeel, DynamicInferenceEngine};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;

static INFERENCE_ENGINE: GILOnceCell<DynamicInferenceEngine> = GILOnceCell::new();

/// Diacritize Arabic text.
#[pyfunction]
fn tashkeel(
    py: Python,
    text: String,
    taskeen_threshold: Option<f32>,
    preprocessed: Option<bool>,
) -> PyResult<String> {
    let preprocessed = preprocessed.unwrap_or_default();
    let engine = match INFERENCE_ENGINE.get(py) {
        Some(eng) => eng,
        None => {
            let error =
                PyRuntimeError::new_err("Failed to retrieve inference engine global instance");
            return Err(error);
        }
    };
    match do_tashkeel(engine, &text, taskeen_threshold, preprocessed) {
        Ok(diacritized_text) => Ok(diacritized_text),
        Err(e) => {
            let error =
                PyRuntimeError::new_err(format!("Failed to diacritize text. Caused by: {}", e));
            Err(error)
        }
    }
}

/// A Python wrapper for libtashkeel.
#[pymodule]
fn pylibtashkeel(py: Python, m: &PyModule) -> PyResult<()> {
    let engine = match create_inference_engine(None) {
        Ok(eng) => eng,
        Err(e) => {
            let error = PyRuntimeError::new_err(
                format!("Failed to create inference engine.\nCaused by: {}\nPlease make sure the system dependencies are properly installed", e)
            );
            return Err(error);
        }
    };
    INFERENCE_ENGINE.set(py, engine).ok();

    m.add_function(wrap_pyfunction!(tashkeel, m)?)?;
    Ok(())
}
