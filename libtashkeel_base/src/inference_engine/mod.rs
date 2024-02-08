use crate::{InferenceEngine, LibtashkeelResult};
use std::path::PathBuf;

pub struct DynamicInferenceEngine(Box<dyn InferenceEngine + Send + Sync>);

impl DynamicInferenceEngine {
    pub fn new(engine: Box<dyn InferenceEngine + Send + Sync>) -> Self {
        Self(engine)
    }
}

impl InferenceEngine for DynamicInferenceEngine {
    fn infer(
        &self,
        input_ids: Vec<i64>,
        diac_ids: Vec<i64>,
        seq_length: usize,
    ) -> LibtashkeelResult<(Vec<u8>, Vec<f32>)> {
        self.0.infer(input_ids, diac_ids, seq_length)
    }
}

#[cfg(feature = "ort")]
mod ort;

#[cfg(feature = "ort")]
pub fn create_inference_engine(
    model_path: Option<PathBuf>,
) -> LibtashkeelResult<DynamicInferenceEngine> {
    use self::ort::OrtEngine;

    log::info!("Built with `ORT` inference backend.");

    match model_path {
        Some(path) => {
            log::info!("Loading model from path: `{}`", path.display());
            let engine = OrtEngine::from_path(&path)?;
            Ok(DynamicInferenceEngine::new(Box::new(engine)))
        }
        None => {
            log::info!("Using bundled model");
            let engine = OrtEngine::with_bundled_model()?;
            Ok(DynamicInferenceEngine::new(Box::new(engine)))
        }
    }
}
