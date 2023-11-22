use crate::{InferenceEngine, LibtashkeelError, LibtashkeelResult};
use ndarray::{Array1, Array2};
use ort::{GraphOptimizationLevel, Session, SessionInputs, Value};
use std::path::Path;

impl From<ort::Error> for LibtashkeelError {
    fn from(other: ort::Error) -> Self {
        LibtashkeelError::InferenceError(format!(
            "Failed to run model using onnxruntime via ort. Caused by {}",
            other
        ))
    }
}

fn ort_session_run(
    session: &ort::Session,
    input_ids: Vec<i64>,
    seq_length: usize,
) -> LibtashkeelResult<(Vec<u8>, Vec<f32>)> {
    let input_ids =
        Array2::<i64>::from_shape_vec((1, seq_length), input_ids).unwrap();
    let input_length = Array1::<i64>::from_iter([seq_length as i64]);

    let (target_ids, logits): (Vec<u8>, Vec<f32>) = {
        let inputs = vec![
            Value::from_array(input_ids).unwrap(),
            Value::from_array(input_length).unwrap(),
        ];
        let outputs = session.run(SessionInputs::from(inputs.as_slice()))?;
        let target_ids = outputs[0].extract_tensor::<u8>()?;
        let logits = outputs[1].extract_tensor::<f32>()?;
        let target_ids_vec = Vec::from_iter(target_ids.view().iter().copied());
        let logits_vec = Vec::from_iter(logits.view().iter().copied());
        (target_ids_vec, logits_vec)
    };

    Ok((target_ids, logits))
}

const MODEL_BYTES: &[u8] = include_bytes!("../../data/ort/model.onnx");

pub struct OrtEngine(Session);

impl OrtEngine {
    pub fn from_bytes(model_bytes: &[u8]) -> LibtashkeelResult<OrtEngine> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_allocator(ort::AllocatorType::Arena)?
            .with_memory_pattern(true)?
            .with_parallel_execution(true)?
            .with_inter_threads(2)?
            .with_intra_threads(2)?
            .with_model_from_memory(model_bytes)?;

        Ok(Self(session))
    }
    pub fn from_path(model_path: impl AsRef<Path>) -> LibtashkeelResult<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_allocator(ort::AllocatorType::Arena)?
            .with_memory_pattern(true)?
            .with_parallel_execution(true)?
            .with_inter_threads(2)?
            .with_intra_threads(2)?
            .with_model_from_file(model_path)?;

        Ok(Self(session))
    }
    pub fn with_bundled_model() -> LibtashkeelResult<OrtEngine> {
        Self::from_bytes(MODEL_BYTES)
    }
}

impl InferenceEngine for OrtEngine {
    fn infer(
        &self,
        input_ids: Vec<i64>,
        seq_length: usize,
    ) -> LibtashkeelResult<(Vec<u8>, Vec<f32>)> {
        ort_session_run(&self.0, input_ids, seq_length)
    }
}
