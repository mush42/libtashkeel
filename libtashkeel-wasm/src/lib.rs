mod utils;

use libtashkeel_base::{LibtashkeelResult, DynamicInferenceEngine};
use once_cell::sync::Lazy;
use wasm_bindgen::prelude::*;


static ENGINE: Lazy<LibtashkeelResult<DynamicInferenceEngine>> = Lazy::new(|| {
    utils::set_panic_hook();
    #[cfg(target_arch = "wasm32")]
    ort::wasm::initialize();

    libtashkeel_base::create_inference_engine(None)
});


#[wasm_bindgen]
pub fn do_tashkeel(
    text: &str,
    taskeen_threshold: Option<f32>,
    preprocessed: Option<bool>,
) -> String {
    let engine = ENGINE.as_ref().unwrap();
    let preprocessed = preprocessed.unwrap_or_default();
    libtashkeel_base::do_tashkeel(engine, text, taskeen_threshold, preprocessed).unwrap()
}

