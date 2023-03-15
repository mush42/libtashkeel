use lazy_static::lazy_static;
use ndarray::Array2;
use ndarray_stats::QuantileExt;
use ort::{
    tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder,
};
use std::collections::HashMap;
use std::sync::Arc;

const MAX_INPUT_CHARS: usize = 315;
const MODEL_BYTES: &[u8; 10261536] = include_bytes!("data/model.ort");
const INPUT_VOCAB_TO_INT_STR: &str = include_str!("data/dictionary/input_vocab_to_int.txt");
const OUTPUT_INT_TO_VOCAB_STR: &str = include_str!("data/dictionary/output_int_to_vocab.txt");
const PAD_STR: &str = "<PAD>";
const INVALID_HARAKA: [&str; 2] = ["<UNK>", "ـ"];

lazy_static! {
    static ref HARAKAT_CHARS: Vec<char> =
        (1612..1619).map(|n| char::from_u32(n).unwrap()).collect();
}

lazy_static! {
    static ref INPUT_VOCAB_TO_INT: HashMap<String, u8> = INPUT_VOCAB_TO_INT_STR
        .lines()
        .map(|line| {
            let pair: Vec<&str> = line.split('|').collect();
            let vocab = pair[0].to_string();
            let vid: u8 = pair[1].parse().unwrap();
            (vocab, vid)
        })
        .collect();
    static ref UNK_INPUT_ID: u8 = *INPUT_VOCAB_TO_INT.get("<UNK>").unwrap();
}

lazy_static! {
    static ref OUTPUT_INT_TO_VOCAB: HashMap<usize, String> = OUTPUT_INT_TO_VOCAB_STR
        .lines()
        .map(|line| {
            let pair: Vec<&str> = line.split('|').collect();
            let vid: usize = pair[0].parse().unwrap();
            let vocab = pair[1].to_string();
            (vid, vocab)
        })
        .collect();
}

lazy_static! {
    static ref _ENVIRONMENT: Arc<ort::Environment> = Arc::new(
        Environment::builder()
            .with_name("libtashkeel")
            .with_execution_providers([ExecutionProvider::cpu()])
            .build()
            .unwrap()
    );
    pub static ref ORT_SESSION: ort::InMemorySession<'static> = SessionBuilder::new(&_ENVIRONMENT)
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_parallel_execution(true)
        .unwrap()
        .with_intra_threads(4)
        .unwrap()
        .with_memory_pattern(true)
        .unwrap()
        .with_model_from_memory(MODEL_BYTES)
        .unwrap();
}

pub fn do_tashkeel(text: String) -> String {
    let input_sent: Vec<char> = text
        .chars()
        .filter(|c| !HARAKAT_CHARS.contains(c))
        .collect();
    let mut input_ids: Vec<f32> = input_sent
        .iter()
        .map(|c| {
            INPUT_VOCAB_TO_INT
                .get(&c.to_string())
                .unwrap_or(&UNK_INPUT_ID)
        })
        .map(|id| *id as f32)
        .collect();
    input_ids.resize(MAX_INPUT_CHARS, 0.0);
    let input_array = Array2::<f32>::from_shape_vec((1, input_ids.len()), input_ids).unwrap();

    let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> = ORT_SESSION
        .run([InputTensor::from_array(input_array.into_dyn())])
        .unwrap();
    let logets: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();

    let mut predicted_herakats = logets
        .view()
        .rows()
        .into_iter()
        .map(|row| row.argmax().unwrap())
        .filter_map(|idx| {
            let prediction = OUTPUT_INT_TO_VOCAB.get(&idx).unwrap();
            if prediction == PAD_STR {
                None
            } else {
                Some(prediction.as_str())
            }
        })
        .collect::<Vec<&str>>();
    predicted_herakats.resize(input_sent.len(), "");
    combine_text_with_harakat(input_sent, predicted_herakats)
}

fn combine_text_with_harakat(input_sent: Vec<char>, output_sent: Vec<&str>) -> String {
    let mut text = String::new();
    for (character, haraka) in input_sent.iter().zip(output_sent.iter()) {
        if INVALID_HARAKA.contains(haraka) {
            text.push(*character);
        } else {
            text.push(*character);
            text.push_str(haraka);
        }
    }
    text
}

// ==============================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works_as_expected() {
        let text = "مرحبا";

        let expected = "مَرْحَبًا";
        let tashkeeled = do_tashkeel(text.to_string());

        assert_ne!(tashkeeled, text);
        assert_eq!(tashkeeled, expected);
    }
}
