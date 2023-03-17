use lazy_static::lazy_static;
use ndarray::Array;
use ndarray_stats::QuantileExt;
use ort::{
    tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder,
};
use std::collections::HashMap;
use std::sync::Arc;


const NUM_INPUT_CHARS: usize = 1024;
const MODEL_BYTES: &[u8; 10920272] = include_bytes!("data/model.ort");
const ARABIC_LETTERS_LIST_STR: &str = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي";
const DIACRITICS_LIST_STR: &str = "ًٌٍَُِّْ";
const CHARACTERS_MAPPING_STR: &str = include_str!("data/dictionary/CHARACTERS_MAPPING.txt");
const REV_CLASSES_MAPPING_STR: &str = include_str!("data/dictionary/REV_CLASSES_MAPPING.txt");


lazy_static! {
    static ref ARABIC_LETTERS_LIST: Vec<char> = ARABIC_LETTERS_LIST_STR.chars().collect();
    static ref DIACRITICS_LIST: Vec<char> = DIACRITICS_LIST_STR.chars().collect();
}

lazy_static! {
    static ref CHARACTERS_MAPPING: HashMap<String, f32> = CHARACTERS_MAPPING_STR
        .split("#")
        .map(|line| {
            let pair: Vec<&str> = line.split('|').collect();
            let vocab = pair[0].to_string();
            let vid: f32 = pair[1].parse().unwrap();
            (vocab, vid)
        })
        .collect();
    // Special values
    static ref SOS_INPUT_ID: f32 = *CHARACTERS_MAPPING.get("<SOS>").unwrap();
    static ref EOS_INPUT_ID: f32 = *CHARACTERS_MAPPING.get("<EOS>").unwrap();
    static ref UNK_INPUT_ID: f32 = *CHARACTERS_MAPPING.get("<UNK>").unwrap();
    static ref PAD_INPUT_ID: f32 = *CHARACTERS_MAPPING.get("<PAD>").unwrap();
}


lazy_static! {
    static ref REV_CLASSES_MAPPING: HashMap<usize, String> = REV_CLASSES_MAPPING_STR
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
        .with_inter_threads(4)
        .unwrap()
        .with_memory_pattern(true)
        .unwrap()
        .with_model_from_memory(MODEL_BYTES)
        .unwrap();
}

pub fn do_tashkeel(text: String) -> String {
    let input_sent: Vec<char> = text
        .chars()
        .filter(|c| !DIACRITICS_LIST.contains(c))
        .collect();
    let mut input_ids: Vec<f32> = input_sent
        .iter()
        .map(|c| {
            CHARACTERS_MAPPING
                .get(&c.to_string())
                .unwrap_or(&UNK_INPUT_ID)
        })
        .map(|id| id.clone())
        .collect();
    input_ids.insert(0, *SOS_INPUT_ID );
    input_ids.push(*EOS_INPUT_ID);
    input_ids.resize(NUM_INPUT_CHARS, *PAD_INPUT_ID);
    let input_array = Array::from_shape_vec((1, NUM_INPUT_CHARS), input_ids).unwrap();
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
            let prediction = REV_CLASSES_MAPPING.get(&idx).unwrap();
            if prediction.starts_with("<") {
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
        if !ARABIC_LETTERS_LIST.contains(character) {
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
