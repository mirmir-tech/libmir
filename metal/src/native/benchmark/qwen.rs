use std::{
    env,
    path::{Path, PathBuf},
};

use foundation::model::{BackendTarget, ModelManifest, Quantization};
use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{LoadedModel, Result, greedy_token};

const PROMPT: [u32; 9] = [151_644, 872, 198, 13_048, 151_645, 198, 151_644, 77_091, 198];
const EXPECTED: [u32; 16] = [
    151_667, 198, 32_313, 11, 279, 1_196, 1_053, 330, 13_048, 3_263, 2_938, 594, 264, 42_113, 13,
    358,
];

#[test]
#[ignore = "loads Qwen3-8B; set MIRMIR_QWEN_MODEL"]
fn preserves_qwen3_qk_norm_greedy_tokens() -> Result<()> {
    let mut ignored = |_event| {};
    let mut model = LoadedModel::load(&manifest(&model_path()?), &mut ignored)?;
    let session = Uuid::new_v4();
    let output = model.prefill(session, &PROMPT, SamplingLogits::None, &mut ignored)?;
    let mut token = greedy_token(&output.output)?;
    let mut generated = vec![token];
    for _ in 1..EXPECTED.len() {
        let output = model.decode(session, token, SamplingLogits::None)?;
        token = greedy_token(&output)?;
        generated.push(token);
    }
    assert_eq!(generated, EXPECTED);
    assert_eq!(model.session_cached_tokens(session)?, PROMPT.len() + EXPECTED.len() - 1);
    Ok(())
}

fn model_path() -> Result<PathBuf> {
    env::var_os("MIRMIR_QWEN_MODEL")
        .map(PathBuf::from)
        .ok_or_else(|| super::Error::Benchmark("set MIRMIR_QWEN_MODEL".into()))
}

fn manifest(path: &Path) -> ModelManifest {
    ModelManifest {
        id: "qwen3-qk-norm-regression".into(),
        path: path.to_string_lossy().into_owned(),
        tokenizer_path: None,
        context_len: 40_960,
        quantization: Quantization::Int4,
        preferred_backends: vec![BackendTarget::Metal],
    }
}
