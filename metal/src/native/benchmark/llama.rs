use std::{
    env,
    path::{Path, PathBuf},
};

use foundation::model::{BackendTarget, ModelFamily, ModelManifest, Quantization};
use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{LoadedModel, Result, greedy_token};

const PROMPT: [u32; 36] = [
    128_000, 128_006, 9125, 128_007, 271, 38_766, 1303, 33_025, 2696, 25, 6790, 220, 2366, 18, 198,
    15_724, 2696, 25, 220, 806, 10_263, 220, 2366, 21, 271, 128_009, 128_006, 882, 128_007, 271,
    13_347, 128_009, 128_006, 78_191, 128_007, 271,
];
const EXPECTED: [u32; 8] = [4438, 649, 358, 7945, 499, 3432, 30, 128_009];

#[test]
#[ignore = "loads Llama 3.2; set MIRMIR_LLAMA_MODEL"]
fn preserves_tied_embedding_tokens_with_piecewise_rope() -> Result<()> {
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
    env::var_os("MIRMIR_LLAMA_MODEL")
        .map(PathBuf::from)
        .ok_or_else(|| super::Error::Benchmark("set MIRMIR_LLAMA_MODEL".into()))
}

fn manifest(path: &Path) -> ModelManifest {
    ModelManifest {
        id: "tied-embedding-rope-regression".into(),
        family: ModelFamily::Unknown,
        path: path.to_string_lossy().into_owned(),
        tokenizer_path: None,
        context_len: 131_072,
        quantization: Quantization::Int4,
        preferred_backends: vec![BackendTarget::Metal],
    }
}
