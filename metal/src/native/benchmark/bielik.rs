use std::{
    env,
    path::{Path, PathBuf},
};

use foundation::model::{BackendTarget, ModelManifest, Quantization};
use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{LoadedModel, Result, greedy_token};

const PROMPT: [u32; 22] = [
    1, 3, 1780, 336, 17, 31960, 695, 397, 31909, 18878, 3707, 20228, 284, 646, 31956, 4, 17, 3,
    31888, 3988, 19681, 17,
];
const EXPECTED: [u32; 16] =
    [31960, 695, 397, 31964, 450, 2897, 322, 24831, 851, 1347, 5148, 31888, 851, 4311, 284, 17541];

#[test]
#[ignore = "loads Bielik; set MIRMIR_BIELIK_MODEL"]
fn preserves_bielik_greedy_tokens_through_kv_decode() -> Result<()> {
    let path = bielik_path()?;
    let model = manifest(&path);
    let mut ignored = |_event| {};
    let mut model = LoadedModel::load(&model, &mut ignored)?;
    let session = Uuid::new_v4();
    let output = model.prefill(session, &PROMPT, SamplingLogits::None, None, &mut ignored)?;
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

fn bielik_path() -> Result<PathBuf> {
    env::var_os("MIRMIR_BIELIK_MODEL")
        .map(PathBuf::from)
        .ok_or_else(|| super::Error::Benchmark("set MIRMIR_BIELIK_MODEL".into()))
}

fn manifest(path: &Path) -> ModelManifest {
    ModelManifest {
        id: "bielik-greedy-regression".into(),
        // Native selection must depend on checkpoint structure, not a caller label.
        path: path.to_string_lossy().into_owned(),
        tokenizer_path: None,
        context_len: 32_768,
        quantization: Quantization::Int4,
        preferred_backends: vec![BackendTarget::Metal],
    }
}
