use std::{env, fs, path::PathBuf};

use libmir::{Error, RuntimeConfig};

pub struct Config {
    pub model: PathBuf,
    pub prompt_tokens: usize,
    pub chunk_tokens: usize,
    pub warmup_runs: usize,
    pub prompt: String,
    requests: Option<PathBuf>,
}

impl Config {
    pub fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let model = env::args_os()
            .nth(1)
            .or_else(|| env::var_os("MODEL"))
            .map(PathBuf::from)
            .ok_or(Error::MissingEnvironment("MODEL or the first argument"))?;
        let prompt_tokens = argument(2, 2_048)?;
        let chunk_tokens = argument(3, 2_096)?;
        let warmup_runs = argument(4, 1)?;
        let prompt = env::var_os("MIRMIR_PREFILL_PROFILE_PROMPT")
            .map(fs::read_to_string)
            .transpose()?
            .unwrap_or_else(|| "Explain continuous batching in an LLM inference server.".into());
        let requests = env::var_os("MIRMIR_PREFILL_PROFILE_REQUESTS").map(PathBuf::from);
        if chunk_tokens == 0 {
            return Err("chunk token count must be positive".into());
        }
        Ok(Self {
            model,
            prompt_tokens,
            chunk_tokens,
            warmup_runs,
            prompt,
            requests,
        })
    }

    #[cfg_attr(not(feature = "cuda"), allow(clippy::unused_self))]
    pub fn runtime(&self) -> RuntimeConfig {
        let runtime = RuntimeConfig {
            automatic_kv_cache: true,
            memory: libmir::MemoryRuntimeConfig {
                reserve_percent: Some(1),
                reserve_bytes: Some(512 * 1_024 * 1_024),
            },
            ..RuntimeConfig::default()
        };
        #[cfg(feature = "cuda")]
        let mut runtime = runtime;
        #[cfg(feature = "cuda")]
        {
            runtime.cuda.model_session.prefill_chunk_tokens = self.chunk_tokens;
            runtime.cuda.tuning.cache_directory =
                env::var_os("MIRMIR_CUDA_TUNING_CACHE").map(PathBuf::from);
        }
        runtime
    }

    pub fn write_requests(&self, tokens: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
        let Some(path) = &self.requests else {
            return Ok(());
        };
        let mut warmup = tokens.to_vec();
        let length = warmup.len();
        warmup.rotate_left(1 % length);
        let payload = serde_json::json!({
            "measured": completion_request(tokens),
            "warmup": completion_request(&warmup),
        });
        fs::write(path, serde_json::to_vec_pretty(&payload)?)?;
        Ok(())
    }
}

fn completion_request(tokens: &[u32]) -> serde_json::Value {
    serde_json::json!({
        "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "prompt": tokens,
        "max_tokens": 1,
        "temperature": 0.0,
    })
}

fn argument(index: usize, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    env::args().nth(index).map_or(Ok(default), |value| Ok(value.parse()?))
}
