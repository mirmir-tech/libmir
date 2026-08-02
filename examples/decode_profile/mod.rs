use std::{env, path::PathBuf};

#[cfg(feature = "cuda")]
use libmir::{
    CudaDenseVectorPolicy, CudaDenseVendorPolicy, CudaDenseWeightPolicy, CudaKernelAdmission,
    CudaNumericalPolicy, DenseRole,
};
use libmir::{Error, KvCacheDType, RuntimeConfig};

pub struct Config {
    pub model: PathBuf,
    pub prompt_tokens: usize,
    pub warmup_steps: usize,
    pub measured_steps: usize,
    pub prefix_fill_sessions: usize,
    pub prefix_fill_cooldown_seconds: u64,
    pub clear_allocator_after_fill: bool,
    pub kv_cache_dtype: Option<KvCacheDType>,
    #[cfg(feature = "cuda")]
    dense: DenseMode,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum DenseMode {
    Stable,
    Tuned,
    Vendor,
    BlockFp8,
    Fp8Int4,
    BlockFp8Down,
    Fp8Int4Down,
    BlockFp8GateUp,
    Fp8Int4GateUp,
    Throughput,
}

impl Config {
    pub fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let model = env::args_os()
            .nth(1)
            .or_else(|| env::var_os("MODEL"))
            .map(PathBuf::from)
            .ok_or(Error::MissingEnvironment("MODEL or the first argument"))?;
        let prompt_tokens = argument(2, 128)?;
        let warmup_steps = argument(3, 8)?;
        let measured_steps = argument(4, 32)?;
        let prefix_fill_sessions = environment_usize("MIRMIR_DECODE_PROFILE_PREFIX_FILL", 0)?;
        let prefix_fill_cooldown_seconds =
            environment_u64("MIRMIR_DECODE_PROFILE_COOLDOWN_SECONDS", 0)?;
        let clear_allocator_after_fill =
            enabled("MIRMIR_DECODE_PROFILE_CLEAR_ALLOCATOR_AFTER_FILL");
        let kv_cache_dtype = match env::var("MIRMIR_KV_CACHE_DTYPE") {
            Ok(value) => Some(value.parse()?),
            Err(env::VarError::NotPresent) => None,
            Err(error) => return Err(error.into()),
        };
        if prompt_tokens == 0 || measured_steps == 0 {
            return Err("prompt tokens and measured steps must be positive".into());
        }
        Ok(Self {
            model,
            prompt_tokens,
            warmup_steps,
            measured_steps,
            prefix_fill_sessions,
            prefix_fill_cooldown_seconds,
            clear_allocator_after_fill,
            kv_cache_dtype,
            #[cfg(feature = "cuda")]
            dense: DenseMode::parse(5)?,
        })
    }

    pub fn configure(&self, runtime: &mut RuntimeConfig) {
        if let Some(dtype) = self.kv_cache_dtype {
            runtime.kv_cache.dtype = dtype;
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let _ = runtime;
        #[cfg(feature = "cuda")]
        self.configure_cuda(runtime);
        #[cfg(feature = "metal")]
        self.configure_metal(runtime);
    }

    #[cfg(feature = "cuda")]
    fn configure_cuda(&self, runtime: &mut RuntimeConfig) {
        runtime.cuda.tuning.cache_directory =
            env::var_os("MIRMIR_CUDA_TUNING_CACHE").map(PathBuf::from);
        if !matches!(self.dense, DenseMode::Stable) {
            runtime.cuda.planning.numerical = CudaNumericalPolicy::Throughput;
            runtime.cuda.planning.admission = CudaKernelAdmission::Experimental;
            match self.dense {
                DenseMode::Tuned => {
                    runtime.cuda.planning.dense_vectors = CudaDenseVectorPolicy::Tuned;
                },
                DenseMode::Vendor => {
                    runtime.cuda.planning.dense_vendor = CudaDenseVendorPolicy::Tuned;
                },
                DenseMode::BlockFp8 => {
                    runtime.cuda.planning.dense_weights =
                        CudaDenseWeightPolicy::BlockFp8Role(DenseRole::AttentionOutput);
                },
                DenseMode::Fp8Int4 => {
                    runtime.cuda.planning.dense_weights =
                        CudaDenseWeightPolicy::Fp8Int4Role(DenseRole::AttentionOutput);
                },
                DenseMode::BlockFp8Down => {
                    runtime.cuda.planning.dense_weights =
                        CudaDenseWeightPolicy::BlockFp8Role(DenseRole::DenseDown);
                },
                DenseMode::Fp8Int4Down => {
                    runtime.cuda.planning.dense_weights =
                        CudaDenseWeightPolicy::Fp8Int4Role(DenseRole::DenseDown);
                },
                DenseMode::BlockFp8GateUp => {
                    runtime.cuda.planning.dense_weights =
                        CudaDenseWeightPolicy::BlockFp8Role(DenseRole::DenseGateUp);
                },
                DenseMode::Fp8Int4GateUp => {
                    runtime.cuda.planning.dense_weights =
                        CudaDenseWeightPolicy::Fp8Int4Role(DenseRole::DenseGateUp);
                },
                DenseMode::Throughput => {
                    runtime.cuda.planning.dense_vectors = CudaDenseVectorPolicy::Tuned;
                    runtime.cuda.planning.dense_vendor = CudaDenseVendorPolicy::Tuned;
                    runtime.cuda.planning.dense_weights =
                        CudaDenseWeightPolicy::BlockFp8Role(DenseRole::DenseGateUp);
                },
                DenseMode::Stable => {},
            }
        }
    }

    #[cfg(feature = "metal")]
    #[allow(clippy::unused_self)]
    fn configure_metal(&self, runtime: &mut RuntimeConfig) {
        runtime.metal.diagnostics.profile_components = enabled("MIRMIR_METAL_PROFILE_COMPONENTS");
        runtime.metal.diagnostics.profile_layers = enabled("MIRMIR_METAL_PROFILE_LAYERS");
        runtime.metal.tuning.cache_directory =
            env::var_os("MIRMIR_METAL_TUNING_CACHE").map(PathBuf::from);
        if disabled("MIRMIR_METAL_FUSED_DENSE_GATE_UP") {
            runtime.metal.fusion.dense_gate_up = libmir::FeatureToggle::Disabled;
        }
    }

    #[cfg(feature = "cuda")]
    pub const fn dense_label(&self) -> &'static str {
        match self.dense {
            DenseMode::Stable => " dense=stable",
            DenseMode::Tuned => " dense_vectors=tuned",
            DenseMode::Vendor => " dense_vendor=tuned",
            DenseMode::BlockFp8 => " dense_weight=block-fp8",
            DenseMode::Fp8Int4 => " dense_weight=fp8-int4",
            DenseMode::BlockFp8Down => " dense_down_weight=block-fp8",
            DenseMode::Fp8Int4Down => " dense_down_weight=fp8-int4",
            DenseMode::BlockFp8GateUp => " dense_gate_up_weight=block-fp8",
            DenseMode::Fp8Int4GateUp => " dense_gate_up_weight=fp8-int4",
            DenseMode::Throughput => " dense=throughput",
        }
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::unused_self)]
    pub const fn dense_label(&self) -> &'static str {
        ""
    }
}

#[cfg(feature = "cuda")]
impl DenseMode {
    fn parse(index: usize) -> Result<Self, Box<dyn std::error::Error>> {
        match env::args().nth(index).as_deref().unwrap_or("stable") {
            "stable" => Ok(Self::Stable),
            "tuned" => Ok(Self::Tuned),
            "vendor" => Ok(Self::Vendor),
            "block-fp8" => Ok(Self::BlockFp8),
            "fp8-int4" => Ok(Self::Fp8Int4),
            "block-fp8-down" => Ok(Self::BlockFp8Down),
            "fp8-int4-down" => Ok(Self::Fp8Int4Down),
            "block-fp8-gate-up" => Ok(Self::BlockFp8GateUp),
            "fp8-int4-gate-up" => Ok(Self::Fp8Int4GateUp),
            "throughput" => Ok(Self::Throughput),
            _ => Err("unsupported CUDA dense policy".into()),
        }
    }
}

fn argument(index: usize, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    env::args().nth(index).map_or(Ok(default), |value| Ok(value.parse()?))
}

fn environment_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    env::var(name).map_or(Ok(default), |value| Ok(value.parse()?))
}

fn environment_u64(name: &str, default: u64) -> Result<u64, Box<dyn std::error::Error>> {
    env::var(name).map_or(Ok(default), |value| Ok(value.parse()?))
}

fn enabled(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true" | "TRUE" | "yes" | "YES"))
}

#[cfg(feature = "metal")]
fn disabled(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("0" | "false" | "FALSE" | "no" | "NO"))
}
