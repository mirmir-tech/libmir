use std::{
    collections::BTreeSet,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
};

use libmir::BackendTarget;

use super::TestResult;

mod bitsandbytes;
mod descriptor;
mod float8;
mod gptq;
mod modelopt;
mod mxfp4;
mod mxfp8;
mod nvfp4;
mod reference;
mod types;

pub use bitsandbytes::BitsAndBytes4BitReference;
pub use descriptor::{
    validate_affine_descriptor, validate_awq_descriptor, validate_bitsandbytes_descriptor,
    validate_descriptor, validate_float8_descriptor, validate_gptq_descriptor,
    validate_modelopt_descriptor, validate_mxfp4_descriptor, validate_mxfp8_descriptor,
    validate_nvfp4_descriptor, validate_packed_int4_descriptor, validate_packed_int8_descriptor,
};
pub use float8::Float8Reference;
pub use modelopt::validate_modelopt_mixed_for;
pub use mxfp4::MxFp4Reference;
pub use mxfp8::MxFp8Reference;
pub use nvfp4::NvFp4Reference;
pub use types::*;

impl Catalog {
    pub fn parse(source: &str) -> TestResult<Self> {
        Ok(toml::from_str(source)?)
    }

    pub fn validate(&self) -> TestResult<()> {
        require(self.schema == 1, "dense checkpoint catalog schema must be 1")?;
        let families = self.fixtures.iter().map(|fixture| fixture.family).collect::<BTreeSet<_>>();
        require(
            families
                == BTreeSet::from([
                    Family::Dense,
                    Family::DenseAndRouted,
                    Family::SharedRouted,
                    Family::ClampedRouted,
                ]),
            "dense checkpoint catalog must contain exactly all four semantic families",
        )?;
        let model_envs =
            self.fixtures.iter().map(|fixture| &fixture.model_env).collect::<BTreeSet<_>>();
        let reference_envs = self
            .fixtures
            .iter()
            .map(|fixture| &fixture.reference_env)
            .collect::<BTreeSet<_>>();
        require(model_envs.len() == 4, "model environment variables must be unique")?;
        require(reference_envs.len() == 4, "reference environment variables must be unique")
    }
}

impl Family {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::DenseAndRouted => "dense_and_routed",
            Self::SharedRouted => "shared_routed",
            Self::ClampedRouted => "clamped_routed",
        }
    }
}

pub fn active_target() -> BackendTarget {
    if cfg!(all(feature = "metal", target_os = "macos")) {
        BackendTarget::Metal
    } else if cfg!(feature = "cuda") {
        BackendTarget::Cuda
    } else {
        BackendTarget::Metal
    }
}

pub fn required_path(name: &str) -> TestResult<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| validation_error(format!("required environment variable is unset: {name}")))
}

pub fn load_reference(path: &Path) -> TestResult<Reference> {
    Reference::parse(&fs::read_to_string(path)?)
}

pub fn require(condition: bool, message: impl Into<String>) -> TestResult<()> {
    if condition {
        Ok(())
    } else {
        Err(validation_error(message))
    }
}

pub fn validation_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::other(message.into()))
}
