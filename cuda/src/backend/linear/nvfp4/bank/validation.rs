use models::weights::TensorInfo;

use super::{NvFp4ExpertBankConfig, NvFp4ExpertSource};
use crate::{Error, Result};

pub(super) fn validate(
    config: NvFp4ExpertBankConfig,
    sources: &[NvFp4ExpertSource<'_>],
) -> Result<()> {
    if config.experts == 0
        || config.input_features == 0
        || config.output_features == 0
        || !config.input_features.is_multiple_of(16)
        || sources.len() != config.experts
    {
        return Err(Error::InvalidNvFp4("invalid expert bank geometry"));
    }
    let weight_shape = [config.output_features, config.input_features / 2];
    let scale_shape = [config.output_features, config.input_features / 16];
    for source in sources {
        tensor(source.weight, "U8", &weight_shape)?;
        tensor(source.weight_scale, "F8_E4M3", &scale_shape)?;
        scalar(source.weight_scale_2, "F32")?;
        scalar(source.input_scale, "F32")?;
    }
    Ok(())
}

fn scalar(info: &TensorInfo, dtype: &str) -> Result<()> {
    if info.dtype != dtype || !(info.shape.is_empty() || info.shape == [1]) {
        return Err(Error::InvalidNvFp4("expert bank scalar metadata mismatch"));
    }
    Ok(())
}

fn tensor(info: &TensorInfo, dtype: &str, shape: &[usize]) -> Result<()> {
    if info.dtype != dtype || info.shape != shape {
        return Err(Error::InvalidNvFp4("expert bank tensor metadata mismatch"));
    }
    Ok(())
}
