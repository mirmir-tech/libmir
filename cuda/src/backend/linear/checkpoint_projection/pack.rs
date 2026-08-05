use super::CheckpointProjectionWeight;
use crate::{CudaBackend, Result};

impl CheckpointProjectionWeight {
    pub(in crate::backend) fn pack_direct_fp8<const N: usize>(
        backend: &CudaBackend,
        weights: [&Self; N],
    ) -> Result<Option<Self>> {
        let direct = weights.map(|weight| match weight {
            Self::DirectFp8(weight) => Some(weight),
            _ => None,
        });
        let Some(direct) = direct.into_iter().collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };
        let Ok(direct): std::result::Result<[&crate::DirectFp8CheckpointWeight; N], _> =
            direct.try_into()
        else {
            return Ok(None);
        };
        Ok(crate::DirectFp8CheckpointWeight::pack::<N>(backend, direct)?.map(Self::DirectFp8))
    }
}
