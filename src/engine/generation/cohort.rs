use runtime::backend::PrefillRequest;

use super::{Engine, EngineInner};
use crate::Result;

#[derive(Default)]
pub struct EnginePrefillCohort {
    #[cfg(feature = "metal")]
    pub(super) metal: Option<metal::MetalPrefillCohort>,
}

impl Engine {
    pub(crate) fn prepare_generation_prefill_cohort(
        &self,
        requests: &[PrefillRequest],
    ) -> Result<EnginePrefillCohort> {
        #[cfg(not(feature = "metal"))]
        let _ = requests;
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => Ok(EnginePrefillCohort::default()),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => Ok(EnginePrefillCohort {
                metal: Some(metal.prepare_prefill_cohort(requests)?),
            }),
        }
    }
}
