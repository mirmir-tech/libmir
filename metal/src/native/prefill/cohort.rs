use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use runtime::backend::PrefillRequest;
use uuid::Uuid;

use super::super::{
    error::{Error, Result},
    model::LoadedModel,
    prefix::RestoredPrefix,
};

#[derive(Clone)]
pub struct MetalPrefillCohort {
    model_id: String,
    prefixes: Arc<Mutex<HashMap<Uuid, CohortPrefix>>>,
}

pub(in crate::native) enum CohortPrefix {
    Hit(RestoredPrefix),
    Miss,
}

impl MetalPrefillCohort {
    pub(in crate::native) fn prepare(
        loaded: &mut LoadedModel,
        requests: &[PrefillRequest],
    ) -> Result<Self> {
        let model_id = loaded.info.manifest.id.clone();
        let mut prefixes = HashMap::with_capacity(requests.len());
        let mut leased_groups = HashSet::new();
        let mut hits = 0;
        for request in requests {
            let leased = loaded.prefixes.lease_longest(&model_id, &request.prompt_tokens)?;
            hits += usize::from(leased.is_some());
            if let Some(leased) = leased.as_ref() {
                leased_groups.insert(leased.memory_group);
            }
            if prefixes
                .insert(
                    request.session_id,
                    leased.map_or(CohortPrefix::Miss, |leased| CohortPrefix::Hit(leased.restored)),
                )
                .is_some()
            {
                return Err(Error::InvalidPrefillBatch(
                    "prefill cohort contains a duplicate session".into(),
                ));
            }
        }
        let misses = requests.len().saturating_sub(hits);
        let evicted_leases = loaded.prefixes.evict_groups(&leased_groups);
        let evicted_misses = loaded.prefixes.reserve_batch_slots(misses);
        let evicted = evicted_leases || evicted_misses;
        if evicted {
            crate::engine::clear_memory_cache()?;
        }
        tracing::debug!(
            model = model_id,
            rows = requests.len(),
            prefix_hits = hits,
            prefix_misses = misses,
            leased_groups = leased_groups.len(),
            prefix_slots_reserved = misses.min(loaded.prefixes.capacity()),
            evicted,
            "leased logical Metal prefill cohort"
        );
        Ok(Self {
            model_id,
            prefixes: Arc::new(Mutex::new(prefixes)),
        })
    }

    pub(in crate::native) fn take(&self, session: Uuid) -> Result<CohortPrefix> {
        self.prefixes.lock()?.remove(&session).ok_or_else(|| {
            Error::InvalidPrefillBatch("prefill session is absent from its logical cohort".into())
        })
    }

    pub(in crate::native) fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl CohortPrefix {
    pub(in crate::native) fn into_restored(self) -> Option<RestoredPrefix> {
        match self {
            Self::Hit(restored) => Some(restored),
            Self::Miss => None,
        }
    }
}

pub(in crate::native) fn restore_prefix(
    loaded: &mut LoadedModel,
    request: &PrefillRequest,
    leased: Option<CohortPrefix>,
) -> Result<Option<RestoredPrefix>> {
    match leased {
        Some(leased) => Ok(leased.into_restored()),
        None => loaded
            .prefixes
            .restore_longest(&loaded.info.manifest.id, &request.prompt_tokens),
    }
}
