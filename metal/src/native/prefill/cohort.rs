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
        super::validation::validate_requests(requests)?;
        #[cfg(test)]
        loaded.apply_history_pressure(crate::engine::memory_stats()?)?;
        let model_id = loaded.info.manifest.id.clone();
        let mut prefixes = HashMap::with_capacity(requests.len());
        let mut leased_groups = HashSet::new();
        let cached_groups_before = loaded.prefixes.group_count();
        let cached_entries_before = loaded.prefixes.entry_count();
        let cached_bytes_before = loaded.prefixes.resident_bytes();
        let mut hits = 0;
        for request in requests {
            let leased = loaded.prefixes.lease_longest(&model_id, &request.prompt_tokens)?;
            hits += usize::from(leased.is_some());
            if let Some(leased) = leased.as_ref() {
                leased_groups.insert(leased.memory_group);
            }
            prefixes.insert(
                request.session_id,
                leased.map_or(CohortPrefix::Miss, |leased| CohortPrefix::Hit(leased.restored)),
            );
        }
        let misses = requests.len().saturating_sub(hits);
        let evicted_leases = loaded.prefixes.evict_groups(&leased_groups);
        let evicted_misses = loaded.prefixes.reserve_batch_slots(misses);
        let initially_evicted = evicted_leases || evicted_misses;
        if initially_evicted {
            crate::engine::clear_memory_cache()?;
        }
        let pressure_evicted = hits > 0 && loaded.reclaim_unleased_prefixes_for_prefill()?;
        let evicted = initially_evicted || pressure_evicted;
        tracing::debug!(
            model = model_id,
            rows = requests.len(),
            prefix_hits = hits,
            prefix_misses = misses,
            leased_groups = leased_groups.len(),
            prefix_slots_reserved = misses.min(loaded.prefixes.capacity()),
            cached_groups_before,
            cached_entries_before,
            cached_bytes_before,
            cached_groups_after = loaded.prefixes.group_count(),
            evicted,
            "leased logical Metal prefill cohort"
        );
        Ok(Self {
            model_id,
            prefixes: Arc::new(Mutex::new(prefixes)),
        })
    }

    pub(in crate::native) fn take(
        &self,
        sessions: impl Iterator<Item = Uuid> + Clone,
    ) -> Result<Vec<CohortPrefix>> {
        let mut prefixes = self.prefixes.lock()?;
        let missing = || {
            Error::InvalidPrefillBatch("prefill session is absent from its logical cohort".into())
        };
        // Validate the whole wave before consuming any of its cohort leases.
        if sessions.clone().any(|session| !prefixes.contains_key(&session)) {
            return Err(missing());
        }
        sessions.map(|session| prefixes.remove(&session).ok_or_else(missing)).collect()
    }

    pub(in crate::native) fn discard(&self, sessions: &[Uuid]) -> Result<()> {
        self.prefixes.lock()?.retain(|session, _| !sessions.contains(session));
        Ok(())
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
        None => Ok(loaded
            .prefixes
            .lease_longest(&loaded.info.manifest.id, &request.prompt_tokens)?
            .map(|lease| lease.restored)),
    }
}
