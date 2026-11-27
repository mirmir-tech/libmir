//! Writers sharing one profile: each persists under the profile's writer
//! lock, onto what the file holds then, so two tuners that measured
//! different shapes keep both decisions rather than the last snapshot.

use std::{collections::HashMap, fs::File, hash::Hash, path::Path};

use super::{StoredProfile, load, persist};

impl StoredProfile {
    /// Adds `other`'s decisions for keys this profile does not hold.
    fn fill(&mut self, other: Self) {
        fill(&mut self.gate_up, other.gate_up);
        fill(&mut self.attention, other.attention);
        fill(&mut self.batch_attention, other.batch_attention);
        fill(&mut self.experts, other.experts);
        fill(&mut self.routing, other.routing);
        fill(&mut self.decode_plans, other.decode_plans);
    }

    /// Takes `other`'s decisions, over this profile's for the same keys.
    fn overlay(&mut self, other: &Self) {
        self.gate_up.extend(other.gate_up.iter().map(|(key, value)| (*key, *value)));
        self.attention.extend(other.attention.iter().map(|(key, value)| (*key, *value)));
        self.batch_attention
            .extend(other.batch_attention.iter().map(|(key, value)| (*key, *value)));
        self.experts.extend(other.experts.iter().map(|(key, value)| (*key, *value)));
        self.routing.extend(other.routing.iter().map(|(key, value)| (*key, *value)));
        self.decode_plans
            .extend(other.decode_plans.iter().map(|(key, value)| (key.clone(), *value)));
    }
}

fn fill<K: Eq + Hash, V>(into: &mut HashMap<K, V>, from: HashMap<K, V>) {
    for (key, value) in from {
        into.entry(key).or_insert(value);
    }
}

/// Persists `known`, what a tuner holds, and `measured`, what it measured
/// itself, onto the profile at `path` as another writer may have left it:
/// a decision measured here replaces the file's, one only loaded fills in
/// what the file no longer has, and the file's others stay.
pub(in crate::engine::gate_up_tuning) fn persist_merged(
    path: &Path,
    known: StoredProfile,
    measured: &StoredProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new(".")))?;
    let lock = File::create(path.with_extension("lock"))?;
    lock.lock()?;
    let mut merged = load(path).unwrap_or_default();
    merged.fill(known);
    merged.overlay(measured);
    persist(
        path,
        &merged.gate_up,
        &merged.attention,
        &merged.batch_attention,
        &merged.experts,
        &merged.routing,
        &merged.decode_plans,
    )
    // The lock is released as `lock` closes.
}
