use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use super::{
    super::{
        attention_batch_tuning::{BatchAttentionExecution, BatchAttentionKey},
        attention_tuning::AttentionKey,
        decode_plan_tuning::{DecodePlan, DecodePlanKey},
        expert_tuning::{ExpertExecution, ExpertKey},
        kernels::PagedExecution,
        route_tuning::{RoutingExecution, RoutingKey},
    },
    GateUpExecution, GateUpKey,
};

const SCHEMA: u32 = 19;

#[derive(Debug, Deserialize, Serialize)]
struct ProfileFile {
    schema: u32,
    engine_version: String,
    host_architecture: String,
    gate_up: Vec<StoredEntry>,
    attention: Vec<StoredAttentionEntry>,
    batch_attention: Vec<StoredBatchAttentionEntry>,
    experts: Vec<StoredExpertEntry>,
    routing: Vec<StoredRoutingEntry>,
    decode_plans: Vec<StoredDecodePlan>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct StoredEntry {
    key: GateUpKey,
    execution: GateUpExecution,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct StoredAttentionEntry {
    key: AttentionKey,
    execution: PagedExecution,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct StoredBatchAttentionEntry {
    key: BatchAttentionKey,
    execution: BatchAttentionExecution,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct StoredExpertEntry {
    key: ExpertKey,
    execution: ExpertExecution,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct StoredRoutingEntry {
    key: RoutingKey,
    execution: RoutingExecution,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StoredDecodePlan {
    key: DecodePlanKey,
    plan: DecodePlan,
}

#[derive(Debug, Default)]
pub(super) struct StoredProfile {
    pub(super) gate_up: HashMap<GateUpKey, GateUpExecution>,
    pub(super) attention: HashMap<AttentionKey, PagedExecution>,
    pub(super) batch_attention: HashMap<BatchAttentionKey, BatchAttentionExecution>,
    pub(super) experts: HashMap<ExpertKey, ExpertExecution>,
    pub(super) routing: HashMap<RoutingKey, RoutingExecution>,
    pub(super) decode_plans: HashMap<DecodePlanKey, DecodePlan>,
}

pub(super) fn load(path: &Path) -> Option<StoredProfile> {
    let bytes = fs::read(path).ok()?;
    let file: ProfileFile = serde_json::from_slice(&bytes).ok()?;
    (file.schema == SCHEMA
        && file.engine_version == env!("CARGO_PKG_VERSION")
        && file.host_architecture == std::env::consts::ARCH)
        .then(|| StoredProfile {
            gate_up: file.gate_up.into_iter().map(|entry| (entry.key, entry.execution)).collect(),
            attention: file
                .attention
                .into_iter()
                .map(|entry| (entry.key, entry.execution))
                .collect(),
            batch_attention: file
                .batch_attention
                .into_iter()
                .map(|entry| (entry.key, entry.execution))
                .collect(),
            experts: file.experts.into_iter().map(|entry| (entry.key, entry.execution)).collect(),
            routing: file.routing.into_iter().map(|entry| (entry.key, entry.execution)).collect(),
            decode_plans: file
                .decode_plans
                .into_iter()
                .map(|entry| (entry.key, entry.plan))
                .collect(),
        })
}

pub(super) fn persist(
    path: &Path,
    decisions: &HashMap<GateUpKey, GateUpExecution>,
    attention: &HashMap<AttentionKey, PagedExecution>,
    batch_attention: &HashMap<BatchAttentionKey, BatchAttentionExecution>,
    experts: &HashMap<ExpertKey, ExpertExecution>,
    routing: &HashMap<RoutingKey, RoutingExecution>,
    decode_plans: &HashMap<DecodePlanKey, DecodePlan>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = temporary_path(path);
    let mut gate_up = decisions
        .iter()
        .map(|(key, execution)| StoredEntry { key: *key, execution: *execution })
        .collect::<Vec<_>>();
    gate_up.sort_by_key(|entry| {
        (
            entry.key.tokens,
            entry.key.input,
            entry.key.gate,
            entry.key.up,
            entry.key.group_size,
            entry.key.bits,
            entry.key.dtype,
        )
    });
    let mut attention = attention
        .iter()
        .map(|(key, execution)| StoredAttentionEntry { key: *key, execution: *execution })
        .collect::<Vec<_>>();
    attention.sort_by_key(|entry| {
        (
            entry.key.context_bucket,
            entry.key.query_heads,
            entry.key.kv_heads,
            entry.key.head_dim,
            entry.key.page_size,
            entry.key.dtype,
        )
    });
    let mut batch_attention = batch_attention
        .iter()
        .map(|(key, execution)| StoredBatchAttentionEntry { key: *key, execution: *execution })
        .collect::<Vec<_>>();
    batch_attention.sort_by_key(|entry| {
        (
            entry.key.batch,
            entry.key.sequence,
            entry.key.context_bucket,
            entry.key.query_heads,
            entry.key.kv_heads,
            entry.key.head_dim,
            entry.key.dtype,
            entry.key.causal,
            entry.key.fragmented,
            entry.key.view,
        )
    });
    let mut experts = experts
        .iter()
        .map(|(key, execution)| StoredExpertEntry { key: *key, execution: *execution })
        .collect::<Vec<_>>();
    experts.sort_by_key(|entry| {
        (
            entry.key.routes,
            entry.key.experts,
            entry.key.input,
            entry.key.gate,
            entry.key.up,
            entry.key.group_size,
            entry.key.bits,
            entry.key.dtype,
        )
    });
    let mut routing = routing
        .iter()
        .map(|(key, execution)| StoredRoutingEntry { key: *key, execution: *execution })
        .collect::<Vec<_>>();
    routing.sort_by_key(|entry| {
        (
            entry.key.route_bucket,
            entry.key.experts,
            entry.key.top_k,
            entry.key.input,
            entry.key.intermediate,
            entry.key.group_size,
            entry.key.bits,
            entry.key.dtype,
            entry.key.activation,
            entry.key.fused_unsorted,
        )
    });
    let decode_plans = sorted_decode_plans(decode_plans);
    let file = ProfileFile {
        schema: SCHEMA,
        engine_version: env!("CARGO_PKG_VERSION").into(),
        host_architecture: std::env::consts::ARCH.into(),
        gate_up,
        attention,
        batch_attention,
        experts,
        routing,
        decode_plans,
    };
    fs::write(&temporary, serde_json::to_vec_pretty(&file)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn sorted_decode_plans(plans: &HashMap<DecodePlanKey, DecodePlan>) -> Vec<StoredDecodePlan> {
    let mut stored = plans
        .iter()
        .map(|(key, plan)| StoredDecodePlan { key: key.clone(), plan: *plan })
        .collect::<Vec<_>>();
    stored.sort_by(|left, right| {
        (&left.key.model, left.key.weight_bytes, left.key.context_bucket, left.key.batch).cmp(&(
            &right.key.model,
            right.key.weight_bytes,
            right.key.context_bucket,
            right.key.batch,
        ))
    });
    stored
}

pub(super) fn cache_name() -> &'static str {
    "execution-v19-metal-gpu0.json"
}

fn temporary_path(path: &Path) -> PathBuf {
    path.with_extension(format!("tmp-{}", std::process::id()))
}
