use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use super::{
    AttentionProfileRequest, MoeProfileExecution, MoeProfileRequest, MoeRuntimeEntry,
    QuantizedProfileExecution, QuantizedProfileRequest, moe::MoeProfileFormat,
};
use crate::{AttentionExecution, DenseExecution, DensePlanRequest};

const SCHEMA: u32 = 14;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) struct DeviceKey {
    pub(super) name: String,
    pub(super) compute_capability: (i32, i32),
    pub(super) multiprocessors: u32,
    pub(super) integrated: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct ProfileFile {
    schema: u32,
    engine_version: String,
    device: DeviceKey,
    dense: Vec<StoredDenseEntry>,
    attention: Vec<StoredAttentionEntry>,
    moe: Vec<StoredMoeEntry>,
    quantized: Vec<StoredQuantizedEntry>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(super) struct StoredDenseEntry {
    pub(super) request: DensePlanRequest,
    pub(super) execution: DenseExecution,
    pub(super) average_ns: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(super) struct StoredAttentionEntry {
    pub(super) request: AttentionProfileRequest,
    pub(super) execution: AttentionExecution,
    pub(super) average_ns: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(super) struct StoredMoeEntry {
    pub(super) request: MoeProfileRequest,
    pub(super) execution: MoeProfileExecution,
    pub(super) average_ns: u64,
}

pub(super) fn stored_moe_entries(
    entries: &HashMap<MoeProfileRequest, MoeRuntimeEntry>,
) -> Vec<StoredMoeEntry> {
    let mut stored = entries
        .iter()
        .map(|(request, entry)| StoredMoeEntry {
            request: *request,
            execution: entry.execution,
            average_ns: entry.average_ns,
        })
        .collect::<Vec<_>>();
    stored.sort_by_key(|entry| {
        let format = match entry.request.format {
            MoeProfileFormat::NvFp4 { activation } => (0, 0, 4, activation.code()),
            MoeProfileFormat::Affine { group_size, bits, activation } => {
                (1, group_size, bits, activation.code())
            },
            MoeProfileFormat::Clamped { storage } => (2, storage as usize, 4, 0),
            MoeProfileFormat::MxFp4 { storage, activation } => {
                (3, 32 + storage as usize, 4, activation.code())
            },
            MoeProfileFormat::MxFp8 { storage, bias, activation } => {
                (4, 32 + storage as usize, 8 + usize::from(bias), activation.code())
            },
        };
        (
            entry.request.phase as u8,
            entry.request.tokens,
            entry.request.experts,
            entry.request.top_k,
            entry.request.hidden_features,
            entry.request.intermediate_features,
            format,
        )
    });
    stored
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(super) struct StoredQuantizedEntry {
    pub(super) request: QuantizedProfileRequest,
    pub(super) execution: QuantizedProfileExecution,
    pub(super) average_ns: u64,
}

#[derive(Debug, Default)]
pub(super) struct StoredProfile {
    pub(super) dense: Vec<StoredDenseEntry>,
    pub(super) attention: Vec<StoredAttentionEntry>,
    pub(super) moe: Vec<StoredMoeEntry>,
    pub(super) quantized: Vec<StoredQuantizedEntry>,
}

pub(super) fn load(path: &Path, device: &DeviceKey) -> Option<StoredProfile> {
    let bytes = fs::read(path).ok()?;
    let file: ProfileFile = serde_json::from_slice(&bytes).ok()?;
    (file.schema == SCHEMA
        && file.engine_version == env!("CARGO_PKG_VERSION")
        && file.device == *device)
        .then_some(StoredProfile {
            dense: file.dense,
            attention: file.attention,
            moe: file.moe,
            quantized: file.quantized,
        })
}

pub(super) fn persist(
    path: &Path,
    device: &DeviceKey,
    profile: StoredProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = temporary_path(path);
    let file = ProfileFile {
        schema: SCHEMA,
        engine_version: env!("CARGO_PKG_VERSION").into(),
        device: device.clone(),
        dense: profile.dense,
        attention: profile.attention,
        moe: profile.moe,
        quantized: profile.quantized,
    };
    fs::write(&temporary, serde_json::to_vec_pretty(&file)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub(super) fn cache_name(device: &DeviceKey) -> String {
    let name = device
        .name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    format!(
        "execution-v{SCHEMA}-{name}-sm{}{}-{}.json",
        device.compute_capability.0, device.compute_capability.1, device.multiprocessors
    )
}

fn temporary_path(path: &Path) -> PathBuf {
    path.with_extension(format!("tmp-{}", std::process::id()))
}
