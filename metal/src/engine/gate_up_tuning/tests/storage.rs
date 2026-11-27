use std::{fs, time::Duration};

use runtime::tuning::TuningConfig;

use super::{GateUpExecution, MetalTuner, fixture_key};

#[test]
fn rejects_profiles_from_a_different_mlx_or_gpu() -> Result<(), Box<dyn std::error::Error>> {
    let directory =
        std::env::temp_dir().join(format!("libmir-metal-environment-{}", std::process::id()));
    let mut tuner = MetalTuner::new(TuningConfig {
        cache_directory: Some(directory.clone()),
        ..TuningConfig::default()
    });
    tuner.record(fixture_key(), GateUpExecution::Separate, Duration::from_millis(1));
    tuner.persist();
    let path = directory.join(super::super::storage::cache_name());
    let original = fs::read(&path)?;
    assert!(super::super::storage::load(&path).is_some());
    for field in ["mlx_version", "device_name"] {
        let mut profile: serde_json::Value = serde_json::from_slice(&original)?;
        profile["environment"][field] = "different-environment".into();
        fs::write(&path, serde_json::to_vec(&profile)?)?;
        assert!(super::super::storage::load(&path).is_none(), "stale {field}");
    }
    let mut profile: serde_json::Value = serde_json::from_slice(&original)?;
    let hash = &mut profile["environment"]["executable_hash"][0];
    *hash = ((hash.as_u64().ok_or("missing executable hash")? + 1) % 256).into();
    fs::write(&path, serde_json::to_vec(&profile)?)?;
    assert!(super::super::storage::load(&path).is_none(), "stale implementation");
    fs::write(&path, &original)?;
    assert!(super::super::storage::load(&path).is_some());
    fs::remove_dir_all(directory)?;
    Ok(())
}

/// Two tuners over one empty cache, each measuring its own shape, persist
/// one after the other: the profile keeps both decisions, where the second
/// snapshot used to replace the first's. Where both measured one key, the
/// later measurement stands.
#[test]
fn tuners_sharing_a_cache_keep_each_others_decisions() -> Result<(), Box<dyn std::error::Error>> {
    let directory =
        std::env::temp_dir().join(format!("libmir-metal-two-tuners-{}", std::process::id()));
    drop(fs::remove_dir_all(&directory));
    let config = || TuningConfig {
        cache_directory: Some(directory.clone()),
        ..TuningConfig::default()
    };
    let one = fixture_key();
    let two = super::super::GateUpKey { tokens: 2, ..one };
    let mut first = MetalTuner::new(config());
    let mut second = MetalTuner::new(config());
    first.record(one, GateUpExecution::Separate, Duration::from_millis(1));
    second.record(two, GateUpExecution::Fused, Duration::from_millis(1));
    first.persist();
    second.persist();
    let path = directory.join(super::super::storage::cache_name());
    let stored = super::super::storage::load(&path).ok_or("the profile did not load")?;
    assert_eq!(stored.gate_up.get(&one), Some(&GateUpExecution::Separate));
    assert_eq!(stored.gate_up.get(&two), Some(&GateUpExecution::Fused));

    let mut third = MetalTuner::new(config());
    third.record(one, GateUpExecution::Fused, Duration::from_millis(1));
    first.persist();
    third.persist();
    let stored = super::super::storage::load(&path).ok_or("the profile did not load")?;
    assert_eq!(stored.gate_up.get(&one), Some(&GateUpExecution::Fused));
    assert_eq!(stored.gate_up.get(&two), Some(&GateUpExecution::Fused));
    fs::remove_dir_all(directory)?;
    Ok(())
}
