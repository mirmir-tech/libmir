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
