use std::{sync::mpsc, thread, time::Duration};

use super::*;

const GIB: u64 = 1024 * 1024 * 1024;

#[test]
fn tracks_loading_and_resident_reservations_until_release() -> Result<()> {
    let manager = ModelMemoryManager::default();
    let lease = manager.reserve(
        "model-a".into(),
        estimate(4 * GIB),
        &memory(16 * GIB),
        MemoryRuntimeConfig::default(),
        false,
    )?;
    {
        let ledger = lock_ledger(&manager)?;
        assert_eq!(ledger.entries.len(), 1);
        assert_eq!(
            ledger.entries.values().next().map(|entry| entry.state),
            Some(ReservationState::Loading)
        );
        assert_eq!(ledger.entries.values().next().map(|entry| entry.bytes), Some(6 * GIB));
        drop(ledger);
    }

    lease.mark_resident()?;
    {
        let ledger = lock_ledger(&manager)?;
        assert_eq!(
            ledger.entries.values().next().map(|entry| entry.state),
            Some(ReservationState::Resident)
        );
        drop(ledger);
    }

    drop(lease);
    assert!(lock_ledger(&manager)?.entries.is_empty());
    Ok(())
}

#[test]
fn rejects_loads_that_exceed_the_remaining_safe_budget() -> Result<()> {
    let manager = ModelMemoryManager::default();
    let first = manager.reserve(
        "model-a".into(),
        estimate(5 * GIB),
        &memory(16 * GIB),
        MemoryRuntimeConfig::default(),
        false,
    )?;
    first.mark_resident()?;

    let error = manager
        .reserve(
            "model-b".into(),
            estimate(5 * GIB),
            &memory(16 * GIB),
            MemoryRuntimeConfig::default(),
            false,
        )
        .err()
        .ok_or_else(|| runtime::RuntimeError::Config("second load was admitted".into()))?;

    assert!(matches!(
        error,
        Error::MemoryAdmission {
            required_bytes,
            available_bytes,
            ..
        } if required_bytes == 15 * GIB / 2 && available_bytes == 9 * GIB / 2
    ));
    Ok(())
}

#[test]
fn explicit_overcommit_preserves_forced_load_semantics() -> Result<()> {
    let manager = ModelMemoryManager::default();
    let lease = manager.reserve(
        "model".into(),
        estimate(16 * GIB),
        &memory(8 * GIB),
        MemoryRuntimeConfig::default(),
        true,
    )?;

    drop(lease);
    Ok(())
}

#[test]
fn serializes_model_load_critical_sections() -> Result<()> {
    let manager = ModelMemoryManager::default();
    let held = manager.serialize_load()?;
    let contender = manager.clone();
    let (sender, receiver) = mpsc::channel();
    let worker = thread::spawn(move || -> Result<()> {
        let guard = contender.serialize_load()?;
        let Ok(()) = sender.send(()) else {
            return Err(runtime::RuntimeError::Config("load signal failed".into()).into());
        };
        drop(guard);
        Ok(())
    });

    assert!(receiver.recv_timeout(Duration::from_millis(20)).is_err());
    drop(held);
    let Ok(()) = receiver.recv_timeout(Duration::from_secs(1)) else {
        return Err(runtime::RuntimeError::Config("serialized load timed out".into()).into());
    };
    let Ok(result) = worker.join() else {
        return Err(runtime::RuntimeError::Config("load worker panicked".into()).into());
    };
    result?;
    Ok(())
}

fn lock_ledger(manager: &ModelMemoryManager) -> Result<MutexGuard<'_, MemoryLedger>> {
    let Ok(ledger) = manager.ledger.lock() else {
        return Err(poisoned("test ledger"));
    };
    Ok(ledger)
}

fn estimate(required_bytes: u64) -> ModelMemoryEstimate {
    ModelMemoryEstimate {
        weight_bytes: required_bytes,
        kv_cache_bytes: 0,
        workspace_bytes: 0,
        required_bytes,
        kv_bytes_per_token: 0,
        cache_capacity_tokens: 0,
        model_context_tokens: 0,
    }
}

fn memory(total: u64) -> MemorySnapshot {
    MemorySnapshot {
        total_bytes: Some(total),
        available_bytes: Some(total),
        active_bytes: 0,
        cached_bytes: 0,
        allocation_reserve_bytes: 0,
        source: "test".into(),
        unified: true,
    }
}
