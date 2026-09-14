use super::*;

#[test]
fn long_wait_does_not_join_an_existing_short_window() -> crate::Result<()> {
    let cohort = CacheCohort::new(20, 16, 4);
    cohort
        .shared
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .short
        .deadline = Some(Instant::now() + Duration::from_millis(1));

    let elapsed = cohort.wait(true, 17, &crate::CancellationToken::new())?;

    assert!(elapsed >= Duration::from_millis(10), "long window lasted only {elapsed:?}");
    Ok(())
}

#[test]
fn identical_long_fill_waits_for_the_leader() -> crate::Result<()> {
    assert_fill_ownership(CacheCohort::new(20, 16, 4))?;
    assert_fill_ownership(CacheCohort::for_backend(
        &foundation::model::BackendTarget::Cuda,
        2_000,
        16,
        4,
    ))
}

fn assert_fill_ownership(cohort: CacheCohort) -> crate::Result<()> {
    let cohort = Arc::new(cohort);
    let claim = cohort.claim_fill(&[7; 19], &[17], 0, &crate::CancellationToken::new())?;
    assert!(matches!(claim, FillClaim::Leader(Some(_))));
    let FillClaim::Leader(Some(leader)) = claim else {
        return Ok(());
    };
    let follower = Arc::clone(&cohort);
    let (sender, receiver) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let claim = follower.claim_fill(&[7; 21], &[17], 0, &crate::CancellationToken::new());
        assert!(sender.send(claim).is_ok());
    });
    assert!(receiver.recv_timeout(Duration::from_millis(10)).is_err());
    drop(leader);
    assert!(matches!(
        receiver.recv_timeout(Duration::from_secs(1)),
        Ok(Ok(FillClaim::Retry(_)))
    ));
    assert!(worker.join().is_ok());
    Ok(())
}

#[test]
fn follower_cancellation_does_not_release_the_leaders_fill() -> crate::Result<()> {
    let cohort = Arc::new(CacheCohort::new(20, 16, 4));
    let token = crate::CancellationToken::new();
    let leader = cohort.claim_fill(&[7; 19], &[17], 0, &token)?;
    let follower = cohort.clone();
    let cancelled = token.clone();
    let (sent, received) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result = follower.claim_fill(&[7; 21], &[17], 0, &cancelled);
        assert!(sent.send(matches!(result, Err(crate::Error::Cancelled))).is_ok());
    });
    assert!(received.recv_timeout(Duration::from_millis(20)).is_err());
    token.cancel();
    assert_eq!(received.recv_timeout(Duration::from_secs(1)), Ok(true));
    assert_eq!(
        cohort
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .fills
            .len(),
        1
    );
    drop(leader);
    assert!(
        cohort
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .fills
            .is_empty()
    );
    assert!(worker.join().is_ok());
    Ok(())
}

#[test]
fn cancelled_cohort_wait_does_not_wait_for_a_long_admission_window() {
    let cohort = Arc::new(CacheCohort::new(10_000, 16, 4));
    let token = crate::CancellationToken::new();
    let cancellation = token.clone();
    let (sent, received) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        assert!(
            sent.send(matches!(cohort.wait(true, 32, &cancellation), Err(crate::Error::Cancelled)))
                .is_ok()
        );
    });
    assert!(received.recv_timeout(Duration::from_millis(20)).is_err());
    token.cancel();
    assert_eq!(received.recv_timeout(Duration::from_secs(1)), Ok(true));
    assert!(worker.join().is_ok());
}

#[test]
fn cuda_eviction_has_no_artificial_wait_and_still_honors_cancellation() -> crate::Result<()> {
    let cohort = CacheCohort::for_backend(&foundation::model::BackendTarget::Cuda, 2_000, 16, 4);
    let token = crate::CancellationToken::new();
    assert_eq!(cohort.wait(true, 32, &token)?, Duration::ZERO);
    assert_eq!(cohort.wait(true, 8, &token)?, Duration::ZERO);
    token.cancel();
    assert!(matches!(cohort.wait(true, 32, &token), Err(crate::Error::Cancelled)));
    let metal = CacheCohort::for_backend(&foundation::model::BackendTarget::Metal, 2_000, 16, 4);
    assert_eq!(metal.long_wait, Duration::from_secs(2));
    assert_eq!(metal.short_wait, Duration::from_millis(100));
    Ok(())
}
