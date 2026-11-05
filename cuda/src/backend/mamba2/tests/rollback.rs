//! A verification pass rolled back to any accepted prefix leaves exactly the
//! state and output of a pass over that prefix alone.

use super::{Device, SMALL, parameters, pattern, read, run, width};
use crate::{CudaConfig, Result, backend::CudaBackend};

#[test]
fn rolling_back_keeps_the_state_after_the_accepted_prefix() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let device = Device::upload(&backend, &parameters(SMALL))?;
    let width = width(SMALL);
    let inner = SMALL.inner()?;
    let prompt = pattern(5 * width, 40, 1.0);
    let pass = pattern(4 * width, 41, 1.0);
    let after = pattern(2 * width, 42, 1.0);
    for kept in 1..=4 {
        let mut speculative = backend.prepare_mamba2_state(SMALL)?;
        let mut alone = backend.prepare_mamba2_state(SMALL)?;
        run(&backend, &mut speculative, &prompt, &device)?;
        run(&backend, &mut alone, &prompt, &device)?;
        speculative.arm_rollback(3)?;
        let output = run(&backend, &mut speculative, &pass, &device)?;
        speculative.roll_back(kept)?;
        let expected = run(&backend, &mut alone, &pass[..kept * width], &device)?;
        assert_eq!(output[..kept * inner], expected, "output of {kept} kept tokens");
        assert_eq!(speculative.offset(), alone.offset(), "offset after {kept} kept tokens");
        assert_eq!(read(&backend, &speculative.state)?, read(&backend, &alone.state)?);
        assert_eq!(read(&backend, &speculative.history)?, read(&backend, &alone.history)?);
        let next = run(&backend, &mut speculative, &after, &device)?;
        assert_eq!(next, run(&backend, &mut alone, &after, &device)?, "after {kept} kept tokens");
    }
    Ok(())
}

#[test]
fn a_resident_verification_keeps_the_accepted_prefix() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let device = Device::upload(&backend, &parameters(SMALL))?;
    let width = width(SMALL);
    let inner = SMALL.inner()?;
    let mut verify = backend.prepare_mamba2_verify_state(SMALL, 4)?;
    let mut speculative = backend.prepare_mamba2_state(SMALL)?;
    let mut alone = backend.prepare_mamba2_state(SMALL)?;
    let prompt = pattern(5 * width, 50, 1.0);
    run(&backend, &mut speculative, &prompt, &device)?;
    run(&backend, &mut alone, &prompt, &device)?;
    // Consecutive passes keep the state resident; a solo step between them
    // takes it back.
    for (step, kept) in [2_usize, 4, 1, 3].into_iter().enumerate() {
        let pass = pattern(4 * width, 60 + u64::try_from(step)?, 1.0);
        verify.pack(&speculative)?;
        let input = super::copy(&backend, &pass)?;
        let mut output = backend.inner.pool.allocate(&backend.inner.stream, 4 * inner)?;
        verify.execute(&input, device.view(), &mut output)?;
        verify.commit(&mut speculative, kept)?;
        let expected = run(&backend, &mut alone, &pass[..kept * width], &device)?;
        assert_eq!(read(&backend, &output)?[..kept * inner], expected, "step {step}");
        assert_eq!(speculative.offset(), alone.offset(), "offset after step {step}");
        if step == 1 {
            let solo = pattern(width, 70, 1.0);
            assert_eq!(
                run(&backend, &mut speculative, &solo, &device)?,
                run(&backend, &mut alone, &solo, &device)?
            );
        }
    }
    let after = pattern(2 * width, 80, 1.0);
    assert_eq!(
        run(&backend, &mut speculative, &after, &device)?,
        run(&backend, &mut alone, &after, &device)?
    );
    Ok(())
}
