use std::path::PathBuf;

use mircuda::{Compiler, Driver, bf16};

use super::{LogitShaping, LogitShapingBf16};
use crate::Result;

#[test]
fn identity_shaping_is_none_and_invalid_factors_are_refused() -> Result<()> {
    assert_eq!(LogitShaping::new(None, None)?, None);
    assert_eq!(LogitShaping::new(Some(1.0), None)?, None);
    assert_eq!(
        LogitShaping::new(Some(8.0), None)?,
        Some(LogitShaping { divisor: 8.0, softcap: None })
    );
    assert!(LogitShaping::new(Some(0.0), None).is_err());
    assert!(LogitShaping::new(None, Some(-1.0)).is_err());
    Ok(())
}

/// Logits are divided, then soft-capped.
#[test]
fn shaping_divides_then_caps_logits_on_cuda() -> Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let include = [PathBuf::from("/usr/local/cuda/include")];
    let compiler = Compiler::with_include_paths(context.clone(), include)?;
    let values = [16.0_f32, -4.0, 100.0];
    let shaping = LogitShaping { divisor: 4.0, softcap: Some(10.0) };
    let operation = LogitShapingBf16::compile(&compiler, values.len(), shaping)?;
    let mut host = context.allocate_pinned::<bf16>(values.len())?;
    host.copy_from_slice(&values.map(bf16::from_f32))?;
    let mut device = pool.allocate::<bf16>(&stream, values.len())?;
    stream.copy_to_device(&mut host, &mut device)?;
    operation.execute(&stream, &mut device)?;
    stream.copy_to_host(&device, &mut host)?;
    for (actual, value) in host.to_vec()?.into_iter().zip(values) {
        let expected = (value / 4.0 / 10.0).tanh() * 10.0;
        assert!(
            (actual.to_f32() - expected).abs() <= expected.abs() / 64.0,
            "{actual} vs {expected}"
        );
    }
    Ok(())
}
