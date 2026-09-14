use std::{path::PathBuf, time::Duration};

use super::*;
use crate::native::error::Error;

#[test]
#[ignore = "one decode-only shader trace, attach after ready; set MIRMIR_BENCH_TRACE_DIR"]
fn captures_decode_shaders() -> Result<()> {
    let mut model = tiles::load_model()?;
    let mut sample = None;
    let observation = run_with_decode(&mut model, MoePrefill::Default, 2049, |model, inputs| {
        wait_for_trace()?;
        let (observation, measured) = probe::measure(model, inputs)?;
        sample = Some(measured);
        Ok(observation)
    })?;
    writeln!(
        std::io::stderr().lock(),
        "decode.shaders: {}",
        json!({
            "observation": observation, "sample": sample,
        })
    )?;
    Ok(())
}

/// An external observer starts only after the test reaches a settled boundary.
pub(super) fn wait_for_trace() -> Result<()> {
    let directory = std::env::var_os("MIRMIR_BENCH_TRACE_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| Error::Benchmark("MIRMIR_BENCH_TRACE_DIR is required".into()))?;
    assert!(!directory.join("ready").exists() && !directory.join("go").exists());
    std::fs::write(directory.join("ready"), std::process::id().to_string())?;
    let deadline = Instant::now() + Duration::from_secs(30);
    while !directory.join("go").exists() {
        if Instant::now() >= deadline {
            return Err(Error::Benchmark("trace observer did not start within 30 s".into()));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    Ok(())
}
