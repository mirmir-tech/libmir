use std::sync::Arc;

use super::{SourceAttention, StepShare};

/// A source's attention stays for its readers within the step and leaves
/// with the last of them, rather than lasting into the next step.
#[test]
fn the_last_reader_takes_a_sources_attention() -> crate::engine::Result<()> {
    let step = StepShare::new(&[2, 0]);
    step.publish(0, SourceAttention::Rows(Vec::new()))?;
    let first = step.attended(0)?;
    assert_eq!(Arc::strong_count(&first), 2, "the slot keeps it for the second reader");
    let second = step.attended(0)?;
    drop(first);
    assert_eq!(Arc::strong_count(&second), 1, "the slot let go of it");
    assert!(step.attended(0).is_err(), "a third read in one step");
    // A layer no later layer reads keeps nothing.
    step.publish(1, SourceAttention::Rows(Vec::new()))?;
    assert!(step.attended(1).is_err());
    // The next step publishes afresh.
    step.publish(0, SourceAttention::Rows(Vec::new()))?;
    assert!(step.attended(0).is_ok());
    Ok(())
}
