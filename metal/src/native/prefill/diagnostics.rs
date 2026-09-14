pub(in crate::native) use crate::engine::diagnostics::Stage;
use crate::native::error::Result;

pub(in crate::native) fn measure<T>(
    stage: Stage,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    crate::engine::diagnostics::measure(stage, operation)
}
