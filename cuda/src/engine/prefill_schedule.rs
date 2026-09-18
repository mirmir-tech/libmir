/// Scheduling strategy supported by a loaded CUDA generation runner.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CudaPrefillSchedule {
    /// Share each round across pending rows.
    #[default]
    RoundRobin,
    /// Complete rows in queue order, carrying unused budget into the next row.
    CompletionFirst,
}
