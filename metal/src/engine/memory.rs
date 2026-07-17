use super::Result;

#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    pub active: usize,
    pub cached: usize,
    pub peak: usize,
    pub limit: usize,
    pub recommended: Option<usize>,
}

pub fn configure_recommended_wired_limit() -> Result<bool> {
    Ok(mirtal::memory::configure_recommended_wired_limit()?)
}

pub fn memory_stats() -> Result<MemoryStats> {
    let stats = mirtal::memory::stats()?;
    Ok(MemoryStats {
        active: stats.active,
        cached: stats.cached,
        peak: stats.peak,
        limit: stats.limit,
        recommended: (stats.recommended > 0).then_some(stats.recommended),
    })
}
