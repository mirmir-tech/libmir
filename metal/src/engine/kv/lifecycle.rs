use super::{KvCache, Result};

impl KvCache {
    pub(crate) fn plan_contiguous(&mut self, tokens: usize) {
        if let Some(pages) = self.pages.as_mut() {
            pages.plan_contiguous(tokens);
        }
    }

    pub(crate) fn detach_evaluated_graphs(&self) -> Result<()> {
        for array in [&self.keys, &self.values].into_iter().flatten() {
            array.native().detach_graph()?;
        }
        if let Some(pages) = &self.pages {
            pages.detach_evaluated_graph()?;
        }
        Ok(())
    }
}
