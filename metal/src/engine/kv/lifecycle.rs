use super::{KvCache, Result, Stream};

impl KvCache {
    #[cfg(test)]
    pub(crate) fn release_history(&mut self) {
        self.history = None;
    }

    pub fn reset(&mut self) -> Result<()> {
        #[cfg(test)]
        {
            self.history = None;
        }
        self.keys = None;
        self.values = None;
        if let Some(pages) = self.pages.as_mut() {
            pages.reset()?;
        }
        self.offset = 0;
        self.capacity = 0;
        self.write_index = 0;
        Ok(())
    }

    pub(crate) fn release_reservation_after(&mut self, tokens: usize) -> Result<()> {
        #[cfg(test)]
        {
            self.history = None;
        }
        if let Some(pages) = self.pages.as_mut() {
            pages.release_reservation_after(tokens)?;
        }
        Ok(())
    }

    pub(crate) fn plan_contiguous(&mut self, tokens: usize) {
        if let Some(pages) = self.pages.as_mut() {
            pages.plan_contiguous(tokens);
        }
    }

    pub(crate) fn detach_evaluated_graphs(&self, stream: &Stream) -> Result<()> {
        for array in [&self.keys, &self.values].into_iter().flatten() {
            array.detach_graph(stream)?;
        }
        if let Some(pages) = &self.pages {
            pages.detach_evaluated_graph(stream)?;
        }
        Ok(())
    }
}
