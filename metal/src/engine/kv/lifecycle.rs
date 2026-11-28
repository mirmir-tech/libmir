use super::{KvCache, KvStorage, Result, Stream};

impl KvCache {
    pub(crate) fn extend_graph_roots<'a>(&'a self, roots: &mut Vec<&'a super::Array>) {
        match &self.storage {
            KvStorage::Contiguous(store) => {
                roots.extend([&store.keys, &store.values].into_iter().flatten());
            },
            KvStorage::Sliding(store) => {
                roots.extend([&store.keys, &store.values].into_iter().flatten());
            },
            KvStorage::Paged(_) => {},
        }
    }

    pub(crate) fn has_quantized_pages(&self) -> bool {
        match &self.storage {
            KvStorage::Paged(pages) => pages.quantized(),
            KvStorage::Contiguous(_) | KvStorage::Sliding(_) => false,
        }
    }

    pub(crate) const fn is_windowed(&self) -> bool {
        matches!(self.storage, KvStorage::Sliding(_))
    }

    pub(crate) const fn has_active_pages(&self) -> bool {
        match &self.storage {
            KvStorage::Paged(pages) => pages.active(),
            KvStorage::Contiguous(_) | KvStorage::Sliding(_) => false,
        }
    }

    pub(crate) fn release_history(&mut self) {
        self.history = None;
    }

    pub fn reset(&mut self) -> Result<()> {
        self.history = None;
        match &mut self.storage {
            KvStorage::Contiguous(store) => store.clear(),
            KvStorage::Sliding(store) => {
                store.keys = None;
                store.values = None;
                store.capacity = 0;
                store.write_index = 0;
                store.last_update_tokens = 0;
            },
            KvStorage::Paged(pages) => pages.reset()?,
        }
        self.offset = 0;
        Ok(())
    }

    pub(crate) fn detach_evaluated_graphs(&self, stream: &Stream) -> Result<()> {
        match &self.storage {
            KvStorage::Contiguous(store) => {
                for array in [&store.keys, &store.values].into_iter().flatten() {
                    array.detach_graph(stream)?;
                }
            },
            KvStorage::Sliding(store) => {
                for array in [&store.keys, &store.values].into_iter().flatten() {
                    array.detach_graph(stream)?;
                }
            },
            KvStorage::Paged(pages) => pages.detach_evaluated_graph(stream)?,
        }
        Ok(())
    }
}
