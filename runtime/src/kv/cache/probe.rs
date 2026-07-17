use crate::kv::{BlockHash, KvCache, PrefixProbe};

impl KvCache {
    #[must_use]
    pub fn probe_prefix(&self, model: &str, tokens: &[u32]) -> PrefixProbe {
        self.probe_prefix_inner(model, tokens)
    }

    pub(crate) fn probe_prefix_recorded(&mut self, model: &str, tokens: &[u32]) -> PrefixProbe {
        let probe = self.probe_prefix_inner(model, tokens);
        self.counters.record_prefix_probe(probe.cached_tokens, probe.missing_tokens);
        probe
    }

    fn probe_prefix_inner(&self, model: &str, tokens: &[u32]) -> PrefixProbe {
        let mut cached_blocks = Vec::new();
        let mut cached_tokens = 0;
        let mut parent = None;
        for chunk in tokens.chunks(self.config.block_size) {
            let hash = BlockHash::from_tokens(model, parent, chunk);
            let Some(block) = self.prefix.get(hash) else {
                break;
            };
            cached_blocks.push(block);
            cached_tokens += chunk.len();
            parent = Some(hash);
        }
        PrefixProbe {
            cached_blocks,
            cached_tokens,
            missing_tokens: tokens.len().saturating_sub(cached_tokens),
            last_hash: parent,
        }
    }
}
