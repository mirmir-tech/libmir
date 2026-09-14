use super::{DecodeCapacity, KvCache};
use crate::engine::{Error, Result};

#[cfg(test)]
mod tests;

impl KvCache {
    pub(crate) fn plan_decode_capacity(
        &self,
        tokens: usize,
        threshold: usize,
        plan: &mut DecodeCapacity,
    ) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        let end = self.offset.checked_add(tokens).ok_or(Error::ShapeOverflow)?;
        if let Some(pages) = &self.pages
            && (pages.active() || end >= threshold)
        {
            pages.plan_decode(self.keys.as_ref(), self.offset, tokens, plan)?;
        }
        Ok(())
    }
}
