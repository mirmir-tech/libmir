use runtime::kv::BlockId;

use super::{DecodeInput, LoadedModel, NativeOutput};
use crate::native::error::{Error, Result};

impl LoadedModel {
    #[cfg(test)]
    pub fn decode(
        &mut self,
        session: uuid::Uuid,
        token: u32,
        sampling: runtime::backend::SamplingLogits,
    ) -> Result<NativeOutput> {
        self.decode_with_blocks(session, token, &[], sampling)
    }

    pub fn decode_with_blocks(
        &mut self,
        session: uuid::Uuid,
        token: u32,
        blocks: &[BlockId],
        sampling: runtime::backend::SamplingLogits,
    ) -> Result<NativeOutput> {
        let mut input = DecodeInput::single(session, token, sampling);
        input.blocks = blocks.to_vec();
        let inputs = [input];
        self.release_settled_prefill_reserve()?;
        self.validate_decode_inputs(&inputs)?;
        self.bind_decode_blocks(&inputs)?;
        let result = self.decode_admitted(session, token, &inputs[0].sampling, None);
        self.finish_decode(&inputs, result)
    }

    pub(super) fn bind_decode_blocks(&mut self, inputs: &[DecodeInput]) -> Result<()> {
        let stream = &self.stream;
        let model_id = &self.info.manifest.id;
        for input in inputs {
            if input.blocks.is_empty() {
                continue;
            }
            let state = self.sessions.get_mut(&input.session).ok_or_else(|| Error::Session {
                model: model_id.clone(),
                session: input.session,
            })?;
            state.cache.bind_blocks(&input.blocks, None, stream)?;
            if let Some(speculation) = state.speculation.as_mut() {
                speculation.bind_blocks(&input.blocks, stream)?;
            }
        }
        Ok(())
    }
}
