//! Pixel-shuffle vision prefill on Metal: the contract is discovered and
//! prepared backend-neutrally, but no Metal tower runs it yet. The tower
//! (prefix tokens, resized position grid, ViT blocks, pixel shuffle and
//! RMS-normalized ReLU² projector) and a decoder it feeds belong here.

use models::vision::{PixelShufflePreprocessedImage, PixelShufflePromptTokens};
use runtime::{
    Error as RuntimeError, Result as RuntimeResult,
    backend::{ModelHandle, PrefillOutput, SamplingLogits},
    kv::BlockTable,
};
use uuid::Uuid;

use super::super::MetalBackend;
use crate::MetalProgressEvent;

impl MetalBackend {
    #[allow(clippy::too_many_arguments, clippy::unused_self)]
    pub fn prefill_pixel_shuffle_vision_with_progress(
        &self,
        _model: &ModelHandle,
        _session_id: Uuid,
        _prompt: &PixelShufflePromptTokens,
        _image: &PixelShufflePreprocessedImage,
        _block_table: &BlockTable,
        _sampling_logits: SamplingLogits,
        _progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> RuntimeResult<PrefillOutput> {
        Err(RuntimeError::BackendUnavailable(
            "pixel-shuffle vision prefill is not implemented on Metal".into(),
        ))
    }
}
