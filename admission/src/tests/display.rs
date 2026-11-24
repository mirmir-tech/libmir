//! Refusals read as words, not as `Debug` of their types.

use crate::{
    ArchitectureError, Rejection, Runtime, TaskRefusal,
    capability::{
        AttentionScale, AttentionScaleValue, FeedForwardShape, LayerShape, MixerKind,
        Requirement, ScaleSource,
    },
    cuda::CudaDecoderRuntime,
    decoder::Feature,
};

#[test]
fn names_what_each_runtime_lacks() {
    let error = ArchitectureError::NoRuntime {
        rejections: vec![Rejection {
            runtime: Runtime::Cuda(CudaDecoderRuntime::Dense),
            missing: vec![
                Requirement::Layer(LayerShape {
                    mixer: MixerKind::Linear,
                    feed_forward: FeedForwardShape::Dense,
                }),
                Requirement::Feature(Feature::LayerNorm),
                Requirement::AttentionScale(AttentionScale {
                    value: AttentionScaleValue::Other,
                    source: ScaleSource::Override,
                }),
            ],
        }],
    };
    assert_eq!(
        error.to_string(),
        "no runtime executes the decoder; the CUDA dense runtime lacks linear attention layers \
         with a dense MLP, LayerNorm, another attention scale overriding the configuration"
    );
    assert_eq!(
        ArchitectureError::Task(TaskRefusal::EmbeddingPooling).to_string(),
        "embeddings are pooled only from the normalized last token"
    );
}
