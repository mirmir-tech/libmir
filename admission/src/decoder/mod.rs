//! Backend-neutral decoder features, lowered once for both backends.

mod features;

pub use features::{DecoderFeatures, Feature, lower_features};
