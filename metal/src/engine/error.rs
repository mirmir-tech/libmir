use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("persistent history budget is exhausted or suspended")]
    HistoryBudgetUnavailable,
    #[error("MLX {0} returned a null handle")]
    NullHandle(&'static str),
    #[error("model tensor is missing: {0}")]
    MissingTensor(String),
    #[error("invalid quantized tensor layout: {0}")]
    InvalidQuantization(String),
    #[error("invalid native model configuration: {0}")]
    InvalidModel(String),
    /// A GGUF file maps its names by its model's metadata and loads only by
    /// its share of the model's weight catalog.
    #[error("GGUF file {} is not in the model's weight catalog", .0.display())]
    UncataloguedGguf(std::path::PathBuf),
    /// Caches are indexed by layer, so the layers sharing an earlier
    /// layer's keys and values, which keep none, must come last.
    #[error("layer {layer} caches keys and values after a layer that shares them")]
    CachingLayerAfterSharing { layer: usize },
    #[error("multi-token attention cannot read a paged K/V context")]
    MultiTokenPagedAttention,
    #[error(
        "Metal execution requires {required} K/V pages in layer {layer} but its capacity is {maximum}"
    )]
    KvPageCapacity {
        layer: usize,
        required: usize,
        maximum: usize,
    },
    #[error("Metal K/V block {block} in layer {layer} exceeds arena capacity {capacity}")]
    KvBlockOutOfRange {
        layer: usize,
        block: u32,
        capacity: usize,
    },
    #[error("invalid sampling configuration: {0}")]
    InvalidSampling(String),
    #[error("float conversion failed: {0}")]
    Float(#[from] std::num::ParseFloatError),
    #[error("model layout error: {0}")]
    Model(#[from] models::ModelsError),
    #[error("Metal does not run this architecture: {0}")]
    Architecture(#[from] architecture::ArchitectureError),
    #[error("mirtal error: {0}")]
    Mirtal(#[from] mirtal::Error),
    #[error("integer conversion failed: {0}")]
    Integer(#[from] std::num::TryFromIntError),
    #[error("integer parse failed: {0}")]
    IntegerParse(#[from] std::num::ParseIntError),
    #[cfg(test)]
    #[error("benchmark clock is before the Unix epoch: {0}")]
    BenchmarkClock(#[from] std::time::SystemTimeError),
    #[cfg(test)]
    #[error("benchmark output failed: {0}")]
    BenchmarkOutput(#[from] std::io::Error),
    #[cfg(test)]
    #[error("benchmark stability failed for {case} ({variant}): {spread_percent}% spread")]
    BenchmarkStability {
        case: String,
        variant: String,
        spread_percent: f64,
    },
    #[cfg(test)]
    #[error("test fixture JSON failed: {0}")]
    TestJson(#[from] serde_json::Error),
    #[error("shape {shape:?} contains {elements} elements, data contains {data}")]
    Shape {
        shape: Vec<i32>,
        elements: usize,
        data: usize,
    },
    #[error("shape element count overflowed usize")]
    ShapeOverflow,
}
