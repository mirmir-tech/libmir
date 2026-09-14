use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[cfg(test)]
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
    #[error(
        "Metal execution requires {required} K/V pages in layer {layer} but its capacity is {maximum}"
    )]
    KvPageCapacity {
        layer: usize,
        required: usize,
        maximum: usize,
    },
    #[error("invalid sampling configuration: {0}")]
    InvalidSampling(String),
    #[error("float conversion failed: {0}")]
    Float(#[from] std::num::ParseFloatError),
    #[error("model layout error: {0}")]
    Model(#[from] models::ModelsError),
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
