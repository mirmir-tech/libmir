#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRequirement {
    pub label: String,
    pub aliases: Vec<String>,
    pub include_dense_dtype: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderTensorSchema {
    pub requirements: Vec<TensorRequirement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionTensorSchema {
    pub requirements: Vec<TensorRequirement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorReadiness {
    pub required: usize,
    pub present: usize,
    pub missing: Vec<String>,
    pub dtypes: Vec<String>,
}
