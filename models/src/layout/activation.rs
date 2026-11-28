//! Normalized feed-forward activation function parsed once from model configs.

use std::fmt;

/// Activation function declared by a decoder, encoder, or vision configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HiddenActivation {
    Silu,
    Gelu,
    GeluTanh,
    Relu,
    Other(String),
}

impl HiddenActivation {
    /// Parses a Hugging Face `hidden_act` / `hidden_activation` value,
    /// resolving standard aliases once at the config boundary.
    #[must_use]
    pub fn parse(name: &str) -> Self {
        match name {
            "silu" | "swish" => Self::Silu,
            "gelu" => Self::Gelu,
            "gelu_pytorch_tanh" | "gelu_new" | "gelu_tanh" => Self::GeluTanh,
            "relu" => Self::Relu,
            other => Self::Other(other.to_owned()),
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Silu => "silu",
            Self::Gelu => "gelu",
            Self::GeluTanh => "gelu_pytorch_tanh",
            Self::Relu => "relu",
            Self::Other(name) => name,
        }
    }
}

impl fmt::Display for HiddenActivation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}
