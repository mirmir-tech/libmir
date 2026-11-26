//! The gate activation of a gated MLP, `act(gate) * up`: one mapping from a
//! layer's activation to the operation that runs it, for every runtime with
//! a plain gated MLP, so what admission lets through is what executes.

use models::semantic::ActivationSpec;

use crate::engine::{Array, Error, Result, Stream};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateActivation {
    /// `SiLU`: Qwen, Llama, Mistral, `DeepSeek`.
    Silu,
    /// Tanh-approximated GELU: Gemma.
    GeluTanh,
}

impl GateActivation {
    /// The activation of a plain gated MLP: `SwiGLU`, or tanh GELU.
    pub fn from_spec(activation: &ActivationSpec) -> Result<Self> {
        match activation {
            ActivationSpec::SwiGlu => Ok(Self::Silu),
            ActivationSpec::GeluTanh => Ok(Self::GeluTanh),
            other => Err(Error::InvalidModel(format!("dense Metal MLP has no {other:?} gate"))),
        }
    }

    /// `act(gate) * up`.
    pub fn gate(self, gate: &Array, up: &Array, stream: &Stream) -> Result<Array> {
        match self {
            Self::Silu => gate.silu_mul(up, stream),
            Self::GeluTanh => gate.gelu_approx_mul(up, stream),
        }
    }
}
