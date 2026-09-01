use super::BoundLinear;
use crate::engine::{
    FusedAttention, FusedExpertGateUp, FusedGateUp, FusedKeyValue, Result, Stream,
};

impl BoundLinear {
    pub(in crate::engine) fn fuse_gate_up(
        &self,
        up: &Self,
        stream: &Stream,
    ) -> Result<Option<FusedGateUp>> {
        match (self, up) {
            (Self::Affine(gate), Self::Affine(up)) => gate.fuse_gate_up(up, stream),
            (Self::MxFp4(gate), Self::MxFp4(up)) => gate.fuse_gate_up(up, stream),
            _ => Ok(None),
        }
    }

    pub(in crate::engine) fn fused_gate_up_bytes(&self, up: &Self) -> Result<Option<usize>> {
        match (self, up) {
            (Self::Affine(gate), Self::Affine(up)) => gate.fused_gate_up_bytes(up),
            (Self::MxFp4(gate), Self::MxFp4(up)) => gate.fused_gate_up_bytes(up),
            _ => Ok(None),
        }
    }

    pub(in crate::engine) fn fuse_expert_gate_up(
        &self,
        up: &Self,
        stream: &Stream,
    ) -> Result<Option<FusedExpertGateUp>> {
        match (self, up) {
            (Self::Affine(gate), Self::Affine(up)) => gate.fuse_expert_gate_up(up, stream),
            _ => Ok(None),
        }
    }

    pub(in crate::engine) fn fused_expert_gate_up_bytes(&self, up: &Self) -> Result<Option<usize>> {
        match (self, up) {
            (Self::Affine(gate), Self::Affine(up)) => gate.fused_expert_gate_up_bytes(up),
            _ => Ok(None),
        }
    }

    pub(in crate::engine) fn fuse_attention(
        &self,
        key: &Self,
        value: Option<&Self>,
        stream: &Stream,
    ) -> Result<Option<FusedAttention>> {
        match (self, key, value) {
            (Self::Affine(query), Self::Affine(key), Some(Self::Affine(value))) => {
                query.fuse_attention(key, Some(value), stream)
            },
            (Self::Affine(query), Self::Affine(key), None) => {
                query.fuse_attention(key, None, stream)
            },
            _ => Ok(None),
        }
    }

    pub(in crate::engine) fn fuse_key_value(
        &self,
        value: Option<&Self>,
        stream: &Stream,
    ) -> Result<Option<FusedKeyValue>> {
        match (self, value) {
            (Self::Affine(key), Some(Self::Affine(value))) => {
                key.fuse_key_value(Some(value), stream)
            },
            _ => Ok(None),
        }
    }
}
