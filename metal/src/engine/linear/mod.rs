use super::{
    Array, Error, FusedAttention, FusedExpertGateUp, FusedGateUp, FusedKeyValue, ModelTensors,
    QuantizedArrays, Result, RouterOutput, Stream,
};

mod gptq;
mod quantization;
pub(in crate::engine) use gptq::GptqLinear;
pub(super) use quantization::infer_bits;

#[derive(Debug)]
pub struct QuantizedLinear {
    arrays: QuantizedArrays,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
}
impl QuantizedLinear {
    pub(in crate::engine) fn from_quantized(
        arrays: QuantizedArrays,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self { arrays, bias: None, group_size, bits }
    }

    pub fn load(tensors: &ModelTensors, prefix: &str, group_size: i32) -> Result<Self> {
        Self::load_names(
            tensors,
            &format!("{prefix}.weight"),
            &format!("{prefix}.scales"),
            &format!("{prefix}.biases"),
            Some(&format!("{prefix}.bias")),
            group_size,
        )
    }

    pub(in crate::engine) fn load_names(
        tensors: &ModelTensors,
        weight: &str,
        scales: &str,
        biases: &str,
        output_bias: Option<&str>,
        group_size: i32,
    ) -> Result<Self> {
        let weight = tensors.get(weight)?;
        let scales = tensors.get(scales)?;
        let biases = tensors.get(biases)?;
        let bits = infer_bits(&weight, &scales, group_size)?;
        let arrays = QuantizedArrays::new(weight, scales, biases, group_size, bits)?;
        let affine_bias = output_bias.map(|name| tensors.get_optional(name)).transpose()?.flatten();
        Ok(Self {
            arrays,
            bias: affine_bias,
            group_size,
            bits,
        })
    }

    pub fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let output =
            input.quantized_matmul(&self.arrays, true, stream)?.astype_like(input, stream)?;
        match self.bias.as_ref() {
            Some(bias) => output.add(bias, stream),
            None => Ok(output),
        }
    }

    pub fn gather(
        &self,
        input: &Array,
        indices: &Array,
        sorted_indices: bool,
        stream: &Stream,
    ) -> Result<Array> {
        input
            .gather_qmm(
                &self.arrays,
                indices,
                mirtal::GatherQmmOptions { transpose: true, sorted_indices },
                stream,
            )?
            .astype_like(input, stream)
    }

    pub fn route(
        &self,
        input: &Array,
        norm_scale: &Array,
        expert_scale: &Array,
        eps: f32,
        top_k: i32,
        stream: &Stream,
    ) -> Result<RouterOutput> {
        if eps.to_bits() == 1.0e-6_f32.to_bits()
            && self.group_size == 64
            && self.bits == 8
            && top_k == 8
        {
            let [weight, scales, biases] = self.arrays.native_components();
            let [indices, weights] = stream.affine_router([
                input.native(),
                norm_scale.native(),
                weight,
                scales,
                biases,
                expert_scale.native(),
            ])?;
            return Ok(RouterOutput {
                indices: Array::from_native(indices)?,
                weights: Array::from_native(weights)?,
            });
        }
        let normalized = input.rms_norm(norm_scale, eps, stream)?;
        let scores = normalized
            .quantized_matmul(&self.arrays, true, stream)?
            .astype_like(input, stream)?;
        scores.router_top_k(expert_scale, top_k, stream)
    }

    #[must_use]
    pub fn bits(&self) -> i32 {
        self.bits
    }

    #[must_use]
    pub const fn group_size(&self) -> i32 {
        self.group_size
    }

    #[must_use]
    pub(super) const fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    pub(super) fn graph_parts(&self) -> Result<(mirtal::QuantizedArrays, Option<mirtal::Array>)> {
        let [weight, scales, biases] = self.arrays.native_components().map(Clone::clone);
        Ok((
            mirtal::QuantizedArrays {
                weight,
                scales,
                biases,
                format: mirtal::Quantization::new(self.group_size, self.bits)?,
            },
            self.bias.as_ref().map(|bias| bias.native().clone()),
        ))
    }

    pub(super) fn fuse_gate_up(&self, up: &Self, stream: &Stream) -> Result<Option<FusedGateUp>> {
        if self.group_size != up.group_size || self.bits != up.bits {
            return Ok(None);
        }
        FusedGateUp::new(&self.arrays, &up.arrays, self.group_size, self.bits, stream).map(Some)
    }

    pub(super) fn fused_gate_up_bytes(&self, up: &Self) -> Result<Option<usize>> {
        self.fused_pair_bytes(up)
    }

    pub(super) fn fuse_expert_gate_up(
        &self,
        up: &Self,
        stream: &Stream,
    ) -> Result<Option<FusedExpertGateUp>> {
        if self.group_size != up.group_size || self.bits != up.bits {
            return Ok(None);
        }
        FusedExpertGateUp::new(&self.arrays, &up.arrays, self.group_size, self.bits, stream)
            .map(Some)
    }

    pub(super) fn fused_expert_gate_up_bytes(&self, up: &Self) -> Result<Option<usize>> {
        self.fused_pair_bytes(up)
    }

    fn fused_pair_bytes(&self, up: &Self) -> Result<Option<usize>> {
        if self.group_size != up.group_size || self.bits != up.bits {
            return Ok(None);
        }
        let arrays = [&self.arrays, &up.arrays];
        arrays.into_iter().try_fold(Some(0_usize), |total, arrays| {
            let bytes =
                [arrays.weight.byte_len()?, arrays.scales.byte_len()?, arrays.biases.byte_len()?]
                    .into_iter()
                    .try_fold(0_usize, |total, bytes| {
                        total.checked_add(bytes).ok_or(Error::ShapeOverflow)
                    })?;
            total
                .and_then(|total| total.checked_add(bytes))
                .map(Some)
                .ok_or(Error::ShapeOverflow)
        })
    }

    pub(super) fn fuse_attention(
        &self,
        key: &Self,
        value: Option<&Self>,
        stream: &Stream,
    ) -> Result<Option<FusedAttention>> {
        if self.group_size != key.group_size
            || self.bits != key.bits
            || value
                .is_some_and(|value| self.group_size != value.group_size || self.bits != value.bits)
        {
            return Ok(None);
        }
        FusedAttention::new(
            &self.arrays,
            &key.arrays,
            value.map(|value| &value.arrays),
            self.group_size,
            self.bits,
            stream,
        )
        .map(Some)
    }

    pub(super) fn fuse_key_value(
        &self,
        value: Option<&Self>,
        stream: &Stream,
    ) -> Result<Option<FusedKeyValue>> {
        let Some(value) = value else {
            return Ok(None);
        };
        if self.group_size != value.group_size || self.bits != value.bits {
            return Ok(None);
        }
        FusedKeyValue::new(&self.arrays, &value.arrays, self.group_size, self.bits, stream)
            .map(Some)
    }
}
