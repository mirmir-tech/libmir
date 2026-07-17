use mirtal::{Array, Compiled, DType, Graph, Quantization, Quantized, Result, Stream};

use super::moe::route_top_k;

#[mirtal::compiled(shapeless)]
fn geglu(graph: Graph<'_>, [gate, input]: [Array; 2]) -> Result<[Array; 1]> {
    let cube = graph.power_scalar(&gate, 3.0)?;
    let correction = graph.multiply_scalar(&cube, 0.044_715)?;
    let inner = graph.add(&gate, &correction)?;
    let scaled = graph.multiply_scalar(&inner, 0.797_884_6)?;
    let activation = graph.add_scalar(&graph.tanh(&scaled)?, 1.0)?;
    let gated = graph.multiply(&graph.multiply_scalar(&gate, 0.5)?, &activation)?;
    Ok([graph.multiply(&gated, &input)?])
}

#[mirtal::compiled(shapeless)]
fn swiglu(graph: Graph<'_>, [gate, input]: [Array; 2]) -> Result<[Array; 1]> {
    Ok([graph.multiply(&graph.silu(&gate)?, &input)?])
}

#[mirtal::compiled(shapeless)]
fn precise_swiglu(graph: Graph<'_>, [reference, gate, input]: [Array; 3]) -> Result<[Array; 1]> {
    let gate = graph.astype(&gate, DType::Float32)?;
    let gate = graph.multiply(&gate, &graph.sigmoid(&gate)?)?;
    let input = graph.astype(&input, DType::Float32)?;
    let output = graph.multiply(&gate, &input)?;
    Ok([graph.astype(&output, reference.dtype()?)?])
}

#[mirtal::compiled(shapeless)]
fn logit_softcap(graph: Graph<'_>, [input, cap]: [Array; 2]) -> Result<[Array; 1]> {
    let scaled = graph.divide(&input, &cap)?;
    Ok([graph.multiply(&graph.tanh(&scaled)?, &cap)?])
}

#[mirtal::compiled]
fn affine_router(
    graph: Graph<'_>,
    [input, norm_scale, weight, scales, biases, expert_scale]: [Array; 6],
) -> Result<[Array; 2]> {
    let normalized = graph.rms_norm(&input, &norm_scale, 1.0e-6)?;
    let normalized = graph.astype(&normalized, input.dtype()?)?;
    let scores = graph.quantized_matmul(
        &normalized,
        Quantized {
            weight: &weight,
            scales: &scales,
            biases: &biases,
            format: Quantization::new(64, 8)?,
        },
        true,
    )?;
    let scores = graph.astype(&scores, input.dtype()?)?;
    route_top_k(graph, &scores, Some(&expert_scale), 8)
}

#[derive(Debug)]
pub(super) struct CompiledGraphs {
    geglu: Compiled<2, 1>,
    swiglu: Compiled<2, 1>,
    precise_swiglu: Compiled<3, 1>,
    logit_softcap: Compiled<2, 1>,
    router: Compiled<6, 2>,
}

impl CompiledGraphs {
    pub(super) fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            geglu: compile_geglu(stream)?,
            swiglu: compile_swiglu(stream)?,
            precise_swiglu: compile_precise_swiglu(stream)?,
            logit_softcap: compile_logit_softcap(stream)?,
            router: compile_affine_router(stream)?,
        })
    }

    pub(super) fn geglu(&self, gate: &Array, input: &Array, stream: &Stream) -> Result<Array> {
        let [output] = self.geglu.call(stream, [gate, input])?;
        Ok(output)
    }

    pub(super) fn swiglu(&self, gate: &Array, input: &Array, stream: &Stream) -> Result<Array> {
        let [output] = self.swiglu.call(stream, [gate, input])?;
        Ok(output)
    }

    pub(super) fn precise_swiglu(
        &self,
        reference: &Array,
        gate: &Array,
        input: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let [output] = self.precise_swiglu.call(stream, [reference, gate, input])?;
        Ok(output)
    }

    pub(super) fn logit_softcap(
        &self,
        input: &Array,
        cap: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let [output] = self.logit_softcap.call(stream, [input, cap])?;
        Ok(output)
    }

    pub(super) fn router(&self, inputs: [&Array; 6], stream: &Stream) -> Result<[Array; 2]> {
        self.router.call(stream, inputs)
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::{Array, Result, Stream};

    #[test]
    fn applies_sigmoid_gate_on_the_explicit_gpu_stream() -> Result<()> {
        let stream = Stream::new_gpu()?;
        let gate = Array::from_f32(&[0.0], &[1])?;
        let input = Array::from_f32(&[4.0], &[1])?;
        let output = gate.sigmoid_mul(&input, &stream)?;

        assert_eq!(output.to_vec_f32_on_stream(&stream)?, vec![2.0]);
        Ok(())
    }

    #[test]
    fn executes_cached_activation_and_softcap_graphs() -> Result<()> {
        let stream = Stream::new_gpu()?;
        let gate = Array::from_f32(&[1.0, 0.0], &[2])?;
        let input = Array::from_f32(&[2.0, 4.0], &[2])?;

        let geglu = gate.gelu_approx_mul(&input, &stream)?.to_vec_f32_on_stream(&stream)?;
        let swiglu = gate.silu_mul(&input, &stream)?.to_vec_f32_on_stream(&stream)?;
        let softcap = Array::from_f32(&[20.0, -20.0], &[2])?
            .logit_softcap(10.0, &stream)?
            .to_vec_f32_on_stream(&stream)?;

        let expected_geglu = approximate_gelu(1.0) * 2.0;
        assert!((geglu[0] - expected_geglu).abs() < 1.0e-5);
        assert!(geglu[1].abs() < f32::EPSILON);
        assert!((swiglu[0] - 1.462_117_2).abs() < 1.0e-5);
        assert!(swiglu[1].abs() < f32::EPSILON);
        assert!((softcap[0] - 9.640_276).abs() < 1.0e-5);
        assert!((softcap[1] + 9.640_276).abs() < 1.0e-5);
        Ok(())
    }

    #[test]
    fn preserves_bfloat16_geglu_reference_value() -> Result<()> {
        let stream = Stream::new_gpu()?;
        let gate_values =
            (0_u16..4_096).map(|index| (f32::from(index) * 0.013).sin()).collect::<Vec<_>>();
        let input_values =
            (0_u16..4_096).map(|index| (f32::from(index) * 0.017).cos()).collect::<Vec<_>>();
        let gate = Array::from_f32(&gate_values, &[1, 1, 4_096])?;
        let input = Array::from_f32(&input_values, &[1, 1, 4_096])?;
        let gate = as_bfloat16(&gate, &stream)?;
        let input = as_bfloat16(&input, &stream)?;
        let candidate = as_float32(
            &Array::from_native(stream.gelu_approx_mul(gate.native(), input.native())?)?
                .astype_like(&gate, &stream)?,
            &stream,
        )?
        .to_vec_f32_on_stream(&stream)?;

        assert!((candidate[3_516] + 0.820_312_5).abs() < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn preserves_float32_precise_swiglu_arithmetic() -> Result<()> {
        let stream = Stream::new_gpu()?;
        let reference = as_bfloat16(&Array::from_f32(&[0.0], &[1])?, &stream)?;
        let gate = as_bfloat16(&Array::from_f32(&[0.010_314_941], &[1])?, &stream)?;
        let input = as_bfloat16(&Array::from_f32(&[-2.968_75], &[1])?, &stream)?;
        let compiled = super::compile_precise_swiglu(stream.native())?;
        let [output] =
            compiled.call(stream.native(), [reference.native(), gate.native(), input.native()])?;
        let precise = stream.native().read::<f32>(&output)?;
        let ordinary = gate.silu_mul(&input, &stream)?.to_vec_f32_on_stream(&stream)?;

        assert_eq!(output.dtype()?, mirtal::DType::Bfloat16);
        assert_eq!(precise, vec![-0.015_380_859]);
        assert_eq!(ordinary, vec![-0.015_319_824]);
        Ok(())
    }

    fn as_bfloat16(input: &Array, stream: &Stream) -> Result<Array> {
        Array::from_native(stream.native().graph().astype(input.native(), mirtal::DType::Bfloat16)?)
    }

    fn as_float32(input: &Array, stream: &Stream) -> Result<Array> {
        Array::from_native(stream.native().graph().astype(input.native(), mirtal::DType::Float32)?)
    }

    fn approximate_gelu(value: f32) -> f32 {
        let corrected = 0.044_715_f32.mul_add(value.powi(3), value);
        let inner = 0.797_884_6 * corrected;
        0.5 * value * (1.0 + inner.tanh())
    }
}
