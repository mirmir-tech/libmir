use super::*;

mirtal::metal_kernel! {
    pub(in crate::engine) fn packed_decode {
        name: "mirmir_gdn_four_lane_fused_decode",
        templates: [
            InT: dtype = bf16, StT: dtype = f32, DK: int = 128, DV: int = 128,
            HK: int = 8, HV: int = 16, NORMALIZE: bool = true,
        ],
        inputs: [query: InT, key: InT, value: InT, alpha: float, beta: float,
            a_log: float, dt_bias: float, state: StT],
        outputs: [output: InT, next_state: StT],
        source: file "kernels/gated_delta/decode.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

pub(in crate::engine) fn dispatch(inputs: [&Array; 8], normalize: bool) -> Result<Dispatch> {
    let q = inputs[0].shape()?.dimensions().to_vec();
    let v = inputs[2].shape()?.dimensions().to_vec();
    super::super::validate_decode(inputs, &q, &v)?;
    assert!(q[3] == 128 && v[3].is_multiple_of(8));
    Ok(Dispatch::new([32, v[3] / 8, v[0] * v[2]], [32, 2, 1]).templates([
        TemplateArg::dtype("InT", inputs[0].dtype()?),
        TemplateArg::dtype("StT", DType::Float32),
        template("DK", q[3])?,
        template("DV", v[3])?,
        template("HK", q[2])?,
        template("HV", v[2])?,
        TemplateArg::bool("NORMALIZE", normalize),
    ]))
}
