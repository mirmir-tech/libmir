use super::*;
use crate::engine::gated_delta::layer::llama_cpp as layout;

fn tiled() -> Result<layout::TiledValues> {
    let config = GatedDeltaLayerConfig::from_linear_attention(
        &LinearAttentionConfig {
            convolution_kernel_size: 2,
            key_heads: 2,
            value_heads: 4,
            key_head_dim: 2,
            value_head_dim: 2,
            layout: LinearAttentionLayout::LlamaCpp,
        },
        1.0e-6,
    )?;
    layout::TiledValues::new(config)?
        .ok_or_else(|| Error::InvalidModel("more value than key heads tile".into()))
}

#[test]
fn grouped_value_rows_follow_the_leading_rows() -> Result<()> {
    let stream = Stream::new_gpu()?;
    // Stored head `t` is grouped head `(t % 2) * 2 + t / 2`, so grouped heads
    // 0..4 come from stored heads 0, 2, 1, 3 after four leading rows.
    let rows = tiled()?.rows(4, 2)?;
    assert_eq!(rows.to_vec_u32(&stream)?, [0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11]);
    Ok(())
}

#[test]
fn tiling_inverts_the_grouped_value_order() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let grouped = Array::from_f32(&[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], &[1, 1, 8])?;
    let output = tiled()?.tile(&grouped, &stream)?;
    assert_eq!(output.to_vec_f32(&stream)?, [0.0, 0.5, 2.0, 2.5, 1.0, 1.5, 3.0, 3.5]);
    Ok(())
}

#[test]
fn decay_log_recovers_a_log() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let stored = Array::from_f32(&[-1.0, -std::f32::consts::E], &[2])?;
    let restored = decay_log(&stored, &stream)?.to_vec_f32(&stream)?;
    assert!(restored[0].abs() < 1.0e-6 && (restored[1] - 1.0).abs() < 1.0e-6);
    assert!(decay_log(&Array::from_f32(&[0.5], &[1])?, &stream).is_err());
    Ok(())
}
