//! GSQ compressed-tensors widths: 2- and 3-bit groups whose values straddle
//! I32 word boundaries, F32 scales, and per-expert banks.

use super::*;

const COLUMNS: usize = 256;
const ROWS: usize = 4;
const GROUP: usize = 128;
const EXPERTS: usize = 2;

#[test]
fn reinterprets_two_and_three_bit_groups_as_native_affine() -> Result<()> {
    for bits in [CompressedIntegerBits::Two, CompressedIntegerBits::Three] {
        let root = fixture(bits)?;
        let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
        let stream = Stream::new_gpu()?;
        let linear = BoundLinear::load(&tensors, &binding(bits, 0), &stream)?;
        // The identity selects every column: output[c, r] = W[r, c].
        let mut identity = vec![0.0; COLUMNS * COLUMNS];
        for column in 0..COLUMNS {
            identity[column * COLUMNS + column] = 1.0;
        }
        let input =
            Array::from_f32(&identity, &[i32::try_from(COLUMNS)?, i32::try_from(COLUMNS)?])?
                .astype(crate::engine::Dtype::Bfloat16, &stream)?;
        let output = linear.forward(&input, &stream)?.to_vec_f32(&stream)?;
        for column in 0..COLUMNS {
            for row in 0..ROWS {
                // Small integers times powers of two: exact in BF16.
                assert_eq!(
                    output[column * ROWS + row].to_bits(),
                    weight(bits, 0, row, column).to_bits(),
                    "{bits:?} row {row} column {column}"
                );
            }
        }
        drop(tensors);
        fs::remove_dir_all(root)?;
    }
    Ok(())
}

#[test]
fn stacks_individual_packed_experts_into_a_gathered_bank() -> Result<()> {
    let bits = CompressedIntegerBits::Three;
    let root = fixture(bits)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let bindings = (0..EXPERTS).map(|expert| binding(bits, expert)).collect::<Vec<_>>();
    let refs = bindings.iter().collect::<Vec<_>>();
    let bank = BoundLinear::load_individual_bank(&tensors, &refs, &stream)?;
    let x = (0..COLUMNS)
        .map(|column| f32::from(u8::try_from(column % 4).unwrap_or(0)).mul_add(0.5, -0.75))
        .collect::<Vec<_>>();
    let input =
        Array::from_f32(&[x.clone(), x.clone()].concat(), &[2, 1, i32::try_from(COLUMNS)?])?
            .astype(crate::engine::Dtype::Bfloat16, &stream)?;
    let selected = [1_u32, 0];
    let indices = Array::from_u32(&selected, &[2])?;
    let output = bank.gather(&input, &indices, false, &stream)?.to_vec_f32(&stream)?;
    for (token, &expert) in selected.iter().enumerate() {
        let expert = usize::try_from(expert)?;
        for row in 0..ROWS {
            let expected = (0..COLUMNS)
                .map(|column| x[column] * weight(bits, expert, row, column))
                .sum::<f32>();
            let actual = output[token * ROWS + row];
            assert!(
                (actual - expected).abs() <= expected.abs() / 128.0 + 1.0e-3,
                "token {token}, expert {expert}, row {row}: {actual} != {expected}"
            );
        }
    }
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn quantized(bits: CompressedIntegerBits, expert: usize, row: usize, column: usize) -> i32 {
    let levels = 1_i32 << bits.get();
    i32::try_from((row * 31 + column * 7 + expert * 13 + column / 5) % 64).unwrap_or(0) % levels
        - levels / 2
}

/// Powers of two keep `q * scale` exact in BF16 after the F32-to-BF16 cast.
fn scale(expert: usize, row: usize, group: usize) -> f32 {
    [0.125, 0.0625, 0.25][(row + group + expert) % 3]
}

fn weight(bits: CompressedIntegerBits, expert: usize, row: usize, column: usize) -> f32 {
    let value = i16::try_from(quantized(bits, expert, row, column)).map_or(0.0, f32::from);
    value * scale(expert, row, column / GROUP)
}

/// compressed-tensors `pack_to_int32`: offset-binary values in one dense
/// little-endian bitstream per row.
fn packed_row(bits: CompressedIntegerBits, expert: usize, row: usize) -> Vec<u32> {
    let width = usize::from(bits.get());
    let mut words = vec![0_u32; COLUMNS * width / 32];
    for column in 0..COLUMNS {
        let encoded =
            u64::try_from(quantized(bits, expert, row, column) + (1 << (width - 1))).unwrap_or(0);
        let start = column * width;
        let pair = (encoded << (start % 32)).to_le_bytes();
        words[start / 32] |= u32::from_le_bytes([pair[0], pair[1], pair[2], pair[3]]);
        if start % 32 + width > 32 {
            words[start / 32 + 1] |= u32::from_le_bytes([pair[4], pair[5], pair[6], pair[7]]);
        }
    }
    words
}

fn fixture(bits: CompressedIntegerBits) -> Result<std::path::PathBuf> {
    let root = std::env::temp_dir().join(format!(
        "libmir-metal-gsq-{}-{}-{}",
        bits.get(),
        std::process::id(),
        std::thread::current().name().unwrap_or("test").replace("::", "-")
    ));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    let words = COLUMNS * usize::from(bits.get()) / 32;
    let groups = COLUMNS / GROUP;
    let (mut header, mut payload) = (Vec::new(), Vec::new());
    for expert in 0..EXPERTS {
        let start = payload.len();
        for row in 0..ROWS {
            for word in packed_row(bits, expert, row) {
                payload.extend_from_slice(&word.to_le_bytes());
            }
        }
        header.push(format!(
            r#""experts.{expert}.weight_packed":{{"dtype":"I32","shape":[{ROWS},{words}],"data_offsets":[{start},{}]}}"#,
            payload.len()
        ));
        let start = payload.len();
        for row in 0..ROWS {
            for group in 0..groups {
                payload.extend_from_slice(&scale(expert, row, group).to_le_bytes());
            }
        }
        header.push(format!(
            r#""experts.{expert}.weight_scale":{{"dtype":"F32","shape":[{ROWS},{groups}],"data_offsets":[{start},{}]}}"#,
            payload.len()
        ));
    }
    let mut header = format!("{{{}}}", header.join(","));
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&payload);
    fs::write(root.join("model.safetensors"), data)?;
    Ok(root)
}

fn binding(bits: CompressedIntegerBits, expert: usize) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Output,
        source: format!("experts.{expert}.weight_packed"),
        shape: vec![ROWS, COLUMNS * usize::from(bits.get()) / 32],
        logical_shape: Some(vec![ROWS, COLUMNS]),
        transforms: Vec::new(),
        storage: TensorStorage::PackedInteger {
            format: CompressedIntegerQuantization {
                bits,
                scale_strategy: CompressedIntegerScaleStrategy::Group { group_size: GROUP },
                signedness: CompressedIntegerSignedness::OffsetBinary,
                zero_point: CompressedIntegerZeroPointMode::None,
                activation_order: CompressedIntegerActivationOrder::None,
                packing: CompressedIntegerPacking::DenseLittleEndian,
                storage_dtype: CompressedIntegerStorageDType::I32,
                scale_dtype: CompressedIntegerScaleDType::F32,
            },
            scales: format!("experts.{expert}.weight_scale"),
            shape: format!("experts.{expert}.weight_shape"),
            zero_points: None,
            group_indices: None,
        },
    }
}
