use std::{
    fs,
    path::{Path, PathBuf},
};

use super::{FP4, NF4, validate};
use crate::{
    error::Result,
    weights::{
        BitsAndBytes4BitQuantization, BitsAndBytes4BitType, BitsAndBytesComputeDType,
        BitsAndBytesStorageDType, LogicalTensorRole, TensorBinding, TensorCatalog, TensorInfo,
        TensorStorage,
    },
};

#[test]
fn admits_direct_fp4_contract() -> Result<()> {
    let fixture = Fixture::new(false, BitsAndBytes4BitType::Fp4, false)?;
    validate(&fixture.binding, &[2, 64], &fixture.catalog)?;
    fixture.remove()
}

#[test]
fn admits_nested_nf4_contract() -> Result<()> {
    let fixture = Fixture::new(true, BitsAndBytes4BitType::Nf4, false)?;
    validate(&fixture.binding, &[2, 64], &fixture.catalog)?;
    fixture.remove()
}

#[test]
fn rejects_modified_codebook() -> Result<()> {
    let fixture = Fixture::new(false, BitsAndBytes4BitType::Fp4, true)?;
    assert!(validate(&fixture.binding, &[2, 64], &fixture.catalog).is_err());
    fixture.remove()
}

struct Fixture {
    path: PathBuf,
    binding: TensorBinding,
    catalog: TensorCatalog,
}

impl Fixture {
    fn new(nested: bool, kind: BitsAndBytes4BitType, corrupt_map: bool) -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "libmir-models-bnb4-{}-{}-{}-{}.bin",
            kind.as_str(),
            nested,
            corrupt_map,
            std::process::id()
        ));
        let mut bytes = vec![0_u8; 64];
        let weight_end = bytes.len();
        if nested {
            bytes.extend([0_u8, 1]);
        } else {
            append_f32(&mut bytes, &[2.0, 3.0]);
        }
        let absmax_end = bytes.len();
        let map = if kind == BitsAndBytes4BitType::Fp4 {
            &FP4
        } else {
            &NF4
        };
        for bits in map {
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        if corrupt_map {
            bytes[absmax_end] ^= 1;
        }
        let map_end = bytes.len();
        let state = b"{}";
        bytes.extend_from_slice(state);
        let state_end = bytes.len();
        if nested {
            append_f32(&mut bytes, &[1.0]);
        }
        let nested_absmax_end = bytes.len();
        if nested {
            append_f32(&mut bytes, &[0.0; 256]);
        }
        let end = bytes.len();
        fs::write(&path, bytes)?;
        let mut tensors = vec![
            tensor("weight", &path, "U8", vec![64, 1], 0, weight_end),
            tensor(
                "absmax",
                &path,
                if nested {
                    "U8"
                } else {
                    "F32"
                },
                vec![2],
                weight_end,
                absmax_end,
            ),
            tensor("quant_map", &path, "F32", vec![16], absmax_end, map_end),
            tensor("quant_state", &path, "U8", vec![state.len()], map_end, state_end),
        ];
        if nested {
            tensors.extend([
                tensor("nested_absmax", &path, "F32", vec![1], state_end, nested_absmax_end),
                tensor("nested_quant_map", &path, "F32", vec![256], nested_absmax_end, end),
            ]);
        }
        Ok(Self {
            path,
            binding: binding(nested, kind),
            catalog: TensorCatalog::new(tensors),
        })
    }

    fn remove(self) -> Result<()> {
        fs::remove_file(self.path)?;
        Ok(())
    }
}

fn binding(nested: bool, kind: BitsAndBytes4BitType) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Output,
        source: "weight".into(),
        shape: vec![64, 1],
        logical_shape: Some(vec![2, 64]),
        transforms: Vec::new(),
        storage: TensorStorage::BitsAndBytes4Bit {
            format: BitsAndBytes4BitQuantization {
                quant_type: kind,
                block_size: 64,
                compute_dtype: BitsAndBytesComputeDType::Bf16,
                storage_dtype: BitsAndBytesStorageDType::U8,
                nested_block_size: nested.then_some(256),
            },
            absmax: "absmax".into(),
            quant_map: "quant_map".into(),
            nested_absmax: nested.then(|| "nested_absmax".into()),
            nested_quant_map: nested.then(|| "nested_quant_map".into()),
            quant_state: "quant_state".into(),
            nested_offset_bits: nested.then(|| 0.0_f32.to_bits()),
        },
    }
}

fn tensor(
    name: &str,
    path: &Path,
    dtype: &str,
    shape: Vec<usize>,
    start: usize,
    end: usize,
) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: path.into(),
        dtype: dtype.into(),
        shape,
        data_start: 0,
        data_offsets: [start as u64, end as u64],
    }
}

fn append_f32(output: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
}
