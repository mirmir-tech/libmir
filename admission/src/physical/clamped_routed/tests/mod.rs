use foundation::model::BackendTarget;
use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    gguf::GgmlType,
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{
        AffineBits, AffineGroupAxis, AffinePacking, AffineParameterDType, AffineSignedness,
        AffineStorageDType, AffineZeroPointMode, BindingTransform, BlockQuantization,
        ExpertProjectionRole, GgmlBlockQuantization, GgmlCompanion, GroupedAffineQuantization,
        LayerTensorRole, LogicalTensorRole, TensorBinding, TensorCatalog, TensorPacking,
        TensorStorage, WeightBindingPlan,
    },
};
use serde_json::json;

use super::assess;
use crate::physical::{AdmissionStatus, CheckSubject, EncodingKind};

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

mod layouts;

const EXPERTS: usize = 4;
const BACKENDS: [BackendTarget; 2] = [BackendTarget::Metal, BackendTarget::Cuda];

type Layer = Vec<(ExpertProjectionRole, TensorStorage)>;

fn gpt_oss(layers: &[Layer]) -> TestResult<(TaskExecutionPlan, DecoderExecutionContract)> {
    let decoder = DecoderConfig::from_value(&json!({
        "model_type": "gpt_oss",
        "hidden_size": 64,
        "intermediate_size": 32,
        "num_hidden_layers": layers.len(),
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "swiglu_limit": 7.0
    }))?;
    let semantic = SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: vec![] })?;
    let tensors = layers
        .iter()
        .enumerate()
        .flat_map(|(index, layer)| {
            layer
                .iter()
                .map(move |(projection, storage)| expert(index, *projection, storage))
        })
        .collect();
    let bindings = WeightBindingPlan { tensors };
    Ok((
        TaskExecutionPlan::Generation { decoder },
        DecoderExecutionContract { semantic, bindings },
    ))
}

fn expert(
    index: usize,
    projection: ExpertProjectionRole,
    storage: &TensorStorage,
) -> TensorBinding {
    let mut transforms = vec![BindingTransform::StackedExperts { count: EXPERTS }];
    if projection == ExpertProjectionRole::GateUp {
        transforms.push(BindingTransform::FusedGateUp { interleaved: true });
    }
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index,
            tensor: LayerTensorRole::ExpertProjection { expert: None, projection },
        },
        source: format!("model.layers.{index}.mlp.experts.{projection:?}"),
        shape: Vec::new(),
        logical_shape: None,
        transforms,
        storage: storage.clone(),
    }
}

fn separate(gate: &TensorStorage, up: &TensorStorage, down: &TensorStorage) -> Layer {
    vec![
        (ExpertProjectionRole::Gate, gate.clone()),
        (ExpertProjectionRole::Up, up.clone()),
        (ExpertProjectionRole::Down, down.clone()),
    ]
}

fn uniform(storage: &TensorStorage) -> Layer {
    separate(storage, storage, storage)
}

fn ggml(ggml_type: GgmlType) -> TensorStorage {
    TensorStorage::Ggml {
        format: GgmlBlockQuantization { ggml_type },
        companion: GgmlCompanion::Bias("bias".into()),
    }
}

fn dense(dtype: &str) -> TensorStorage {
    TensorStorage::Dense {
        dtype: models::weights::DenseDType::parse(dtype),
        bias: Some("bias".into()),
    }
}

fn block(format: BlockQuantization, packing: TensorPacking) -> TensorStorage {
    TensorStorage::BlockQuantized {
        format,
        scales: "scales".into(),
        global_scale: None,
        input_scale: None,
        bias: Some("bias".into()),
        packing,
    }
}

fn affine() -> TensorStorage {
    TensorStorage::AffineQuantized {
        format: GroupedAffineQuantization {
            bits: AffineBits::Four,
            group_size: 32,
            group_axis: AffineGroupAxis::Input,
            signedness: AffineSignedness::Unsigned,
            zero_point: AffineZeroPointMode::None,
            packing: AffinePacking::Mlx,
            storage_dtype: AffineStorageDType::U32,
            scale_dtype: AffineParameterDType::F16,
            bias_dtype: None,
        },
        scales: "scales".into(),
        biases: None,
        output_bias: Some("bias".into()),
    }
}

fn refused(layers: &[Layer], backend: BackendTarget) -> TestResult<Option<(EncodingKind, String)>> {
    let (task, contract) = gpt_oss(layers)?;
    Ok(assess(backend, Some(&contract), &task).map(|check| {
        assert_eq!(check.status(), AdmissionStatus::Unsupported, "{check:?}");
        let CheckSubject::Encoding { kind, .. } = check.subject else {
            unreachable!("clamped-routed checks inspect encodings: {check:?}");
        };
        (kind, check.detail)
    }))
}

#[test]
fn admits_the_uniform_expert_banks_each_loader_reads() -> TestResult<()> {
    let ocp = vec![
        (
            ExpertProjectionRole::GateUp,
            block(BlockQuantization::MXFP4, TensorPacking::InterleavedGateUp),
        ),
        (
            ExpertProjectionRole::Down,
            block(BlockQuantization::MXFP4, TensorPacking::Separate),
        ),
    ];
    let checkpoints = [
        ("GGUF MXFP4", uniform(&ggml(GgmlType::MXFP4))),
        ("OpenAI SafeTensors", ocp),
        ("MLX", uniform(&affine())),
        ("BF16", uniform(&dense("BF16"))),
    ];
    for (checkpoint, layer) in checkpoints {
        for backend in BACKENDS {
            let check = refused(&[layer.clone(), layer.clone()], backend)?;
            assert!(check.is_none(), "{checkpoint} on {backend:?}: {check:?}");
        }
    }
    Ok(())
}

#[test]
fn refuses_ggml_expert_banks_other_than_mxfp4() -> TestResult<()> {
    for backend in BACKENDS {
        let check = refused(&[uniform(&ggml(GgmlType::Q4_K))], backend)?;
        assert!(
            check.as_ref().is_some_and(|(kind, detail)| *kind == EncodingKind::Ggml
                && detail.contains("MXFP4")
                && detail.contains("Q4_K")),
            "{backend:?}: {check:?}"
        );
    }
    Ok(())
}

/// The reviewer's catalog: GPT-OSS MXFP4 with only the first layer's down
/// bank dense, which neither loader assembles beside GGML gate and up banks.
#[test]
fn refuses_a_layer_mixing_ggml_gate_and_up_with_a_dense_down_bank() -> TestResult<()> {
    let mxfp4 = ggml(GgmlType::MXFP4);
    let mixed = separate(&mxfp4, &mxfp4, &dense("F32"));
    for backend in BACKENDS {
        let check = refused(&[mixed.clone(), uniform(&mxfp4)], backend)?;
        assert!(
            check.as_ref().is_some_and(|(kind, detail)| *kind == EncodingKind::Dense
                && detail.contains("layer 0 mixes")
                && detail.contains("gate GGML MXFP4")
                && detail.contains("down Dense F32")),
            "{backend:?}: {check:?}"
        );
    }
    Ok(())
}
