use models::weights::{
    BlockQuantization, ExpertProjectionRole, LayerSlot, LayerTensorRole, RoutedExpertBindings,
    TensorBinding, TensorPacking, TensorStorage,
};

use super::{IndividualBank, RoutedExpertLoader};
use crate::metal::MetalDecoderRuntime;

fn dense() -> TensorStorage {
    TensorStorage::Dense {
        dtype: models::weights::DenseDType::Bf16,
        bias: None,
    }
}

fn nvfp4() -> TensorStorage {
    TensorStorage::BlockQuantized {
        format: BlockQuantization::NVFP4,
        scales: "scales".into(),
        global_scale: None,
        input_scale: None,
        bias: None,
        packing: TensorPacking::Separate,
    }
}

fn expert(
    expert: usize,
    projection: ExpertProjectionRole,
    storage: TensorStorage,
) -> TensorBinding {
    TensorBinding {
        role: LayerSlot::Decoder(0)
            .role(LayerTensorRole::ExpertProjection { expert: Some(expert), projection }),
        source: format!("experts.{expert}.{projection:?}"),
        shape: vec![8, 16],
        logical_shape: None,
        transforms: Vec::new(),
        storage,
    }
}

fn individual(storage: &dyn Fn(usize) -> TensorStorage) -> Vec<TensorBinding> {
    (0..2)
        .flat_map(|index| {
            [ExpertProjectionRole::Gate, ExpertProjectionRole::Up, ExpertProjectionRole::Down]
                .map(|projection| expert(index, projection, storage(index)))
        })
        .collect()
}

fn experts(tensors: &[TensorBinding]) -> RoutedExpertBindings<'_> {
    RoutedExpertBindings::Individual { tensors, layer: LayerSlot::Decoder(0) }
}

/// Per-expert banks stack dense, NVFP4 and packed-integer storage, each
/// bank in one storage.
#[test]
fn an_individual_bank_is_one_storage() {
    let bank = |tensors: &[TensorBinding]| {
        IndividualBank::of(&experts(tensors).individual(ExpertProjectionRole::Up))
    };
    assert_eq!(bank(&individual(&|_| dense())), Some(IndividualBank::Dense));
    assert_eq!(bank(&individual(&|_| nvfp4())), Some(IndividualBank::NvFp4));
    assert_eq!(
        bank(&individual(&|index| if index == 0 {
            dense()
        } else {
            nvfp4()
        })),
        None
    );
    assert_eq!(IndividualBank::of(&[]), None);
}

/// Dense experts stored per expert run where their runtime stacks them,
/// and are refused by GPT-OSS's clamped loader, which reads stacked experts
/// only; ungated stacked experts run only on the state-space hybrid.
#[test]
fn each_runtime_admits_the_expert_layouts_its_loader_reads() {
    let tensors = individual(&|_| dense());
    let gated = RoutedExpertLoader::of(MetalDecoderRuntime::Dense);
    assert_eq!(gated, Some(RoutedExpertLoader::Gated));
    assert!(gated.is_some_and(|loader| loader.loads(experts(&tensors))));
    let clamped = RoutedExpertLoader::of(MetalDecoderRuntime::ClampedRouted);
    assert!(clamped.is_some_and(|loader| !loader.loads(experts(&tensors))));
    let ungated = expert(0, ExpertProjectionRole::Up, dense());
    let down = expert(0, ExpertProjectionRole::Down, dense());
    let stacked = RoutedExpertBindings::UngatedStacked { up: &ungated, down: &down };
    assert!(!RoutedExpertLoader::Gated.loads(stacked));
    assert!(!RoutedExpertLoader::GatedStacked.loads(stacked));
    assert!(RoutedExpertLoader::Ungated.loads(stacked));
    let state_space = RoutedExpertLoader::of(MetalDecoderRuntime::StateSpaceHybrid);
    assert!(state_space.is_some_and(|loader| loader.loads(experts(&tensors))));
    assert_eq!(RoutedExpertLoader::of(MetalDecoderRuntime::DenseAndRouted), None);
}
