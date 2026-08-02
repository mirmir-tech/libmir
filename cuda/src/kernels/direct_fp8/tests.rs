use super::{DirectFp8Activation, DirectFp8Format, DirectFp8Scale, DirectFp8Spec};

#[test]
fn accepts_exact_and_explicitly_padded_block_grids() {
    let activation = DirectFp8Activation::Bf16;
    assert!(DirectFp8Spec::new(1, 128, 64, DirectFp8Scale::Tensor, false, activation).is_ok());
    assert!(
        DirectFp8Spec::new(
            1,
            128,
            64,
            DirectFp8Scale::BlockGrid {
                output_groups: 2,
                input_groups: 4,
                output_block_size: 32,
                input_block_size: 32,
            },
            true,
            activation,
        )
        .is_ok()
    );
    assert!(
        DirectFp8Spec::new(
            1,
            128,
            64,
            DirectFp8Scale::BlockGrid {
                output_groups: 3,
                input_groups: 4,
                output_block_size: 32,
                input_block_size: 32,
            },
            false,
            activation,
        )
        .is_err()
    );
    assert!(
        DirectFp8Spec::new(
            1,
            128,
            64,
            DirectFp8Scale::BlockGrid {
                output_groups: 2,
                input_groups: 3,
                output_block_size: 32,
                input_block_size: 32,
            },
            false,
            activation,
        )
        .is_err()
    );
    assert!(
        DirectFp8Spec::new(
            1,
            132,
            65,
            DirectFp8Scale::BlockGrid {
                output_groups: 3,
                input_groups: 2,
                output_block_size: 32,
                input_block_size: 128,
            },
            false,
            activation,
        )
        .is_ok()
    );
}

#[test]
fn keeps_dynamic_e4m3_activation_distinct_from_e5m2_weights() {
    assert!(
        DirectFp8Spec::new_with_format(
            DirectFp8Format::E5M2,
            1,
            128,
            64,
            DirectFp8Scale::Tensor,
            false,
            DirectFp8Activation::DynamicE4M3Token,
        )
        .is_err()
    );
}
