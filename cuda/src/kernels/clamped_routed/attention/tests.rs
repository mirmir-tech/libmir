use super::ClampedRoutedAttention;

#[test]
fn launch_uses_one_warp_only_for_paired_head_64() -> crate::Result<()> {
    assert_eq!(ClampedRoutedAttention::launch(64, 64)?.block, (32, 1, 1));
    assert_eq!(ClampedRoutedAttention::launch(64, 128)?.block, (128, 1, 1));
    assert_eq!(ClampedRoutedAttention::launch(64, 256)?.block, (256, 1, 1));
    Ok(())
}
