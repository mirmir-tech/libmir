use super::AffineGemvSpec;
use crate::Result;

#[test]
fn validates_supported_packing() -> Result<()> {
    let int4 = AffineGemvSpec::new(2_816, 704, 64, 4)?;
    let int8 = AffineGemvSpec::new(2_816, 4_096, 64, 8)?;
    assert_eq!(int4.layout()?.packed_per_matrix, 704 * 352);
    assert_eq!(int8.layout()?.groups_per_matrix, 4_096 * 44);
    Ok(())
}
