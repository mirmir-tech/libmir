use super::{
    fixture::{Family, load_family},
    *,
};
mod admission;
mod generation;
mod scalar;

#[test]
fn reserved_decode_tail_preserves_multimodel_prefix_churn() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (mut model, directory) = load_family(family, 10)?;
        model.stream.set_decode_reservation(crate::config::DecodeReservation::Tokens(
            std::num::NonZeroUsize::new(64).ok_or(crate::engine::Error::ShapeOverflow)?,
        ));
        churn::exercise(&mut model)?;
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}
