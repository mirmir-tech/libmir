use super::{
    fixture::{Family, load_family},
    *,
};

#[test]
fn router_precision_preserves_clamped_and_hybrid_session_lifecycle() -> Result<()> {
    for family in [Family::Clamped, Family::Hybrid] {
        let (model, directory) = load_family(family, 10)?;
        let manifest = model.info.manifest.clone();
        drop(model);
        let mut config = crate::MetalConfig::default();
        config.diagnostics.router_precision = crate::config::RouterPrecision::Float32;
        config.tuning.mode = runtime::tuning::TuningMode::Disabled;
        config.cache.prefix_cache_entries = 10;
        config.cache.paged_attention_min_context = 1;
        let mut model =
            LoadedModel::load_with_config(&manifest, std::sync::Arc::new(config), &mut |_| {})?;
        churn::exercise(&mut model)?;
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}
