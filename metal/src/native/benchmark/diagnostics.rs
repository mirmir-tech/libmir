use std::{env, sync::Arc};

use crate::MetalConfig;

pub(super) fn metal_config() -> Arc<MetalConfig> {
    let mut config = MetalConfig::default();
    config.diagnostics.profile_layers = enabled("MIRMIR_METAL_PROFILE_LAYERS");
    config.diagnostics.profile_components = enabled("MIRMIR_METAL_PROFILE_COMPONENTS");
    config.diagnostics.profile_graph_build = enabled("MIRMIR_METAL_PROFILE_GRAPH_BUILD");
    Arc::new(config)
}

fn enabled(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true" | "TRUE" | "yes" | "YES"))
}
