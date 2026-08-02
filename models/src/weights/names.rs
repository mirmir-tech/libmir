#[must_use]
pub fn alternate_model_tensor_name(name: &str) -> String {
    name.strip_prefix("model.")
        .map_or_else(|| format!("model.{name}"), str::to_owned)
}

#[must_use]
pub fn model_tensor_aliases(name: impl Into<String>) -> Vec<String> {
    let name = name.into();
    let alternate = alternate_model_tensor_name(&name);
    vec![name, alternate]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggles_only_the_leading_model_scope() {
        assert_eq!(alternate_model_tensor_name("model.vision.weight"), "vision.weight");
        assert_eq!(alternate_model_tensor_name("vision.weight"), "model.vision.weight");
    }
}
