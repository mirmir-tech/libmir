use foundation::protocol::ChatCompletionRequest;
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;

use super::config::TemplateTokens;
use crate::error::Result;

pub(super) fn render_model_template(
    body: &str,
    request: &ChatCompletionRequest,
    tokens: &TemplateTokens,
) -> Result<String> {
    let mut environment = Environment::new();
    environment.set_unknown_method_callback(unknown_method_callback);
    environment.add_template("chat", body)?;
    let template = environment.get_template("chat")?;
    Ok(template.render(context! {
        messages => &request.messages,
        bos_token => tokens.bos(),
        eos_token => tokens.eos(),
        add_generation_prompt => true,
        enable_thinking => true,
    })?)
}
