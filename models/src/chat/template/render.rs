use foundation::protocol::ChatCompletionRequest;
use minijinja::{Environment, Error, ErrorKind, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use time::OffsetDateTime;

use super::config::TemplateTokens;
use crate::error::Result;

pub(super) fn render_model_template(
    body: &str,
    request: &ChatCompletionRequest,
    tokens: &TemplateTokens,
) -> Result<String> {
    let mut environment = Environment::new();
    environment.set_unknown_method_callback(unknown_method_callback);
    environment.add_function("strftime_now", strftime_now);
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

fn strftime_now(format: &str) -> std::result::Result<String, Error> {
    if format != "%Y-%m-%d" {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("unsupported strftime_now format `{format}`"),
        ));
    }
    let now = OffsetDateTime::now_utc();
    Ok(format!("{:04}-{:02}-{:02}", now.year(), now.month() as u8, now.day()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_hugging_face_current_date_function() -> Result<()> {
        let rendered = render_model_template(
            "{{ strftime_now(\"%Y-%m-%d\") }}",
            &ChatCompletionRequest {
                model: "test".into(),
                messages: Vec::new(),
                stream: false,
                max_tokens: None,
                temperature: None,
                top_p: None,
                top_k: None,
                repetition_penalty: None,
                seed: None,
            },
            &TemplateTokens::default(),
        )?;

        assert_eq!(rendered.len(), 10);
        assert_eq!(&rendered[4..5], "-");
        assert_eq!(&rendered[7..8], "-");
        Ok(())
    }
}
