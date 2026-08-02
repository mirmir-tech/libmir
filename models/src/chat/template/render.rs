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
    environment.add_function("raise_exception", raise_exception);
    environment.add_template("chat", body)?;
    let template = environment.get_template("chat")?;
    let disabled = request.tool_choice.as_ref().and_then(serde_json::Value::as_str) == Some("none");
    let tools = (!disabled && !request.tools.is_empty()).then_some(&request.tools);
    Ok(template.render(context! {
        messages => &request.messages,
        tools => tools,
        bos_token => tokens.bos(),
        eos_token => tokens.eos(),
        add_generation_prompt => true,
        enable_thinking => true,
    })?)
}

fn raise_exception(message: &str) -> std::result::Result<String, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message.to_owned()))
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
                tools: Vec::new(),
                tool_choice: None,
                stream: false,
                max_tokens: None,
                min_tokens: None,
                ignore_eos: None,
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

    #[test]
    fn renders_mistral_v3_inst_template_constructs() -> Result<()> {
        let body = r#"
{%- set loop_messages = messages %}
{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}
{%- set ns = namespace() %}
{%- set ns.index = 0 %}
{{- bos_token }}
{%- for message in loop_messages %}
    {%- if message.role == "user" %}
        {{- "[INST] " + message.content + "[/INST]" }}
        {%- set ns.index = ns.index + 1 %}
    {%- endif %}
{%- endfor %}
"#;

        let rendered = render_model_template(
            body,
            &request("Explain Rust ownership."),
            &TemplateTokens::new("<s>", "</s>"),
        )?;

        assert_eq!(rendered, "<s>[INST] Explain Rust ownership.[/INST]");
        Ok(())
    }

    #[test]
    fn exposes_tools_only_when_present() -> Result<()> {
        let body = r"{%- if tools is not none -%}[AVAILABLE_TOOLS]{{ tools|tojson }}[/AVAILABLE_TOOLS]{%- endif -%}";
        let mut request = request("Weather?");
        assert_eq!(render_model_template(body, &request, &TemplateTokens::default())?, "");
        request.tools.push(foundation::protocol::ChatTool {
            kind: "function".into(),
            function: foundation::protocol::ChatFunctionDefinition {
                name: "weather".into(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
            },
        });
        let rendered = render_model_template(body, &request, &TemplateTokens::default())?;
        assert!(rendered.starts_with("[AVAILABLE_TOOLS]["));
        assert!(rendered.contains(r#""name":"weather""#));
        assert!(rendered.ends_with("][/AVAILABLE_TOOLS]"));
        request.tool_choice = Some(serde_json::json!("none"));
        assert_eq!(render_model_template(body, &request, &TemplateTokens::default())?, "");
        Ok(())
    }

    fn request(content: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test".into(),
            messages: vec![foundation::protocol::ChatMessage {
                role: "user".into(),
                content: content.into(),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Vec::new(),
            tool_choice: None,
            stream: false,
            max_tokens: None,
            min_tokens: None,
            ignore_eos: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            seed: None,
        }
    }
}
