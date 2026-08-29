use foundation::conversation::{Conversation, ToolChoice};
use minijinja::{Environment, Error, ErrorKind, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use time::OffsetDateTime;

use super::config::TemplateTokens;
use crate::error::Result;

pub(super) fn render_model_template(
    body: &str,
    conversation: &Conversation,
    tokens: &TemplateTokens,
) -> Result<String> {
    let mut environment = Environment::new();
    environment.set_unknown_method_callback(unknown_method_callback);
    environment.add_function("strftime_now", strftime_now);
    environment.add_function("raise_exception", raise_exception);
    environment.add_template("chat", body)?;
    let template = environment.get_template("chat")?;
    let enabled = !matches!(conversation.tool_choice, ToolChoice::None);
    let tools = (enabled && !conversation.tools.is_empty()).then_some(&conversation.tools);
    Ok(template.render(context! {
        messages => &conversation.messages,
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
            &Conversation {
                messages: Vec::new(),
                tools: Vec::new(),
                tool_choice: ToolChoice::default(),
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
        request.tools.push(foundation::conversation::Tool {
            kind: "function".into(),
            function: foundation::conversation::FunctionDefinition {
                name: "weather".into(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
            },
        });
        let rendered = render_model_template(body, &request, &TemplateTokens::default())?;
        assert!(rendered.starts_with("[AVAILABLE_TOOLS]["));
        assert!(rendered.contains(r#""name":"weather""#));
        assert!(rendered.ends_with("][/AVAILABLE_TOOLS]"));
        request.tool_choice = ToolChoice::None;
        assert_eq!(render_model_template(body, &request, &TemplateTokens::default())?, "");
        Ok(())
    }

    fn request(content: &str) -> Conversation {
        Conversation {
            messages: vec![foundation::conversation::Message {
                role: "user".into(),
                content: content.into(),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Vec::new(),
            tool_choice: ToolChoice::default(),
        }
    }
}
