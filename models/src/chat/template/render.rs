use std::collections::BTreeMap;

use foundation::conversation::{Conversation, ToolChoice};
use minijinja::{Environment, Error, ErrorKind, Value};
use minijinja_contrib::pycompat::unknown_method_callback;
use time::OffsetDateTime;

use super::config::TemplateTokens;
use crate::error::Result;

pub(super) fn render_model_template(
    body: &str,
    conversation: &Conversation,
    tokens: &TemplateTokens,
    controls: super::TemplateControls,
) -> Result<String> {
    let mut environment = Environment::new();
    environment.set_unknown_method_callback(unknown_method_callback);
    environment.add_function("strftime_now", strftime_now);
    environment.add_function("raise_exception", raise_exception);
    environment.add_test("iterable", is_iterable);
    environment.add_template("chat", body)?;
    let template = environment.get_template("chat")?;
    let undeclared = template.undeclared_variables(false);
    // A switch is a template input when the template reads it without
    // assigning it, or when it defaults it with the Hugging Face pattern
    // `{% set name = name if name is defined else default %}`, which the
    // analysis counts as an assignment.
    let declares =
        |name: &str| undeclared.contains(name) || body.contains(&format!("{name} is defined"));
    let switch = |name: &'static str, requested: bool| {
        if requested && !declares(name) {
            Err(crate::ModelsError::UnsupportedReasoningControl)
        } else {
            Ok(())
        }
    };
    switch("enable_thinking", controls.reasoning != super::ReasoningMode::ModelDefault)?;
    switch("medium_effort", controls.effort != super::ReasoningEffort::ModelDefault)?;
    switch(
        "truncate_history_thinking",
        controls.history != super::ReasoningHistory::ModelDefault,
    )?;
    let enabled = !matches!(conversation.tool_choice, ToolChoice::None);
    let tools = (enabled && !conversation.tools.is_empty()).then_some(&conversation.tools);
    let mut variables = BTreeMap::from([
        ("messages", Value::from_serialize(&conversation.messages)),
        ("tools", Value::from_serialize(tools)),
        ("bos_token", Value::from_serialize(tokens.bos())),
        ("eos_token", Value::from_serialize(tokens.eos())),
        ("add_generation_prompt", Value::from(true)),
        ("enable_thinking", Value::from(controls.reasoning.thinking_enabled())),
    ]);
    // Undefined values keep the template's own defaults.
    if controls.effort == super::ReasoningEffort::Efficient {
        variables.insert("medium_effort", Value::from(true));
    }
    match controls.history {
        super::ReasoningHistory::ModelDefault => {},
        super::ReasoningHistory::Keep => {
            variables.insert("truncate_history_thinking", Value::from(false));
        },
        super::ReasoningHistory::Truncate => {
            variables.insert("truncate_history_thinking", Value::from(true));
        },
    }
    Ok(template.render(variables)?)
}

/// Jinja2's `iterable` test: `none` and undefined values are not iterable.
/// Templates guard `tools | length` with it when no tools are passed.
fn is_iterable(value: &Value) -> bool {
    !value.is_none() && !value.is_undefined() && value.try_iter().is_ok()
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
            super::super::ReasoningMode::ModelDefault.into(),
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
            super::super::ReasoningMode::ModelDefault.into(),
        )?;

        assert_eq!(rendered, "<s>[INST] Explain Rust ownership.[/INST]");
        Ok(())
    }

    #[test]
    fn exposes_tools_only_when_present() -> Result<()> {
        let body = r"{%- if tools is not none -%}[AVAILABLE_TOOLS]{{ tools|tojson }}[/AVAILABLE_TOOLS]{%- endif -%}";
        let mut request = request("Weather?");
        assert_eq!(
            render_model_template(
                body,
                &request,
                &TemplateTokens::default(),
                super::super::ReasoningMode::ModelDefault.into()
            )?,
            ""
        );
        request.tools.push(foundation::conversation::Tool {
            kind: "function".into(),
            function: foundation::conversation::FunctionDefinition {
                name: "weather".into(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
            },
        });
        let rendered = render_model_template(
            body,
            &request,
            &TemplateTokens::default(),
            super::super::ReasoningMode::ModelDefault.into(),
        )?;
        assert!(rendered.starts_with("[AVAILABLE_TOOLS]["));
        assert!(rendered.contains(r#""name":"weather""#));
        assert!(rendered.ends_with("][/AVAILABLE_TOOLS]"));
        request.tool_choice = ToolChoice::None;
        assert_eq!(
            render_model_template(
                body,
                &request,
                &TemplateTokens::default(),
                super::super::ReasoningMode::ModelDefault.into()
            )?,
            ""
        );
        Ok(())
    }

    #[test]
    fn none_is_not_iterable() -> Result<()> {
        let body = "{%- if tools is iterable and tools | length > 0 -%}tools{%- endif -%}\
                    {%- if messages is iterable -%}messages{%- endif -%}";
        let rendered = render_model_template(
            body,
            &request("Hello"),
            &TemplateTokens::default(),
            super::super::ReasoningMode::ModelDefault.into(),
        )?;
        assert_eq!(rendered, "messages");
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
