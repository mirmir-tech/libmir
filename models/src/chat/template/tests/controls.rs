use super::{request, *};

/// The Nemotron template's switches, which default themselves in place:
/// undefined keeps the template default.
const SWITCHES: &str = "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True -%}\
{%- set medium_effort = medium_effort if medium_effort is defined else False -%}\
{%- set truncate_history_thinking = truncate_history_thinking if truncate_history_thinking is defined else True -%}\
thinking={{ enable_thinking }} effort={{ medium_effort }} truncate={{ truncate_history_thinking }}";

fn jinja(body: &str) -> ChatTemplate {
    ChatTemplate {
        kind: TemplateKind::ModelJinja,
        source: TemplateSource::ChatTemplateFile,
        tokens: TemplateTokens::default(),
        template: Some(body.into()),
    }
}

fn render(
    template: &ChatTemplate,
    effort: ReasoningEffort,
    history: ReasoningHistory,
) -> Result<String> {
    let controls = TemplateControls {
        effort,
        history,
        ..TemplateControls::default()
    };
    Ok(template.render_with_controls(&request("Hi"), controls)?.text)
}

#[test]
fn declared_switches_follow_the_request_and_default_to_the_template() -> Result<()> {
    use ReasoningEffort::{Efficient, ModelDefault as DefaultEffort};
    use ReasoningHistory::{Keep, ModelDefault as DefaultHistory, Truncate};
    let template = jinja(SWITCHES);
    assert_eq!(
        render(&template, DefaultEffort, DefaultHistory)?,
        "thinking=true effort=false truncate=true"
    );
    assert_eq!(
        render(&template, Efficient, DefaultHistory)?,
        "thinking=true effort=true truncate=true"
    );
    assert_eq!(
        render(&template, DefaultEffort, Keep)?,
        "thinking=true effort=false truncate=false"
    );
    assert_eq!(
        render(&template, DefaultEffort, Truncate)?,
        "thinking=true effort=false truncate=true"
    );
    let disabled = TemplateControls {
        reasoning: ReasoningMode::Disabled,
        ..TemplateControls::default()
    };
    assert_eq!(
        template.render_with_controls(&request("Hi"), disabled)?.text,
        "thinking=false effort=false truncate=true"
    );
    Ok(())
}

#[test]
fn undeclared_switches_are_rejected() -> Result<()> {
    let template = jinja("{{ messages[0].content }}");
    assert_eq!(
        render(&template, ReasoningEffort::ModelDefault, ReasoningHistory::ModelDefault)?,
        "Hi"
    );
    assert!(matches!(
        render(&template, ReasoningEffort::Efficient, ReasoningHistory::ModelDefault),
        Err(crate::ModelsError::UnsupportedReasoningControl)
    ));
    assert!(matches!(
        render(&template, ReasoningEffort::ModelDefault, ReasoningHistory::Keep),
        Err(crate::ModelsError::UnsupportedReasoningControl)
    ));
    Ok(())
}
