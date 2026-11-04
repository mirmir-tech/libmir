use super::{ChoiceOption, Question, QuestionKind, Verdicts};
use crate::Result;

#[test]
fn renders_options_like_the_reference_runtime() -> Result<()> {
    let choice = Question::choice(
        "Who is addressed?",
        vec![ChoiceOption::new("anna", Some("the human".into())), ChoiceOption::new("agent", None)],
    )?;
    assert_eq!(choice.rendered_options(), ["anna: the human", "agent"]);
    assert_eq!(choice.kind(), QuestionKind::Choice);

    let score = Question::score("How urgent?", vec!["low".into(), "high".into()])?;
    assert_eq!(score.rendered_options(), ["level 0: low", "level 1: high"]);

    let yes_no = Question::yes_no("Is it a question?", Verdicts::default())?;
    assert_eq!(
        yes_no.rendered_options(),
        ["false: no, the statement does not hold", "true: yes, the statement holds"]
    );
    let custom = Verdicts::default()
        .with_labels("nie", "tak")?
        .with_descriptions(None, Some("pytanie".into()));
    assert_eq!(
        Question::yes_no("Pytanie?", custom)?.rendered_options(),
        ["nie: no, the statement does not hold", "tak: pytanie"]
    );
    Ok(())
}

#[test]
fn rejects_questions_the_head_cannot_answer() {
    assert!(Question::choice("?", Vec::new()).is_err());
    assert!(Question::score("?", Vec::new()).is_err());
    assert!(Question::yes_no("  ", Verdicts::default()).is_err());
    let repeated = vec![ChoiceOption::new("a", None), ChoiceOption::new("a", None)];
    assert!(Question::choice("?", repeated).is_err());
    assert!(Verdicts::default().with_labels("same", "same").is_err());
}

#[test]
fn parses_prompt_names() {
    assert_eq!(QuestionKind::parse("noul"), Some(QuestionKind::YesNo));
    assert_eq!(QuestionKind::parse("yes"), None);
}
