mod options;

pub use options::{ChoiceOption, Verdicts};

use crate::error::{ModelsError, Result};

/// The three decision heads of a Laya checkpoint, in type-embedding order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QuestionKind {
    /// Pick one labelled option.
    Choice,
    /// Place the state on an ordered scale.
    Score,
    /// Decide whether a statement holds.
    YesNo,
}

impl QuestionKind {
    /// Row of the type embedding the head adds for this kind.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::YesNo => 2,
        }
    }

    /// Name the checkpoint was trained to read in the instruction prefix.
    #[must_use]
    pub const fn prompt_name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::YesNo => "noul",
        }
    }

    pub(crate) fn parse(name: &str) -> Option<Self> {
        [Self::Choice, Self::Score, Self::YesNo]
            .into_iter()
            .find(|kind| kind.prompt_name() == name)
    }
}

/// One typed question with a fixed option set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Question {
    instruction: String,
    options: QuestionOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QuestionOptions {
    Choice(Vec<ChoiceOption>),
    Score(Vec<String>),
    YesNo(Verdicts),
}

impl Question {
    /// A choice among options with distinct labels.
    pub fn choice(instruction: impl Into<String>, options: Vec<ChoiceOption>) -> Result<Self> {
        if options.is_empty() {
            return Err(invalid("a choice question needs at least one option"));
        }
        for (index, option) in options.iter().enumerate() {
            if options[..index].iter().any(|earlier| earlier.label == option.label) {
                return Err(invalid(format!("choice label `{}` repeats", option.label)));
            }
        }
        Self::new(instruction.into(), QuestionOptions::Choice(options))
    }

    /// An ordered scale; `levels[0]` describes the lowest level.
    pub fn score(instruction: impl Into<String>, levels: Vec<String>) -> Result<Self> {
        if levels.is_empty() {
            return Err(invalid("a score question needs at least one level"));
        }
        Self::new(instruction.into(), QuestionOptions::Score(levels))
    }

    /// Whether a statement holds.
    pub fn yes_no(instruction: impl Into<String>, verdicts: Verdicts) -> Result<Self> {
        Self::new(instruction.into(), QuestionOptions::YesNo(verdicts))
    }

    fn new(instruction: String, options: QuestionOptions) -> Result<Self> {
        if instruction.trim().is_empty() {
            return Err(invalid("a question needs a non-empty instruction"));
        }
        Ok(Self { instruction, options })
    }

    #[must_use]
    pub const fn kind(&self) -> QuestionKind {
        match self.options {
            QuestionOptions::Choice(_) => QuestionKind::Choice,
            QuestionOptions::Score(_) => QuestionKind::Score,
            QuestionOptions::YesNo(_) => QuestionKind::YesNo,
        }
    }

    #[must_use]
    pub fn instruction(&self) -> &str {
        &self.instruction
    }

    /// Choice labels in option order; empty for other kinds.
    #[must_use]
    pub fn choice_labels(&self) -> Vec<&str> {
        match &self.options {
            QuestionOptions::Choice(options) => {
                options.iter().map(|option| option.label.as_str()).collect()
            },
            QuestionOptions::Score(_) | QuestionOptions::YesNo(_) => Vec::new(),
        }
    }

    /// Option texts in marker order, exactly as the checkpoint reads them.
    #[must_use]
    pub fn rendered_options(&self) -> Vec<String> {
        match &self.options {
            QuestionOptions::Choice(options) => options.iter().map(ChoiceOption::render).collect(),
            QuestionOptions::Score(levels) => levels
                .iter()
                .enumerate()
                .map(|(index, level)| format!("level {index}: {level}"))
                .collect(),
            QuestionOptions::YesNo(verdicts) => verdicts.render().into(),
        }
    }
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

#[cfg(test)]
mod tests;
