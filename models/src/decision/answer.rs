use super::{AgentConfig, Question, QuestionKind};
use crate::error::{ModelsError, Result};

/// The typed outcome of one question.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Label of the most probable option.
    Choice { label: String },
    /// Most probable level and the probability-weighted mean level.
    Score { level: usize, expected: f64 },
    /// Probability that the statement holds.
    YesNo { probability: f64 },
}

/// A decoded answer with its calibrated option distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    pub verdict: Verdict,
    /// Option probabilities after temperature scaling, in option order.
    pub probabilities: Vec<f64>,
    /// Probability of the reported option, the quantity temperature scaling
    /// calibrates.
    pub confidence: f64,
    /// `1 - H(p) / ln(k)`: how concentrated the distribution is. Not
    /// calibrated and not comparable to `confidence`.
    pub concentration: f64,
}

impl Answer {
    /// Decodes the raw head logits of `question`, one per option.
    pub fn decode(question: &Question, logits: &[f32], config: &AgentConfig) -> Result<Self> {
        let options = question.rendered_options().len();
        if logits.len() != options {
            return Err(ModelsError::InvalidConfig(format!(
                "decision head returned {} logits for {options} options",
                logits.len()
            )));
        }
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(ModelsError::InvalidConfig(
                "decision head returned a non-finite logit".into(),
            ));
        }
        let temperature = config.temperature(question.kind(), options);
        let probabilities = softmax(logits, temperature);
        let (best, confidence) = probabilities.iter().copied().enumerate().fold(
            (0, f64::NEG_INFINITY),
            |best, candidate| {
                if candidate.1 > best.1 {
                    candidate
                } else {
                    best
                }
            },
        );
        let verdict = match question.kind() {
            QuestionKind::Choice => Verdict::Choice {
                label: question.choice_labels()[best].to_owned(),
            },
            QuestionKind::Score => Verdict::Score {
                level: best,
                expected: probabilities
                    .iter()
                    .enumerate()
                    .map(|(level, p)| level_value(level) * p)
                    .sum(),
            },
            QuestionKind::YesNo => Verdict::YesNo { probability: probabilities[1] },
        };
        Ok(Self {
            verdict,
            concentration: concentration(&probabilities),
            probabilities,
            confidence,
        })
    }
}

fn softmax(logits: &[f32], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logits.iter().map(|&logit| f64::from(logit) / temperature).collect();
    let maximum = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = scaled.iter().map(|value| (value - maximum).exp()).collect();
    let total: f64 = exponentials.iter().sum();
    exponentials.into_iter().map(|value| value / total).collect()
}

fn concentration(probabilities: &[f64]) -> f64 {
    if probabilities.len() < 2 {
        return 1.0;
    }
    let entropy: f64 = probabilities.iter().map(|&p| -p * p.clamp(1e-12, 1.0).ln()).sum();
    (1.0 - entropy / level_value(probabilities.len()).ln()).clamp(0.0, 1.0)
}

#[expect(clippy::cast_precision_loss, reason = "option counts stay below 2^52")]
const fn level_value(level: usize) -> f64 {
    level as f64
}

#[cfg(test)]
mod tests {
    use super::{Answer, Verdict};
    use crate::{
        Result,
        decision::{AgentConfig, ChoiceOption, Question, Verdicts},
    };

    fn config() -> Result<AgentConfig> {
        AgentConfig::from_json(
            r#"{"max_len": 64, "head_max_len": 32, "head_layers": 2,
                "temperature": [1, 2, 1]}"#,
        )
    }

    #[test]
    fn decodes_each_kind() -> Result<()> {
        let choice = Question::choice(
            "?",
            vec![ChoiceOption::new("a", None), ChoiceOption::new("b", None)],
        )?;
        let answer = Answer::decode(&choice, &[0.0, 2.0_f32.ln()], &config()?)?;
        assert_eq!(answer.verdict, Verdict::Choice { label: "b".into() });
        assert!((answer.confidence - 2.0 / 3.0).abs() < 1e-6);

        let score = Question::score("?", vec!["low".into(), "high".into()])?;
        let answer = Answer::decode(&score, &[0.0, 0.0], &config()?)?;
        assert_eq!(answer.verdict, Verdict::Score { level: 0, expected: 0.5 });
        assert!(answer.concentration.abs() < 1e-9);

        let yes_no = Question::yes_no("?", Verdicts::default())?;
        let answer = Answer::decode(&yes_no, &[0.0, 3.0_f32.ln()], &config()?)?;
        assert!(
            matches!(answer.verdict, Verdict::YesNo { probability } if (probability - 0.75).abs() < 1e-6)
        );
        Ok(())
    }

    #[test]
    fn applies_the_kind_temperature() -> Result<()> {
        let score = Question::score("?", vec!["low".into(), "high".into()])?;
        let answer = Answer::decode(&score, &[0.0, 2.0 * 3.0_f32.ln()], &config()?)?;
        assert!((answer.probabilities[1] - 0.75).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn rejects_a_logit_count_that_does_not_match_the_options() -> Result<()> {
        let yes_no = Question::yes_no("?", Verdicts::default())?;
        assert!(Answer::decode(&yes_no, &[0.0], &config()?).is_err());
        assert!(Answer::decode(&yes_no, &[0.0, f32::NAN], &config()?).is_err());
        Ok(())
    }
}
