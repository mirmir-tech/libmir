//! `strftime_now` for chat templates: the Python `strftime` directives that
//! Hugging Face templates use for the current date (Llama 3.x writes
//! `%d %b %Y`), in UTC. Any other directive is an error rather than a guess.

use minijinja::{Error, ErrorKind};
use time::OffsetDateTime;

pub(super) fn strftime_now(format: &str) -> Result<String, Error> {
    strftime(OffsetDateTime::now_utc(), format)
}

fn strftime(now: OffsetDateTime, format: &str) -> Result<String, Error> {
    let month = now.month().to_string();
    let weekday = now.weekday().to_string();
    let mut output = String::with_capacity(format.len() + 8);
    let mut characters = format.chars();
    while let Some(character) = characters.next() {
        if character != '%' {
            output.push(character);
            continue;
        }
        let directive = characters.next().unwrap_or('%');
        let value = match directive {
            'Y' => now.year().to_string(),
            'y' => format!("{:02}", now.year().rem_euclid(100)),
            'm' => format!("{:02}", now.month() as u8),
            'd' => format!("{:02}", now.day()),
            'e' => format!("{:>2}", now.day()),
            'B' => month.clone(),
            'b' => month.chars().take(3).collect(),
            'A' => weekday.clone(),
            'a' => weekday.chars().take(3).collect(),
            'H' => format!("{:02}", now.hour()),
            'M' => format!("{:02}", now.minute()),
            'S' => format!("{:02}", now.second()),
            '%' => "%".into(),
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("unsupported strftime_now format `{format}`"),
                ));
            },
        };
        output.push_str(&value);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use time::{Date, Month, PrimitiveDateTime, Time};

    use super::strftime;

    #[test]
    fn formats_the_directives_templates_use() -> Result<(), Box<dyn std::error::Error>> {
        let now = PrimitiveDateTime::new(
            Date::from_calendar_date(2026, Month::March, 5)?,
            Time::from_hms(7, 8, 9)?,
        )
        .assume_utc();
        assert_eq!(strftime(now, "%Y-%m-%d")?, "2026-03-05");
        assert_eq!(strftime(now, "%d %b %Y")?, "05 Mar 2026");
        assert_eq!(strftime(now, "%A, %B %e %H:%M:%S %%")?, "Thursday, March  5 07:08:09 %");
        assert!(strftime(now, "%j").is_err());
        Ok(())
    }
}
