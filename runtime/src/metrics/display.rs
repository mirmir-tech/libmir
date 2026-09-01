use std::fmt;

use super::{ThroughputRate, ThroughputUnit};

impl fmt::Display for ThroughputRate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.per_second {
            Some(rate) => write!(formatter, "{rate:.3} {}", self.unit),
            None => formatter.write_str("n/a"),
        }
    }
}

impl fmt::Display for ThroughputUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Token => "tok/s",
        })
    }
}
