//! One admission decision and what it inspected.

use architecture::ArchitectureError;

use super::{super::EncodingKind, AdmissionStatus};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Stable category of a backend admission check.
pub enum AdmissionCheckKind {
    /// One family of weight encodings.
    Encoding(EncodingKind),
    /// Semantic model composition and backend lowering.
    Architecture,
    /// Vision tensors and image preprocessing contract.
    Vision,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// One independently explainable backend admission decision.
pub struct AdmissionCheck {
    /// What was inspected and what was decided.
    pub subject: CheckSubject,
    /// User-facing explanation or missing requirement.
    pub detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// The capability a check inspected, with its result: only an architecture
/// check can carry a typed refusal.
pub enum CheckSubject {
    /// One family of weight encodings.
    Encoding {
        /// The family inspected.
        kind: EncodingKind,
        /// Its result.
        status: AdmissionStatus,
    },
    /// Semantic model composition and backend lowering.
    Architecture(ArchitectureOutcome),
    /// Vision tensors and the image preprocessing contract.
    Vision(AdmissionStatus),
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// The backend's decision on a model's architecture.
pub enum ArchitectureOutcome {
    /// A runtime executes the complete task and decoder.
    Admitted,
    /// Architecture admission has not run.
    Pending,
    /// No runtime executes the model: what is unlowerable or what each
    /// runtime lacks.
    Refused(ArchitectureError),
    /// The backend executes no product model at all.
    NotExecuted,
}

impl AdmissionCheck {
    #[must_use]
    /// The capability family this check inspected.
    pub const fn kind(&self) -> AdmissionCheckKind {
        match self.subject {
            CheckSubject::Encoding { kind, .. } => AdmissionCheckKind::Encoding(kind),
            CheckSubject::Architecture(_) => AdmissionCheckKind::Architecture,
            CheckSubject::Vision(_) => AdmissionCheckKind::Vision,
        }
    }

    #[must_use]
    /// The result of this check.
    pub const fn status(&self) -> AdmissionStatus {
        match &self.subject {
            CheckSubject::Encoding { status, .. } | CheckSubject::Vision(status) => *status,
            CheckSubject::Architecture(ArchitectureOutcome::Admitted) => AdmissionStatus::Supported,
            CheckSubject::Architecture(ArchitectureOutcome::Pending) => AdmissionStatus::Partial,
            CheckSubject::Architecture(
                ArchitectureOutcome::Refused(_) | ArchitectureOutcome::NotExecuted,
            ) => AdmissionStatus::Unsupported,
        }
    }

    #[must_use]
    /// Why the backend refused the architecture, typed.
    pub const fn refusal(&self) -> Option<&ArchitectureError> {
        match &self.subject {
            CheckSubject::Architecture(ArchitectureOutcome::Refused(error)) => Some(error),
            _ => None,
        }
    }
}
