use foundation::model::BackendTarget;
use models::{
    execution::{ArchitectureRequirements, DecoderExecutionContract, TaskExecutionPlan},
    layout::{ImageProcessorConfig, VisionConfig},
    semantic::SemanticModelSpec,
    weights::TensorReadiness,
};

mod dense_execution;
mod label;
mod registry;
mod types;

#[cfg(test)]
mod tests;

pub use types::{
    AdmissionCheck, AdmissionCheckKind, AdmissionStatus, BackendAdmissionReport,
    CheckpointEncoding, WeightEncoding,
};

/// Schema version of the checked-in physical model-format capability registry.
pub const MODEL_FORMAT_REGISTRY_SCHEMA_VERSION: u32 = 2;

pub(super) fn inspect(
    execution: Option<&DecoderExecutionContract>,
    task: &TaskExecutionPlan,
    vision: Option<&VisionConfig>,
    vision_readiness: Option<&TensorReadiness>,
    image_processor: Option<&ImageProcessorConfig>,
    backend: BackendTarget,
) -> BackendAdmissionReport {
    let mut encoding = execution.map_or_else(
        || match task {
            TaskExecutionPlan::SequenceScoring { bindings, .. } => {
                CheckpointEncoding::from_encoder_bindings(bindings)
            },
            TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::Embedding { .. } => {
                CheckpointEncoding::default()
            },
        },
        |execution| CheckpointEncoding::from_bindings(&execution.bindings),
    );
    if let Some(readiness) = vision_readiness {
        encoding.include_dense_dtypes(&readiness.dtypes);
    }
    let semantic = execution.map(|execution| &execution.semantic);
    let architecture = ArchitectureRequirements::discover(task, semantic);
    let architecture_check = assess_architecture(&backend, task, semantic);
    let mut report = BackendAdmissionReport::build(
        backend,
        encoding,
        Some(architecture),
        Some(architecture_check),
    );
    if let Some(check) = dense_execution::assess(&report.backend, execution, task) {
        resolve_dense_check(&mut report.checks, check);
        report.status = aggregate(&report.checks);
    }
    if let Some(vision) = vision {
        report
            .checks
            .push(vision_check(vision, vision_readiness, image_processor, &report.backend));
        report.status = aggregate(&report.checks);
    }
    report
}

fn resolve_dense_check(checks: &mut Vec<AdmissionCheck>, resolved: AdmissionCheck) {
    checks.retain(|check| {
        check.kind != AdmissionCheckKind::Dense || check.status != AdmissionStatus::Partial
    });
    checks.push(resolved);
}

fn vision_check(
    vision: &VisionConfig,
    readiness: Option<&TensorReadiness>,
    processor: Option<&ImageProcessorConfig>,
    backend: &BackendTarget,
) -> AdmissionCheck {
    let (status, detail) = match readiness {
        None => (AdmissionStatus::Unknown, "vision tensor readiness is unavailable".into()),
        Some(readiness) if !readiness.is_ready() => (
            AdmissionStatus::Unsupported,
            format!("{} required vision tensors are missing", readiness.missing.len()),
        ),
        Some(_) if processor.is_none() => (
            AdmissionStatus::Unsupported,
            "checkpoint does not provide a supported image processor".into(),
        ),
        Some(readiness)
            if !readiness
                .dtypes
                .iter()
                .all(|dtype| matches!(dtype.as_str(), "BF16" | "F16" | "F32")) =>
        {
            (
                AdmissionStatus::Unsupported,
                format!("vision tensors use unsupported dtype(s): {}", readiness.dtypes.join(", ")),
            )
        },
        Some(_) if matches!(backend, BackendTarget::CpuReference) => (
            AdmissionStatus::Unsupported,
            format!(
                "the CPU reference backend does not execute {:?} vision tensors",
                vision.pipeline()
            ),
        ),
        Some(readiness) => (
            AdmissionStatus::Supported,
            format!(
                "{:?} vision tensors ({}) and image processor are ready",
                vision.pipeline(),
                readiness.dtypes.join(", ")
            ),
        ),
    };
    AdmissionCheck {
        kind: AdmissionCheckKind::Vision,
        status,
        detail,
    }
}

impl BackendAdmissionReport {
    #[must_use]
    /// Evaluates physical checkpoint encodings against the current backend
    /// capability registry.
    pub fn for_encoding(backend: BackendTarget, encoding: CheckpointEncoding) -> Self {
        Self::build(backend, encoding, None, None)
    }

    fn build(
        backend: BackendTarget,
        encoding: CheckpointEncoding,
        architecture: Option<ArchitectureRequirements>,
        architecture_check: Option<AdmissionCheck>,
    ) -> Self {
        let mut checks = encoding
            .weights
            .iter()
            .map(|encoding| registry::assess(&backend, encoding))
            .collect::<Vec<_>>();
        if !checks.is_empty()
            && checks.iter().all(|check| check.status != AdmissionStatus::Unsupported)
        {
            checks.push(architecture_check.unwrap_or_else(|| AdmissionCheck {
                kind: AdmissionCheckKind::Architecture,
                status: AdmissionStatus::Partial,
                detail: "backend architecture admission is pending".into(),
            }));
        }
        let status = aggregate(&checks);
        Self {
            backend,
            status,
            encoding,
            architecture,
            checks,
        }
    }
}

fn assess_architecture(
    backend: &BackendTarget,
    task: &TaskExecutionPlan,
    semantic: Option<&SemanticModelSpec>,
) -> AdmissionCheck {
    match backend {
        BackendTarget::Metal => {
            admission_result("Metal", architecture::metal::admit(task, semantic))
        },
        BackendTarget::Cuda => admission_result("CUDA", architecture::cuda::admit(task, semantic)),
        BackendTarget::CpuReference => AdmissionCheck {
            kind: AdmissionCheckKind::Architecture,
            status: AdmissionStatus::Unsupported,
            detail: "the CPU reference backend does not execute product models".into(),
        },
    }
}

fn admission_result<T, E: std::fmt::Display>(
    backend: &str,
    result: Result<T, E>,
) -> AdmissionCheck {
    match result {
        Ok(_) => AdmissionCheck {
            kind: AdmissionCheckKind::Architecture,
            status: AdmissionStatus::Supported,
            detail: format!("{backend} admits the complete task and decoder composition"),
        },
        Err(error) => AdmissionCheck {
            kind: AdmissionCheckKind::Architecture,
            status: AdmissionStatus::Unsupported,
            detail: error.to_string(),
        },
    }
}

fn aggregate(checks: &[AdmissionCheck]) -> AdmissionStatus {
    if checks.is_empty() {
        return AdmissionStatus::Unknown;
    }
    if checks.iter().any(|check| check.status == AdmissionStatus::Unsupported) {
        return AdmissionStatus::Unsupported;
    }
    if checks.iter().any(|check| check.status == AdmissionStatus::Unknown) {
        return AdmissionStatus::Unknown;
    }
    if checks.iter().any(|check| check.status == AdmissionStatus::Partial) {
        return AdmissionStatus::Partial;
    }
    AdmissionStatus::Supported
}
