use mircuda::{CaptureMode, Graph, Stream};
use runtime::backend::DecodeSequence;

use super::{CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate};
use crate::{Error, Result};

mod graph;
mod layer;
mod output;
mod prefill;
mod states;

use graph::DecodeResources;
pub use prefill::CudaSharedRoutedPrefillBatch;

#[derive(Debug)]
enum DecodeState {
    Direct(DecodeResources),
    Captured {
        graph: Graph<DecodeResources>,
        partitions: usize,
    },
}

#[derive(Debug)]
pub struct CudaSharedRoutedDecodeBatch {
    state: Option<DecodeState>,
    stream: Stream,
}

impl CudaSharedRoutedDecodeBatch {
    pub(super) fn new(template: &CudaSharedRoutedModelTemplate, rows: usize) -> Result<Self> {
        let resources = DecodeResources::new(template, rows)?;
        Ok(Self {
            state: Some(DecodeState::Direct(resources)),
            stream: template.backend.inner.stream.clone(),
        })
    }

    pub(crate) fn execute(
        &mut self,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
        sequences: &[DecodeSequence],
    ) -> Result<Option<Vec<u32>>> {
        let state = self
            .state
            .take()
            .ok_or(Error::InvalidDecoderKernel("shared-routed decode state is unavailable"))?;
        match state {
            DecodeState::Direct(mut resources) => {
                if let Err(error) = execute_direct(&mut resources, sessions, sequences) {
                    self.state = Some(DecodeState::Direct(resources));
                    return Err(error);
                }
                let sampled = match resources.finish(sessions, sequences) {
                    Ok(sampled) => sampled,
                    Err(error) => {
                        self.state = Some(DecodeState::Direct(resources));
                        return Err(error);
                    },
                };
                let partitions = resources.capture_partitions();
                match capture(&self.stream, resources) {
                    Ok(graph) => {
                        tracing::debug!(
                            rows = sessions.len(),
                            "captured shared-routed CUDA decode graph"
                        );
                        self.state = Some(DecodeState::Captured { graph, partitions });
                        Ok(sampled)
                    },
                    Err((error, resources)) => {
                        tracing::warn!(%error, "CUDA decode graph capture failed; using direct execution");
                        self.state = Some(DecodeState::Direct(resources));
                        Ok(sampled)
                    },
                }
            },
            DecodeState::Captured { mut graph, partitions } => {
                if let Err(error) =
                    graph.with_resources_mut(|resources| resources.prepare(sessions, sequences))
                {
                    self.state = Some(DecodeState::Captured { graph, partitions });
                    return Err(error);
                }
                let next = graph.resources().capture_partitions();
                if next == partitions {
                    if let Err(error) = graph.launch(&self.stream) {
                        self.state = Some(DecodeState::Captured { graph, partitions });
                        return Err(error.into());
                    }
                    let sampled = match graph
                        .with_resources_mut(|resources| resources.finish(sessions, sequences))
                    {
                        Ok(sampled) => sampled,
                        Err(error) => {
                            self.state = Some(DecodeState::Captured { graph, partitions });
                            return Err(error);
                        },
                    };
                    self.state = Some(DecodeState::Captured { graph, partitions });
                    return Ok(sampled);
                }
                let mut resources = graph.into_resources();
                if let Err(error) = DecodeResources::execute(&mut resources) {
                    self.state = Some(DecodeState::Direct(resources));
                    return Err(error);
                }
                let sampled = match resources.finish(sessions, sequences) {
                    Ok(sampled) => sampled,
                    Err(error) => {
                        self.state = Some(DecodeState::Direct(resources));
                        return Err(error);
                    },
                };
                match capture(&self.stream, resources) {
                    Ok(graph) => {
                        tracing::debug!(
                            rows = sessions.len(),
                            partitions = next,
                            "recaptured shared-routed CUDA decode graph"
                        );
                        self.state = Some(DecodeState::Captured { graph, partitions: next });
                        Ok(sampled)
                    },
                    Err((error, resources)) => {
                        tracing::warn!(
                            %error,
                            "CUDA decode graph recapture failed; using direct execution"
                        );
                        self.state = Some(DecodeState::Direct(resources));
                        Ok(sampled)
                    },
                }
            },
        }
    }
}

fn execute_direct(
    resources: &mut DecodeResources,
    sessions: &mut [&mut CudaSharedRoutedModelSession],
    sequences: &[DecodeSequence],
) -> Result<()> {
    resources.prepare(sessions, sequences)?;
    DecodeResources::execute(resources)
}

#[allow(clippy::result_large_err)]
fn capture(
    stream: &Stream,
    resources: DecodeResources,
) -> std::result::Result<Graph<DecodeResources>, (Error, DecodeResources)> {
    stream.capture_or_recover(CaptureMode::ThreadLocal, resources, DecodeResources::execute)
}
