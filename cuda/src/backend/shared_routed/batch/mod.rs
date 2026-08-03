use mircuda::{CaptureMode, Graph, Stream};
use runtime::backend::DecodeSequence;

use super::{CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate};
use crate::{Error, Result};

mod graph;
mod layer;
mod prefill;
mod states;

use graph::DecodeResources;
pub(crate) use prefill::CudaSharedRoutedPrefillBatch;

#[derive(Debug)]
enum DecodeState {
    Direct(DecodeResources),
    Captured {
        graph: Graph<DecodeResources>,
        partitions: usize,
    },
}

#[derive(Debug)]
pub(crate) struct CudaSharedRoutedDecodeBatch {
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
    ) -> Result<()> {
        let state = self
            .state
            .take()
            .ok_or(Error::InvalidDecoderKernel("shared-routed decode state is unavailable"))?;
        let next = match state {
            DecodeState::Direct(mut resources) => {
                if let Err(error) = execute_direct(&mut resources, sessions, sequences) {
                    self.state = Some(DecodeState::Direct(resources));
                    return Err(error);
                }
                let partitions = resources.capture_partitions();
                match capture(&self.stream, resources) {
                    Ok(graph) => {
                        tracing::debug!(
                            rows = sessions.len(),
                            "captured shared-routed CUDA decode graph"
                        );
                        DecodeState::Captured { graph, partitions }
                    },
                    Err((error, resources)) => {
                        tracing::warn!(%error, "CUDA decode graph capture failed; using direct execution");
                        DecodeState::Direct(resources)
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
                    if let Err(error) =
                        graph.with_resources_mut(|resources| resources.commit(sessions))
                    {
                        self.state = Some(DecodeState::Captured { graph, partitions });
                        return Err(error);
                    }
                    DecodeState::Captured { graph, partitions }
                } else {
                    let mut resources = graph.into_resources();
                    if let Err(error) = DecodeResources::execute(&mut resources)
                        .and_then(|()| resources.commit(sessions))
                    {
                        self.state = Some(DecodeState::Direct(resources));
                        return Err(error);
                    }
                    match capture(&self.stream, resources) {
                        Ok(graph) => {
                            tracing::debug!(
                                rows = sessions.len(),
                                partitions = next,
                                "recaptured shared-routed CUDA decode graph"
                            );
                            DecodeState::Captured { graph, partitions: next }
                        },
                        Err((error, resources)) => {
                            tracing::warn!(
                                %error,
                                "CUDA decode graph recapture failed; using direct execution"
                            );
                            DecodeState::Direct(resources)
                        },
                    }
                }
            },
        };
        self.state = Some(next);
        Ok(())
    }
}

fn execute_direct(
    resources: &mut DecodeResources,
    sessions: &mut [&mut CudaSharedRoutedModelSession],
    sequences: &[DecodeSequence],
) -> Result<()> {
    resources.prepare(sessions, sequences)?;
    DecodeResources::execute(resources)?;
    resources.commit(sessions)
}

fn capture(
    stream: &Stream,
    resources: DecodeResources,
) -> std::result::Result<Graph<DecodeResources>, (Error, DecodeResources)> {
    stream.capture_or_recover(CaptureMode::ThreadLocal, resources, DecodeResources::execute)
}
