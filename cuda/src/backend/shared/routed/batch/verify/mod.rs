//! The decoder pass of a lone speculative step as one CUDA graph: the fed
//! token and its drafts are rows of one session at consecutive positions,
//! attention takes each row as a decode row over the session's pages, and
//! every Mamba-2 layer keeps the session's state resident with snapshots
//! after each draft. The graph ends at the final norm of every row.

use mircuda::{CaptureMode, DeviceBuffer, Graph, Stream, bf16};
use runtime::kv::BlockTable;

use super::super::{CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate};
use crate::{Error, Result};

mod resources;

use resources::VerifyResources;

#[derive(Debug)]
enum VerifyState {
    Direct(VerifyResources),
    Captured {
        graph: Graph<VerifyResources>,
        key: (u32, usize),
    },
}

/// Verification passes of one row count, replayed from a graph while the
/// capture key holds.
#[derive(Debug)]
pub(in crate::backend::shared::routed) struct CudaSharedRoutedVerifyBatch {
    state: Option<VerifyState>,
    stream: Stream,
}

impl CudaSharedRoutedVerifyBatch {
    pub(in crate::backend::shared::routed) fn new(
        template: &CudaSharedRoutedModelTemplate,
        rows: usize,
    ) -> Result<Self> {
        Ok(Self {
            state: Some(VerifyState::Direct(VerifyResources::new(template, rows)?)),
            stream: template.backend.inner.stream.clone(),
        })
    }

    /// Runs the pass over `token` and the session's drafts after it, over
    /// pages of `block_count` blocks; the normalized rows are then in
    /// [`Self::normalized`].
    pub(in crate::backend::shared::routed) fn execute(
        &mut self,
        session: &CudaSharedRoutedModelSession,
        token: u32,
        table: &BlockTable,
        block_count: u32,
    ) -> Result<()> {
        let mut resources = match self.state.take() {
            Some(VerifyState::Captured { mut graph, key }) => {
                let prepared = graph.with_resources_mut(|resources| {
                    resources.prepare(session, token, table)?;
                    Ok::<_, Error>(resources.capture_key(block_count))
                });
                match prepared {
                    Ok(next) if next == key => {
                        let launched = graph.launch(&self.stream);
                        self.state = Some(VerifyState::Captured { graph, key });
                        return launched.map_err(Error::from);
                    },
                    Ok(_) => graph.into_resources(),
                    Err(error) => {
                        self.state = Some(VerifyState::Direct(graph.into_resources()));
                        return Err(error);
                    },
                }
            },
            Some(VerifyState::Direct(mut resources)) => {
                if let Err(error) = resources.prepare(session, token, table) {
                    self.state = Some(VerifyState::Direct(resources));
                    return Err(error);
                }
                resources
            },
            None => return Err(Error::InvalidDecoderKernel("verification state is unavailable")),
        };
        if let Err(error) = resources.execute() {
            self.state = Some(VerifyState::Direct(resources));
            return Err(error);
        }
        let key = resources.capture_key(block_count);
        self.state = Some(
            match self.stream.capture_or_recover(
                CaptureMode::ThreadLocal,
                resources,
                VerifyResources::execute,
            ) {
                Ok(graph) => {
                    tracing::debug!(
                        rows = graph.resources().rows,
                        "captured CUDA verification graph"
                    );
                    VerifyState::Captured { graph, key }
                },
                Err((error, resources)) => {
                    tracing::warn!(%error, "CUDA verification graph capture failed; running directly");
                    VerifyState::Direct(resources)
                },
            },
        );
        Ok(())
    }

    /// Keeps the session's recurrent state after `kept` tokens of the pass.
    pub(in crate::backend::shared::routed) fn commit(
        &mut self,
        session: &mut CudaSharedRoutedModelSession,
        kept: usize,
    ) -> Result<()> {
        // Commits copy snapshots after the replay, in its stream order.
        match &mut self.state {
            Some(VerifyState::Direct(resources)) => resources.commit(session, kept),
            Some(VerifyState::Captured { graph, .. }) => {
                graph.with_resources_mut(|resources| resources.commit(session, kept))
            },
            None => Err(Error::InvalidDecoderKernel("verification state is unavailable")),
        }
    }

    pub(in crate::backend::shared::routed) fn normalized(&self) -> Result<&DeviceBuffer<bf16>> {
        match &self.state {
            Some(VerifyState::Direct(resources)) => Ok(&resources.normalized),
            Some(VerifyState::Captured { graph, .. }) => Ok(&graph.resources().normalized),
            None => Err(Error::InvalidDecoderKernel("verification state is unavailable")),
        }
    }
}
