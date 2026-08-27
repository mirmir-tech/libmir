use crate::{
    CudaBackend, Error, Result,
    kernels::{GatedDeltaChunked, GatedDeltaChunkedScratch, GatedDeltaLaunch, GatedDeltaSpec},
};

#[derive(Debug)]
pub(in crate::backend) struct CudaGatedDeltaWorkspace {
    spec: GatedDeltaSpec,
    operation: GatedDeltaChunked,
    scratch: GatedDeltaChunkedScratch,
}

impl CudaGatedDeltaWorkspace {
    fn new(backend: &CudaBackend, spec: GatedDeltaSpec) -> Result<Self> {
        Ok(Self {
            spec,
            operation: GatedDeltaChunked::compile(&backend.inner.compiler, spec)?,
            scratch: GatedDeltaChunkedScratch::new(
                &backend.inner.context,
                &backend.inner.pool,
                &backend.inner.stream,
                spec,
            )?,
        })
    }

    fn execute(&mut self, backend: &CudaBackend, launch: &mut GatedDeltaLaunch<'_>) -> Result<()> {
        self.operation.execute(&backend.inner.stream, launch, &mut self.scratch)
    }
}

impl CudaBackend {
    pub(super) fn execute_gated_delta_chunked(
        &self,
        spec: GatedDeltaSpec,
        launch: &mut GatedDeltaLaunch<'_>,
    ) -> Result<()> {
        let mut workspace =
            self.inner.gated_delta_workspace.lock().map_err(|_| {
                Error::InvalidExecutionPlan("Gated Delta workspace lock is poisoned")
            })?;
        if workspace.as_ref().is_none_or(|current| current.spec != spec) {
            *workspace = Some(CudaGatedDeltaWorkspace::new(self, spec)?);
        }
        workspace
            .as_mut()
            .ok_or(Error::InvalidExecutionPlan("Gated Delta workspace is missing"))?
            .execute(self, launch)
    }
}
