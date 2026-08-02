use std::sync::{Mutex, MutexGuard};

use super::super::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchSpec {
    pub(crate) query_heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) page_capacity: usize,
    pub(crate) blocks: usize,
    pub(crate) reduction_groups: usize,
    pub(crate) head_dim: usize,
    pub(crate) page_size: usize,
    pub(crate) scale_bits: u32,
    pub(crate) dtype: mirtal::DType,
}

#[derive(Debug, Default)]
pub struct PagedAttentionScratch {
    state: Mutex<ScratchState>,
}

#[derive(Debug, Default)]
pub struct ScratchState {
    spec: Option<ScratchSpec>,
    partials: Option<mirtal::Array>,
    sums: Option<mirtal::Array>,
    maximums: Option<mirtal::Array>,
    barrier: Option<mirtal::Array>,
    partial_kernel: Option<mirtal::PreparedAliasing<9, 3>>,
    reduce_dispatch: Option<mirtal::Dispatch>,
    reduce_output: Option<[mirtal::OutputSpec; 1]>,
}

impl PagedAttentionScratch {
    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, ScratchState>> {
        self.state.lock().map_or_else(
            |_| Err(Error::InvalidModel("paged attention scratch lock was poisoned".into())),
            Ok,
        )
    }
}

impl ScratchState {
    pub(crate) fn prepare(
        &mut self,
        spec: ScratchSpec,
        stream: &mirtal::Stream,
        library: &mirtal::MetalLibrary,
        function: &'static str,
    ) -> Result<()> {
        if self.spec == Some(spec) {
            return Ok(());
        }
        let graph = stream.graph();
        let partial_shape =
            mirtal::Shape::new([1, spec.query_heads, 1, spec.blocks, spec.head_dim])?;
        let statistics_shape = mirtal::Shape::new([1, spec.query_heads, 1, spec.blocks])?;
        self.partials = Some(graph.full(&partial_shape, 0.0, spec.dtype)?);
        self.sums = Some(graph.full(&statistics_shape, 0.0, mirtal::DType::Float32)?);
        self.maximums = Some(graph.full(&statistics_shape, 0.0, mirtal::DType::Float32)?);
        self.barrier = None;
        let dispatch = mirtal::AliasingDispatch::new([5, 6, 7])
            .constants([
                u32::try_from(spec.query_heads)?,
                u32::try_from(spec.kv_heads)?,
                u32::try_from(spec.page_capacity)?,
                u32::try_from(spec.blocks)?,
                u32::try_from(spec.page_size)?,
                spec.scale_bits,
            ])
            .grid([32, spec.query_heads, spec.blocks])
            .threadgroup([32, spec.query_heads / spec.kv_heads, 1]);
        self.partial_kernel = Some(library.export(function)?.prepare_aliasing(dispatch)?);
        self.reduce_dispatch = Some(
            mirtal::Dispatch::new(
                [32 * spec.reduction_groups, spec.query_heads, 1],
                [32 * spec.reduction_groups, 1, 1],
            )
            .templates([
                mirtal::TemplateArg::dtype("T", spec.dtype),
                mirtal::TemplateArg::int("HEAD_DIM", i32::try_from(spec.head_dim)?),
                mirtal::TemplateArg::int("BLOCKS", i32::try_from(spec.blocks)?),
                mirtal::TemplateArg::int("REDUCTION_GROUPS", i32::try_from(spec.reduction_groups)?),
            ]),
        );
        self.reduce_output = Some([mirtal::OutputSpec::new(
            mirtal::Shape::new([1, spec.query_heads, 1, spec.head_dim])?,
            spec.dtype,
        )]);
        self.spec = Some(spec);
        Ok(())
    }

    pub(crate) fn reduce(&self) -> Result<(&mirtal::Dispatch, &[mirtal::OutputSpec; 1])> {
        Ok((
            self.reduce_dispatch
                .as_ref()
                .ok_or(Error::NullHandle("paged attention reduce dispatch"))?,
            self.reduce_output
                .as_ref()
                .ok_or(Error::NullHandle("paged attention reduce output"))?,
        ))
    }

    pub(crate) fn partial<'a>(
        &'a mut self,
        initial_barrier: &'a mirtal::Array,
    ) -> Result<(&'a mut mirtal::PreparedAliasing<9, 3>, [&'a mirtal::Array; 4])> {
        let kernel = self
            .partial_kernel
            .as_mut()
            .ok_or(Error::NullHandle("paged attention partial kernel"))?;
        let arrays = [
            self.partials.as_ref().ok_or(Error::NullHandle("paged attention partials"))?,
            self.sums.as_ref().ok_or(Error::NullHandle("paged attention sums"))?,
            self.maximums.as_ref().ok_or(Error::NullHandle("paged attention maximums"))?,
            self.barrier.as_ref().unwrap_or(initial_barrier),
        ];
        Ok((kernel, arrays))
    }

    pub(crate) fn update(
        &mut self,
        [partials, sums, maximums]: [mirtal::Array; 3],
        barrier: mirtal::Array,
    ) {
        self.partials = Some(partials);
        self.sums = Some(sums);
        self.maximums = Some(maximums);
        self.barrier = Some(barrier);
    }
}
