//! A routed layer's router bound to its weights: the projection to expert
//! scores, the layer's top-k with its correction bias, and the selections it
//! writes for a fixed number of rows.

use mircuda::{DeviceBuffer, bf16};

use super::{LayerRouterTopK, RouterSelection};
use crate::{
    CudaBackend, DenseRole, Error, Result,
    backend::linear::{CheckpointProjection, CheckpointProjectionWeight},
    kernels::{RouterUnitScoring, RouterUnitSpec},
};

#[derive(Debug)]
pub(in crate::backend) struct BoundRouter {
    projection: Box<CheckpointProjection>,
    top_k: LayerRouterTopK,
    scores: DeviceBuffer<bf16>,
    selected: DeviceBuffer<u32>,
    routing: DeviceBuffer<bf16>,
}

/// The layer weights a router reads.
#[derive(Clone, Copy)]
pub(in crate::backend) struct RouterWeights<'a> {
    pub(in crate::backend) projection: &'a CheckpointProjectionWeight,
    /// A sigmoid router's `[experts]` correction bias.
    pub(in crate::backend) correction_bias: Option<&'a DeviceBuffer<f32>>,
}

impl BoundRouter {
    /// Routes `spec.tokens` rows of `hidden` features to `spec.top_k` of
    /// `spec.experts` experts.
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        (spec, hidden): (RouterUnitSpec, usize),
        scoring: RouterUnitScoring,
        weights: RouterWeights<'_>,
    ) -> Result<Self> {
        let elements = |width: usize| {
            spec.tokens
                .checked_mul(width)
                .ok_or(Error::InvalidRouter("router selection size overflow"))
        };
        let allocate = |width| -> Result<DeviceBuffer<bf16>> {
            Ok(backend.pool().allocate(backend.stream(), elements(width)?)?)
        };
        Ok(Self {
            projection: Box::new(CheckpointProjection::new(
                backend,
                spec.tokens,
                hidden,
                spec.experts,
                DenseRole::Router,
                weights.projection,
            )?),
            top_k: LayerRouterTopK::compile(
                backend.compiler(),
                spec,
                scoring,
                weights.correction_bias,
            )?,
            scores: allocate(spec.experts)?,
            selected: backend.pool().allocate(backend.stream(), elements(spec.top_k)?)?,
            routing: allocate(spec.top_k)?,
        })
    }

    /// Scores `input` and selects its experts.
    pub(in crate::backend) fn route(
        &mut self,
        backend: &CudaBackend,
        input: &DeviceBuffer<bf16>,
    ) -> Result<RouterSelection<'_>> {
        self.projection.execute(input, &mut self.scores)?;
        self.top_k.execute(
            backend.stream(),
            &self.scores,
            &mut self.selected,
            &mut self.routing,
        )?;
        Ok(self.selection())
    }

    /// The selections the last [`Self::route`] wrote.
    pub(in crate::backend) const fn selection(&self) -> RouterSelection<'_> {
        RouterSelection {
            indices: &self.selected,
            weights: &self.routing,
        }
    }
}
