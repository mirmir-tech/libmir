use super::{
    DenseGateUpLayout, SelectedDenseDispatch, SelectedDenseGateLaunch, SelectedDenseMoe,
    SelectedDenseReduceLaunch,
};
use crate::{
    Result,
    kernels::geometry::{product, require},
};

impl SelectedDenseMoe {
    pub(super) fn validate_dispatch(
        &self,
        selected: &mircuda::DeviceBuffer<u32>,
        dispatch: &SelectedDenseDispatch<'_>,
    ) -> Result<()> {
        let spec = self.spec();
        let assignments = product(spec.tokens, spec.selected_count)?;
        require("dense dispatch selected", assignments, selected.len())?;
        require("dense dispatch counts", spec.expert_count, dispatch.counts.len())?;
        require("dense dispatch offsets", spec.expert_count, dispatch.offsets.len())?;
        require("dense dispatch cursors", spec.expert_count, dispatch.cursors.len())?;
        require("dense dispatch assignments", assignments, dispatch.assignments.len())?;
        require("dense dispatch experts", assignments, dispatch.experts.len())
    }

    pub(super) fn validate_gated(&self, launch: &SelectedDenseGateLaunch<'_>) -> Result<()> {
        let spec = self.spec();
        require(
            "dense expert input",
            product(spec.tokens, spec.input_features)?,
            launch.input.len(),
        )?;
        require(
            "dense selected experts",
            product(spec.tokens, spec.selected_count)?,
            launch.selected.len(),
        )?;
        let rows = if spec.gate_up_layout == DenseGateUpLayout::Separate {
            spec.output_features
        } else {
            product(spec.output_features, 2)?
        };
        let fused = product(product(spec.expert_count, rows)?, spec.input_features)?;
        let separate =
            product(product(spec.expert_count, spec.output_features)?, spec.input_features)?;
        require("dense gate weight", fused, launch.gate_weight.len())?;
        require(
            "dense up weight",
            if spec.gate_up_layout == DenseGateUpLayout::Separate {
                separate
            } else {
                fused
            },
            launch.up_weight.len(),
        )?;
        self.validate_bias("dense gate bias", launch.gate_bias.len(), rows, spec.gate_bias)?;
        self.validate_bias("dense up bias", launch.up_bias.len(), rows, spec.up_bias)?;
        require(
            "dense gated output",
            product(product(spec.tokens, spec.selected_count)?, spec.output_features)?,
            launch.output.len(),
        )
    }

    pub(super) fn validate_reduce(&self, launch: &SelectedDenseReduceLaunch<'_>) -> Result<()> {
        let spec = self.spec();
        require(
            "dense activated experts",
            product(product(spec.tokens, spec.selected_count)?, spec.output_features)?,
            launch.input.len(),
        )?;
        let selections = product(spec.tokens, spec.selected_count)?;
        require("dense selected experts", selections, launch.selected.len())?;
        require("dense routing weights", selections, launch.routing.len())?;
        if (spec.down_transposed && spec.tokens == 1) || self.prefers_expert_major() {
            require(
                "dense down partials",
                product(selections, spec.input_features)?,
                launch.partial.len(),
            )?;
        }
        require(
            "dense down weight",
            product(product(spec.expert_count, spec.input_features)?, spec.output_features)?,
            launch.weight.len(),
        )?;
        self.validate_bias(
            "dense down bias",
            launch.bias.len(),
            spec.input_features,
            spec.down_bias,
        )?;
        require(
            "dense expert output",
            product(spec.tokens, spec.input_features)?,
            launch.output.len(),
        )
    }

    fn validate_bias(
        &self,
        name: &'static str,
        actual: usize,
        rows: usize,
        present: bool,
    ) -> Result<()> {
        if present {
            require(name, product(self.spec().expert_count, rows)?, actual)
        } else {
            Ok(())
        }
    }
}
