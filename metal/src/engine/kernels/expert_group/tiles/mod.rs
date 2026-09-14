use super::super::template;
use crate::engine::{Array, Dtype, Error, Result, Stream};

mirtal::metal_kernel! {
    fn worklist {
        name: "mirmir_expert_worklist",
        templates: [ROUTES: int = 64, EXPERTS: int = 8, CAPACITY: int = 12],
        inputs: [indices: u32], outputs: [tiles: u32],
        source: file "kernels/expert_group/tiles/worklist.metal",
        header: inline "", row_contiguous: true, atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    fn project {
        name: "mirmir_expert_tile_mxfp4",
        templates: [T: dtype = bf16, INPUT: int = 64, OUTPUT: int = 64],
        inputs: [input: T, weight: u32, scales: u8, tiles: u32], outputs: [output: T],
        source: file "kernels/expert_group/tiles/project.metal",
        header: inline "#include <metal_simdgroup_matrix>",
        row_contiguous: true, atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    fn project_wide {
        name: "mirmir_expert_tile_mxfp4_wide",
        templates: [T: dtype = bf16, INPUT: int = 64, OUTPUT: int = 64],
        inputs: [input: T, weight: u32, scales: u8, tiles: u32], outputs: [output: T],
        source: file "kernels/expert_group/tiles/wide.metal",
        header: inline "#include <metal_simdgroup_matrix>",
        row_contiguous: true, atomic_outputs: false,
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ColumnTile {
    Width32,
    Width64,
}

impl ColumnTile {
    pub(crate) const fn columns(self) -> usize {
        match self {
            Self::Width32 => 32,
            Self::Width64 => 64,
        }
    }
}

#[derive(Debug)]
pub struct TilePlan {
    worklist: mirtal::MetalKernel<1, 1>,
    projection: mirtal::MetalKernel<4, 1>,
    columns: ColumnTile,
}

pub struct WorkTiles {
    descriptors: mirtal::Array,
    routes: usize,
    experts: usize,
    capacity: usize,
}

impl TilePlan {
    pub(crate) fn new() -> Result<Self> {
        Self::with_columns(ColumnTile::Width32)
    }

    pub(crate) const fn columns(&self) -> ColumnTile {
        self.columns
    }

    pub(crate) fn with_columns(columns: ColumnTile) -> Result<Self> {
        Ok(Self {
            worklist: worklist()?,
            projection: match columns {
                ColumnTile::Width32 => project()?,
                ColumnTile::Width64 => project_wide()?,
            },
            columns,
        })
    }

    /// Internal contract: IDs are sorted, in range, and describe compact rows.
    pub(crate) fn prepare(
        &self,
        indices: &Array,
        experts: usize,
        stream: &Stream,
    ) -> Result<WorkTiles> {
        let routes = indices.native().len();
        if indices.dtype()? != Dtype::Uint32 || routes == 0 || !(1..=1024).contains(&experts) {
            return Err(Error::InvalidModel("unsupported expert tile routing".into()));
        }
        let capacity = routes.div_ceil(16).checked_add(experts).ok_or(Error::ShapeOverflow)?;
        let output =
            mirtal::OutputSpec::new(mirtal::Shape::new([capacity, 3])?, mirtal::DType::Uint32);
        let [descriptors] = self.worklist.dispatch(
            stream.native(),
            [indices.native()],
            &[output],
            &mirtal::Dispatch::new([experts, 1, 1], [experts, 1, 1]).templates([
                template("ROUTES", routes)?,
                template("EXPERTS", experts)?,
                template("CAPACITY", capacity)?,
            ]),
        )?;
        Ok(WorkTiles { descriptors, routes, experts, capacity })
    }

    pub(crate) fn project(
        &self,
        input: &Array,
        weight: &Array,
        scales: &Array,
        tiles: &WorkTiles,
        stream: &Stream,
    ) -> Result<Array> {
        let ws = weight.shape()?;
        let xs = input.shape()?;
        let [experts, width, packed] = ws.as_slice() else {
            return Err(Error::InvalidModel("expert tiles require a weight bank".into()));
        };
        let k = usize::try_from(*packed)?.checked_mul(8).ok_or(Error::ShapeOverflow)?;
        let n = usize::try_from(*width)?;
        if input.dtype()? != Dtype::Bfloat16
            || weight.dtype()? != Dtype::Uint32
            || scales.dtype()? != Dtype::Uint8
            || usize::try_from(*experts)? != tiles.experts
            || xs != [i32::try_from(tiles.routes)?, 1, i32::try_from(k)?]
            || k == 0
            || !k.is_multiple_of(32)
            || n == 0
            || !n.is_multiple_of(self.columns.columns())
            || scales.shape()? != [*experts, *width, i32::try_from(k / 32)?]
        {
            return Err(Error::InvalidModel(
                "unsupported expert tile projection shape or type".into(),
            ));
        }
        let output = mirtal::OutputSpec::new(
            mirtal::Shape::new([tiles.routes, 1, n])?,
            mirtal::DType::Bfloat16,
        );
        let [output] = self.projection.dispatch(
            stream.native(),
            [input.native(), weight.native(), scales.native(), &tiles.descriptors],
            &[output],
            &mirtal::Dispatch::new(
                [n / self.columns.columns() * 64, tiles.capacity, 1],
                [64, 1, 1],
            )
            .templates([
                mirtal::TemplateArg::dtype("T", mirtal::DType::Bfloat16),
                template("INPUT", k)?,
                template("OUTPUT", n)?,
            ]),
        )?;
        Array::from_native(output)
    }
}

#[cfg(test)]
mod tests;

impl WorkTiles {
    pub(crate) fn descriptors(&self) -> &mirtal::Array {
        &self.descriptors
    }
}

impl super::super::Kernels {
    pub(crate) const fn expert_tiles(&self) -> &TilePlan {
        &self.expert_tiles
    }
}
