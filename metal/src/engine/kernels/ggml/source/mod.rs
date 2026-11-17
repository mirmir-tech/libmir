//! The GGML kernel declarations: each family is compiled twice, with and
//! without the IQ grid tables, so other types do not carry them.

use models::gguf::GgmlType;

mod conversions;
mod products;

use conversions::{
    ggml_dequantize, ggml_dequantize_grids, ggml_embedding, ggml_embedding_grids, ggml_mxfp4_repack,
};
use products::{
    ggml_gathered, ggml_gathered_grids, ggml_gemv, ggml_gemv_grids, ggml_grouped,
    ggml_grouped_grids, ggml_mma, ggml_mma_grids,
};

use super::uses_grids;
use crate::engine::Result;

/// One kernel family: decoders without grid tables, and with them.
#[derive(Debug)]
pub(super) struct Family<const INPUTS: usize> {
    plain: mirtal::MetalKernel<INPUTS, 1>,
    grids: mirtal::MetalKernel<INPUTS, 1>,
}

impl<const INPUTS: usize> Family<INPUTS> {
    pub(super) const fn select(&self, ggml_type: GgmlType) -> &mirtal::MetalKernel<INPUTS, 1> {
        if uses_grids(ggml_type) {
            &self.grids
        } else {
            &self.plain
        }
    }
}

#[derive(Debug)]
pub(in crate::engine::kernels) struct GgmlKernels {
    pub(super) gemv: Family<2>,
    pub(super) mma: Family<2>,
    pub(super) gathered: Family<3>,
    pub(super) grouped: Family<3>,
    pub(super) dequantize: Family<1>,
    pub(super) embedding: Family<2>,
    pub(super) mxfp4_repack: mirtal::MetalKernel<1, 2>,
}

impl GgmlKernels {
    pub(in crate::engine::kernels) fn new() -> Result<Self> {
        Ok(Self {
            gemv: Family {
                plain: ggml_gemv()?,
                grids: ggml_gemv_grids()?,
            },
            mma: Family {
                plain: ggml_mma()?,
                grids: ggml_mma_grids()?,
            },
            gathered: Family {
                plain: ggml_gathered()?,
                grids: ggml_gathered_grids()?,
            },
            grouped: Family {
                plain: ggml_grouped()?,
                grids: ggml_grouped_grids()?,
            },
            dequantize: Family {
                plain: ggml_dequantize()?,
                grids: ggml_dequantize_grids()?,
            },
            embedding: Family {
                plain: ggml_embedding()?,
                grids: ggml_embedding_grids()?,
            },
            mxfp4_repack: ggml_mxfp4_repack()?,
        })
    }
}
