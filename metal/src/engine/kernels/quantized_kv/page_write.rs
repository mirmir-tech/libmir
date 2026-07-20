use super::QuantizedKvKernels;
use crate::engine::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub struct QuantizedPageWriteOptions {
    pub(crate) sequence: usize,
    pub(crate) offset: usize,
    pub(crate) kv_heads: usize,
    pub(crate) page_capacity: usize,
    pub(crate) page_size: usize,
    pub(crate) head_dim: usize,
}

#[derive(Debug, Default)]
pub struct PreparedQuantizedPageWrite {
    kernel: Option<(mirtal::DType, mirtal::PreparedAliasing<7, 4>)>,
}

impl PreparedQuantizedPageWrite {
    fn bind(
        &mut self,
        library: &mirtal::MetalLibrary,
        dtype: mirtal::DType,
        function: &'static str,
        options: QuantizedPageWriteOptions,
    ) -> Result<&mut mirtal::PreparedAliasing<7, 4>> {
        let constants = [
            u32::try_from(options.sequence)?,
            u32::try_from(options.offset)?,
            u32::try_from(options.kv_heads)?,
            u32::try_from(options.page_capacity)?,
            u32::try_from(options.page_size)?,
            u32::try_from(options.head_dim)?,
        ];
        let grid = [32, options.kv_heads, options.sequence];
        let threadgroup = [32, 1, 1];
        match self.kernel.as_mut() {
            Some((prepared_dtype, kernel)) if *prepared_dtype == dtype => {
                kernel.rebind(&constants, grid, threadgroup)?;
            },
            Some(_) => {
                return Err(Error::InvalidModel(
                    "paged K/V dtype changed after preparing INT8 page-write".into(),
                ));
            },
            None => {
                let dispatch = mirtal::AliasingDispatch::new([2, 3, 4, 5])
                    .constants(constants)
                    .strides([
                        mirtal::StrideBinding { input: 0, axis: 1 },
                        mirtal::StrideBinding { input: 0, axis: 2 },
                        mirtal::StrideBinding { input: 0, axis: 3 },
                        mirtal::StrideBinding { input: 1, axis: 1 },
                        mirtal::StrideBinding { input: 1, axis: 2 },
                        mirtal::StrideBinding { input: 1, axis: 3 },
                    ])
                    .grid(grid)
                    .threadgroup(threadgroup);
                self.kernel = Some((dtype, library.export(function)?.prepare_aliasing(dispatch)?));
            },
        }
        self.kernel
            .as_mut()
            .map(|(_, kernel)| kernel)
            .ok_or(Error::NullHandle("prepared INT8 page-write kernel"))
    }
}

impl QuantizedKvKernels {
    pub(super) fn page_write(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 7],
        options: QuantizedPageWriteOptions,
        prepared: &mut PreparedQuantizedPageWrite,
    ) -> Result<[mirtal::Array; 4]> {
        let dtype = inputs[0].dtype()?;
        let function = match dtype {
            mirtal::DType::Float32 => "mirmir_quantized_page_write_f32",
            mirtal::DType::Float16 => "mirmir_quantized_page_write_f16",
            mirtal::DType::Bfloat16 => "mirmir_quantized_page_write_bf16",
            _ => return Err(Error::InvalidModel("INT8 K/V input dtype is unsupported".into())),
        };
        Ok(prepared
            .bind(&self.page_write, dtype, function, options)?
            .dispatch(stream, inputs)?)
    }
}
