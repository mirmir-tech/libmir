use super::Kernels;
use crate::engine::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub struct PageWriteOptions {
    pub(crate) sequence: usize,
    pub(crate) offset: usize,
    pub(crate) kv_heads: usize,
    pub(crate) page_capacity: usize,
    pub(crate) page_size: usize,
    pub(crate) head_dim: usize,
}

#[derive(Debug, Default)]
pub struct PreparedPageWrite {
    kernel: Option<(mirtal::DType, mirtal::PreparedAliasing<5, 2>)>,
}

impl PreparedPageWrite {
    fn bind(
        &mut self,
        library: &mirtal::MetalLibrary,
        dtype: mirtal::DType,
        function: &'static str,
        options: PageWriteOptions,
    ) -> Result<&mut mirtal::PreparedAliasing<5, 2>> {
        let elements = options
            .sequence
            .checked_mul(options.kv_heads)
            .and_then(|value| value.checked_mul(options.head_dim))
            .ok_or(Error::ShapeOverflow)?;
        let constants = [
            u32::try_from(options.sequence)?,
            u32::try_from(options.offset)?,
            u32::try_from(options.kv_heads)?,
            u32::try_from(options.page_capacity)?,
            u32::try_from(options.page_size)?,
            u32::try_from(options.head_dim)?,
        ];
        let grid = [elements, 1, 1];
        let threadgroup = [elements.min(256), 1, 1];
        match self.kernel.as_mut() {
            Some((prepared_dtype, kernel)) if *prepared_dtype == dtype => {
                kernel.rebind(&constants, grid, threadgroup)?;
            },
            Some(_) => {
                return Err(Error::InvalidModel(
                    "paged K/V dtype changed after preparing page-write".into(),
                ));
            },
            None => {
                let dispatch = mirtal::AliasingDispatch::new([2, 3])
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
                let kernel = library.export(function)?.prepare_aliasing(dispatch)?;
                self.kernel = Some((dtype, kernel));
            },
        }
        self.kernel
            .as_mut()
            .map(|(_, kernel)| kernel)
            .ok_or(Error::NullHandle("prepared page-write kernel"))
    }
}

impl Kernels {
    pub(crate) fn page_write(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 5],
        options: PageWriteOptions,
        prepared: &mut PreparedPageWrite,
    ) -> Result<[mirtal::Array; 2]> {
        let dtype = inputs[0].dtype()?;
        let function = match dtype {
            mirtal::DType::Float32 => "mirmir_page_write_f32",
            mirtal::DType::Float16 => "mirmir_page_write_f16",
            mirtal::DType::Bfloat16 => "mirmir_page_write_bf16",
            _ => return Err(Error::InvalidModel("paged K/V dtype is unsupported".into())),
        };
        Ok(prepared
            .bind(&self.paged_kv, dtype, function, options)?
            .dispatch(stream, inputs)?)
    }
}
