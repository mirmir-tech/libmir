use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{Error, Result};

cuda_export!(LogitShapingKernel = "libmir_cuda_logit_shaping_bf16"(
    values: &mut DeviceBuffer<bf16>, elements: u32, divisor: f32, cap: f32,
));

/// What the model does to its output logits: divides them (Granite
/// `logits_scaling`), then soft-caps them (Gemma `final_logit_softcapping`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogitShaping {
    pub divisor: f32,
    pub softcap: Option<f32>,
}

impl LogitShaping {
    /// The shaping of `divisor` and `softcap`; `None` when it leaves the
    /// logits as they are.
    pub(crate) fn new(divisor: Option<f64>, softcap: Option<f64>) -> Result<Option<Self>> {
        let narrow = |value: f64| -> Result<f32> { Ok(value.to_string().parse()?) };
        let divisor = divisor.map(narrow).transpose()?.unwrap_or(1.0);
        let softcap = softcap.map(narrow).transpose()?;
        #[allow(clippy::float_cmp)]
        let identity = divisor == 1.0 && softcap.is_none();
        let valid = divisor.is_finite()
            && divisor > 0.0
            && softcap.is_none_or(|cap| cap.is_finite() && cap > 0.0);
        if !valid {
            return Err(Error::InvalidDecoderKernel("invalid logit divisor or softcap"));
        }
        Ok((!identity).then_some(Self { divisor, softcap }))
    }
}

#[derive(Clone, Debug)]
pub struct LogitShapingBf16 {
    kernel: TypedKernel<LogitShapingKernel>,
    elements: usize,
    shaping: LogitShaping,
}

impl LogitShapingBf16 {
    pub(crate) fn compile(
        compiler: &Compiler,
        elements: usize,
        shaping: LogitShaping,
    ) -> Result<Self> {
        if elements == 0 {
            return Err(Error::InvalidDecoderKernel("empty logit shaping"));
        }
        let source = cuda_kernel_file!("../../../kernels/logit_shaping_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            elements,
            shaping,
        })
    }

    pub(crate) fn execute(&self, stream: &Stream, values: &mut DeviceBuffer<bf16>) -> Result<()> {
        if values.len() != self.elements {
            return Err(Error::InvalidDecoderKernel("logit shaping buffer width differs"));
        }
        let threads = 256_usize;
        let cap = self.shaping.softcap.unwrap_or(0.0);
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(self.elements.div_ceil(threads))?, 1, 1),
                block: (u32::try_from(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (values, u32::try_from(self.elements)?, self.shaping.divisor, cap),
        )?)
    }
}

#[cfg(test)]
mod tests;
