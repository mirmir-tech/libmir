use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use mircuda::{
    CublasGemmOffsets, CublasGemmOperand, CublasGemmSpec, DeviceBuffer, DeviceElement, bf16, f16,
};
use models::{
    decision::{DecisionTensor, DecisionTensorPlan},
    weights::TensorInfo,
};

use super::{
    device::Device,
    plans::{Input, Operands},
};
use crate::{CudaBackend, Error, Result};

/// Whether a product replaces its output or is added to it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Write {
    Overwrite,
    Accumulate,
}

/// `y = x Wᵀ + b` with an `[outputs, inputs]` weight on the device; a bf16
/// weight multiplies bf16 inputs on tensor cores into f32 products.
pub struct DeviceLinear<I: Input> {
    weight: DeviceBuffer<I>,
    bias: Option<DeviceBuffer<f32>>,
    pub inputs: usize,
    pub outputs: usize,
}

/// Layer normalisation; a bias-free norm carries a zero bias.
pub struct DeviceNorm {
    pub weight: DeviceBuffer<f32>,
    pub bias: DeviceBuffer<f32>,
    pub epsilon: f32,
}

impl<I: Input> DeviceLinear<I> {
    /// Writes `rows` products into `output`, sized `rows × outputs`.
    pub fn apply(
        &self,
        device: &Device<'_>,
        input: &DeviceBuffer<I>,
        output: &mut DeviceBuffer<f32>,
        write: Write,
    ) -> Result<()> {
        let rows = input.len() / self.inputs;
        let operand = |leading, transposed| CublasGemmOperand { leading, stride: 0, transposed };
        let spec = CublasGemmSpec::new(
            (rows, self.outputs, self.inputs, 1),
            operand(self.inputs, false),
            operand(self.inputs, true),
            operand(self.outputs, false),
        )?;
        let beta = match write {
            Write::Overwrite => 0.0,
            Write::Accumulate => 1.0,
        };
        let operands = Operands {
            left: input,
            right: &self.weight,
            output: &mut *output,
            offsets: CublasGemmOffsets::default(),
        };
        device.plans.multiply(device.backend, spec, operands, (1.0, beta))?;
        self.bias
            .as_ref()
            .map_or(Ok(()), |bias| device.elementwise.add_bias(device.stream(), output, bias))
    }
}

/// Uploads planned tensors, decoding every compute weight to f32 on the host.
pub struct Uploader<'a> {
    pub backend: &'a CudaBackend,
    pub plan: &'a DecisionTensorPlan,
}

impl Uploader<'_> {
    pub fn float(&self, role: DecisionTensor) -> Result<DeviceBuffer<f32>> {
        let info = self.plan.get(role)?;
        let values = decode(info, &payload(info)?)?;
        self.copy(&values)
    }

    /// The token table stays in its f16 checkpoint encoding.
    pub fn half(&self, role: DecisionTensor) -> Result<DeviceBuffer<f16>> {
        let info = self.plan.get(role)?;
        if info.dtype != "F16" {
            return Err(Error::DTypeMismatch { name: info.name.clone(), expected: "F16" });
        }
        let values: Vec<f16> = payload(info)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| f16::from_le_bytes(*chunk))
            .collect();
        self.copy(&values)
    }

    pub fn linear<I: Input>(
        &self,
        weight: DecisionTensor,
        bias: Option<DecisionTensor>,
    ) -> Result<DeviceLinear<I>> {
        let info = self.plan.get(weight)?;
        let &[outputs, inputs] = info.shape.as_slice() else {
            return Err(Error::InvalidDecoderKernel("decision linear weight is not a matrix"));
        };
        let values: Vec<I> = decode(info, &payload(info)?)?.into_iter().map(I::from_f32).collect();
        Ok(DeviceLinear {
            weight: self.copy(&values)?,
            bias: bias.map(|bias| self.float(bias)).transpose()?,
            inputs,
            outputs,
        })
    }

    pub fn norm(
        &self,
        weight: DecisionTensor,
        bias: Option<DecisionTensor>,
        epsilon: f32,
    ) -> Result<DeviceNorm> {
        let width = self.plan.get(weight)?.shape.iter().product();
        let bias = match bias {
            Some(bias) => self.float(bias)?,
            None => self.copy(&vec![0.0_f32; width])?,
        };
        Ok(DeviceNorm {
            weight: self.float(weight)?,
            bias,
            epsilon,
        })
    }

    fn copy<T: DeviceElement>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        upload(self.backend, values)
    }
}

/// Copies host values into a new device buffer and waits for the copy.
pub fn upload<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let inner = &backend.inner;
    let mut host = inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = inner.pool.allocate(&inner.stream, values.len())?;
    inner.stream.copy_to_device(&mut host, &mut device)?;
    inner.stream.synchronize()?;
    Ok(device)
}

fn payload(info: &TensorInfo) -> Result<Vec<u8>> {
    let mut file = File::open(&info.file)?;
    let _position = file.seek(SeekFrom::Start(info.payload_start()?))?;
    let mut bytes = vec![0; info.payload_bytes()?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn decode(info: &TensorInfo, bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(match info.dtype.as_str() {
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect(),
        "F16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| f16::from_le_bytes(*chunk).to_f32())
            .collect(),
        "BF16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| bf16::from_le_bytes(*chunk).to_f32())
            .collect(),
        _ => {
            return Err(Error::DTypeMismatch {
                name: info.name.clone(),
                expected: "F32, F16 or BF16",
            });
        },
    })
}
