use std::{
    collections::{HashMap, hash_map::Entry},
    fs::File,
    io::{Read, Seek, SeekFrom},
    sync::Mutex,
};

use mircuda::{CublasDenseSpec, CublasF32Plan, DeviceBuffer, DeviceElement, f16};
use models::{
    decision::{DecisionTensor, DecisionTensorPlan},
    weights::TensorInfo,
};

use crate::{CudaBackend, Error, Result, kernels::DecisionElementwise};

/// Whether a product replaces its output or is added to it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Write {
    Overwrite,
    Accumulate,
}

/// Fixed-shape f32 cuBLAS plans reused across calls; creating one costs an
/// allocation and a device synchronisation.
#[derive(Default)]
pub struct Plans {
    plans: Mutex<HashMap<CublasDenseSpec, CublasF32Plan>>,
}

impl Plans {
    fn multiply(
        &self,
        backend: &CudaBackend,
        spec: CublasDenseSpec,
        (input, weight, output): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
        beta: f32,
    ) -> Result<()> {
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| Error::InvalidDecoderKernel("decision plan cache is poisoned"))?;
        let stream = &backend.inner.stream;
        let plan = match plans.entry(spec) {
            Entry::Occupied(plan) => plan.into_mut(),
            Entry::Vacant(slot) => {
                slot.insert(CublasF32Plan::new(&backend.inner.context, stream, spec)?)
            },
        };
        let executed = plan.execute(stream, input, weight, output, 1.0, beta);
        drop(plans);
        Ok(executed?)
    }
}

/// `y = x Wᵀ + b` with an `[outputs, inputs]` f32 weight on the device.
pub struct DeviceLinear {
    weight: DeviceBuffer<f32>,
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

impl DeviceLinear {
    /// Writes `rows` products into `output`, sized `rows × outputs`.
    pub fn apply(
        &self,
        (backend, kernels, plans): (&CudaBackend, &DecisionElementwise, &Plans),
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        write: Write,
    ) -> Result<()> {
        let rows = input.len() / self.inputs;
        let spec = CublasDenseSpec::new(rows, self.outputs, self.inputs)?;
        let beta = match write {
            Write::Overwrite => 0.0,
            Write::Accumulate => 1.0,
        };
        plans.multiply(backend, spec, (input, &self.weight, output), beta)?;
        let stream = &backend.inner.stream;
        self.bias.as_ref().map_or(Ok(()), |bias| kernels.add_bias(stream, output, bias))
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

    pub fn linear(
        &self,
        weight: DecisionTensor,
        bias: Option<DecisionTensor>,
    ) -> Result<DeviceLinear> {
        let shape = &self.plan.get(weight)?.shape;
        let &[outputs, inputs] = shape.as_slice() else {
            return Err(Error::InvalidDecoderKernel("decision linear weight is not a matrix"));
        };
        Ok(DeviceLinear {
            weight: self.float(weight)?,
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
            .map(|chunk| mircuda::bf16::from_le_bytes(*chunk).to_f32())
            .collect(),
        _ => {
            return Err(Error::DTypeMismatch {
                name: info.name.clone(),
                expected: "F32, F16 or BF16",
            });
        },
    })
}
