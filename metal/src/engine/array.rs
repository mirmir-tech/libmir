use super::{Error, Result, Stream};

#[derive(Debug)]
pub struct Array {
    native: mirtal::Array,
}

impl Array {
    pub fn from_f32(data: &[f32], shape: &[i32]) -> Result<Self> {
        validate_shape(data.len(), shape)?;
        let shape = native_shape(shape)?;
        Self::from_native(mirtal::Array::from_shape(data, &shape)?)
    }

    pub fn from_u32(data: &[u32], shape: &[i32]) -> Result<Self> {
        validate_shape(data.len(), shape)?;
        let shape = native_shape(shape)?;
        Self::from_native(mirtal::Array::from_shape(data, &shape)?)
    }

    pub fn add(&self, right: &Self, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().add(&self.native, &right.native)?)
    }

    pub fn rms_norm_unit(&self, eps: f32, stream: &Stream) -> Result<Self> {
        let graph = stream.native().graph();
        let output = graph.rms_norm_unit(&self.native, eps)?;
        Self::from_native(graph.astype(&output, self.native.dtype()?)?)
    }

    pub fn async_eval(&self, stream: &Stream) -> Result<()> {
        Ok(stream.native().eval(&self.native)?)
    }

    pub(crate) fn detach_graph(&self, stream: &Stream) -> Result<()> {
        Ok(stream.native().detach_graph(&self.native)?)
    }

    pub fn to_vec_f32(&self, stream: &Stream) -> Result<Vec<f32>> {
        Ok(stream.native().read::<f32>(&self.native)?)
    }

    pub fn to_vec_u32(&self, stream: &Stream) -> Result<Vec<u32>> {
        Ok(stream.native().read::<u32>(&self.native)?)
    }

    pub fn item_u32(&self, stream: &Stream) -> Result<u32> {
        Ok(stream.native().read_scalar_u32(&self.native)?)
    }

    pub(crate) fn concatenate(inputs: &[&Self], axis: i32, stream: &Stream) -> Result<Self> {
        let inputs = inputs.iter().map(|input| input.native()).collect::<Vec<_>>();
        Self::from_native(stream.native().graph().concatenate(&inputs, axis)?)
    }

    pub(crate) fn slice(&self, start: &[usize], stop: &[usize], stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().slice(self.native(), start, stop)?)
    }

    pub fn slice_update(
        &self,
        update: &Self,
        start: &[usize],
        stop: &[usize],
        stream: &Stream,
    ) -> Result<Self> {
        Self::from_native(stream.native().graph().slice_update(
            self.native(),
            update.native(),
            start,
            stop,
        )?)
    }

    pub(crate) fn take(&self, indices: &Self, axis: i32, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().take(self.native(), indices.native(), axis)?)
    }

    pub(super) const fn native(&self) -> &mirtal::Array {
        &self.native
    }

    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn from_native(native: mirtal::Array) -> Result<Self> {
        Ok(Self { native })
    }
}

fn validate_shape(data: usize, shape: &[i32]) -> Result<()> {
    let elements = shape.iter().try_fold(1_usize, |total, dimension| {
        let dimension = usize::try_from(*dimension)?;
        total.checked_mul(dimension).ok_or(Error::ShapeOverflow)
    })?;
    if elements == data {
        return Ok(());
    }
    Err(Error::Shape { shape: shape.to_vec(), elements, data })
}

pub(super) fn native_shape(shape: &[i32]) -> Result<mirtal::Shape> {
    let dimensions = shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(mirtal::Shape::new(dimensions)?)
}
