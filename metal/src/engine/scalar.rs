use super::{Array, Result, Stream};

impl Array {
    pub fn add_scalar(&self, scalar: f32, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().add_scalar(self.native(), scalar)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjusts_tensor_values_on_the_gpu_stream() -> Result<()> {
        let stream = Stream::new_gpu()?;
        let input = Array::from_f32(&[1.0, 2.0], &[1, 2])?;
        let output = input.add_scalar(1.0, &stream)?;
        output.async_eval(&stream)?;
        stream.synchronize()?;
        assert_eq!(output.to_vec_f32(&stream)?, vec![2.0, 3.0]);
        Ok(())
    }
}
