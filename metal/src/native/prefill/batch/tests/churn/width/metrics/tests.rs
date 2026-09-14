use super::{Result, argmax};
use crate::engine::{Array, Stream};

#[test]
fn teacher_forcing_matches_device_argmax_on_ties() -> Result<()> {
    let stream = Stream::new_gpu()?;
    for values in [[-1.0, 3.0, 3.0, 0.0], [-0.0, 0.0, -0.0, 0.0], [-4.0, -2.0, -2.0, -3.0]] {
        let device = Array::from_f32(&values, &[1, 1, 4])?.argmax_u32(&stream)?;
        assert_eq!(argmax(&values)?, device, "teacher-forced token for {values:?}");
    }
    Ok(())
}
