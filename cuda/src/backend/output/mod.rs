mod affine;
mod batch;
mod block;
mod projection;
#[cfg(all(test, target_os = "linux"))]
mod tests;
mod weight;

pub use affine::CudaAffineOutputHead;
pub(super) use batch::CudaBatchOutputHead;
pub use projection::CudaOutputHead;
pub(super) use weight::CudaOutputHeadTemplate;
