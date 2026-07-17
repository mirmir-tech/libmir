use mircuda::{DeviceBuffer, bf16};

pub(super) type KvArguments<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a mut DeviceBuffer<u8>,
    &'a mut DeviceBuffer<u8>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
);

pub(super) type AttentionArguments<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<u8>,
    &'a DeviceBuffer<u8>,
    &'a DeviceBuffer<u32>,
    &'a mut DeviceBuffer<bf16>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
    u32,
);
