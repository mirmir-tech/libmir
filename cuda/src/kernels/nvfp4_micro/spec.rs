use mircuda::{DeviceBuffer, bf16};

use super::{GatedActivation, NvFp4BankView};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NvFp4MicroSpec {
    pub experts: usize,
    pub selected: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub tokens: usize,
    pub activation: GatedActivation,
}

impl NvFp4MicroSpec {
    pub(super) fn validate(self) -> Result<()> {
        let valid = self.experts > 0
            && self.selected > 0
            && self.selected <= self.experts
            && self.hidden > 0
            && self.intermediate > 0
            && self.tokens > 0
            && self.hidden.is_multiple_of(16)
            && self.intermediate.is_multiple_of(16);
        if valid {
            self.groups().map(|_| ())
        } else {
            Err(Error::InvalidNvFp4("invalid micro expert geometry"))
        }
    }

    pub(super) fn groups(self) -> Result<usize> {
        self.tokens
            .checked_mul(self.selected)
            .ok_or(Error::InvalidNvFp4("micro expert count overflow"))
    }
}

#[derive(Clone, Copy)]
pub struct NvFp4MicroBanks<'a> {
    pub gate: NvFp4BankView<'a>,
    pub up: NvFp4BankView<'a>,
    pub down: NvFp4BankView<'a>,
}

pub struct NvFp4MicroWorkspace<'a> {
    pub gate_packed: &'a mut DeviceBuffer<u8>,
    pub up_packed: &'a mut DeviceBuffer<u8>,
    pub gate_scales: &'a mut DeviceBuffer<u8>,
    pub up_scales: &'a mut DeviceBuffer<u8>,
    pub intermediate_packed: &'a mut DeviceBuffer<u8>,
    pub intermediate_scales: &'a mut DeviceBuffer<u8>,
}

pub struct NvFp4MicroLaunch<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub selected: &'a DeviceBuffer<u32>,
    pub routing: &'a DeviceBuffer<bf16>,
    pub banks: NvFp4MicroBanks<'a>,
    pub workspace: NvFp4MicroWorkspace<'a>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

pub struct NvFp4MicroGateWorkspace<'a> {
    pub gate_packed: &'a mut DeviceBuffer<u8>,
    pub up_packed: &'a mut DeviceBuffer<u8>,
    pub gate_scales: &'a mut DeviceBuffer<u8>,
    pub up_scales: &'a mut DeviceBuffer<u8>,
    pub output_packed: &'a mut DeviceBuffer<u8>,
    pub output_scales: &'a mut DeviceBuffer<u8>,
}

pub struct NvFp4MicroGateLaunch<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub selected: &'a DeviceBuffer<u32>,
    pub banks: NvFp4MicroBanks<'a>,
    pub workspace: NvFp4MicroGateWorkspace<'a>,
    pub output_scale_stride: usize,
}

pub struct NvFp4MicroDownWorkspace<'a> {
    pub packed: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
}

pub struct NvFp4MicroDownLaunch<'a> {
    pub gate: &'a DeviceBuffer<bf16>,
    pub up: &'a DeviceBuffer<bf16>,
    pub selected: &'a DeviceBuffer<u32>,
    pub routing: &'a DeviceBuffer<bf16>,
    pub down: NvFp4BankView<'a>,
    pub workspace: NvFp4MicroDownWorkspace<'a>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

pub(super) fn require_len(name: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::QuantizedGemvLengthMismatch { operand: name, expected, actual })
    }
}
