use std::{
    sync::mpsc::{RecvError, TryRecvError},
    time::{Duration, Instant},
};

use super::{Command, Worker};

const CONTINUATION_POLL: Duration = Duration::from_micros(500);

impl Worker {
    /// Decode continuations return within a fraction of a millisecond; polling
    /// for them avoids an idle-state wake-up on every step.
    pub(super) fn next_command(&self) -> Result<Command, RecvError> {
        if !self.active_decode.is_empty() {
            let deadline = Instant::now() + CONTINUATION_POLL;
            while Instant::now() < deadline {
                match self.commands.try_recv() {
                    Ok(command) => return Ok(command),
                    Err(TryRecvError::Disconnected) => return Err(RecvError),
                    Err(TryRecvError::Empty) => std::hint::spin_loop(),
                }
            }
        }
        self.commands.recv()
    }

    pub(super) fn drain_commands(&mut self) {
        loop {
            match self.commands.try_recv() {
                Ok(command) => self.admit(command),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.stopping = true;
                    return;
                },
            }
        }
    }
}
