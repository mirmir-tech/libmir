mod completion;
mod worker;

use std::{
    sync::{
        Arc, Mutex,
        mpsc::{self, Sender},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use runtime::{
    backend::{DecodeSequence, ModelHandle, PrefillOutput, PrefillRequest},
    kv::CacheConfig,
    progress::ProgressEvent,
    scheduler::SchedulerConfig,
};

use super::{prefill::PrefillResponse, response::DecodeResponse};
use crate::{Engine, Result, engine::EnginePrefillBatch};

pub(super) struct GenerationCoordinator {
    commands: Sender<Command>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

pub(super) enum Command {
    Decode(PendingDecode),
    Prefill(PendingPrefill),
    Release(uuid::Uuid),
    Stop,
}

pub(super) struct PendingDecode {
    pub(super) sequence: DecodeSequence,
    pub(super) response: Arc<DecodeResponse>,
    pub(super) enqueued: Instant,
    pub(super) scheduler_queue: Duration,
    pub(super) newly_active: bool,
}

pub(super) struct PendingPrefill {
    pub(super) request: PrefillRequest,
    pub(super) response: Arc<PrefillResponse>,
    pub(super) enqueued: Instant,
    pub(super) scheduler_queue: Duration,
    pub(super) expects_decode: bool,
}

pub(super) struct ActivePrefill {
    pub(super) batch: EnginePrefillBatch,
    pub(super) requests: Vec<PendingPrefill>,
}

impl GenerationCoordinator {
    pub(super) fn new(
        engine: Engine,
        model: ModelHandle,
        config: SchedulerConfig,
        cache: CacheConfig,
    ) -> Result<Self> {
        let prefill_profile =
            engine.generation_prefill_profile(&model, config.max_batch_tokens, cache)?;
        let (commands, receiver) = mpsc::channel();
        let worker =
            std::thread::Builder::new().name("libmir-generation".into()).spawn(move || {
                worker::Worker::new(engine, model, config, receiver, prefill_profile).run();
            });
        let worker = match worker {
            Ok(worker) => worker,
            Err(error) => {
                return Err(runtime::RuntimeError::Scheduler(format!(
                    "could not start accelerator generation worker: {error}"
                ))
                .into());
            },
        };
        Ok(Self {
            commands,
            worker: Mutex::new(Some(worker)),
        })
    }

    pub(super) fn start_decode(&self, sequence: DecodeSequence) -> Result<Arc<DecodeResponse>> {
        let response = Arc::new(DecodeResponse::new());
        self.send(Command::Decode(PendingDecode {
            sequence,
            response: response.clone(),
            enqueued: Instant::now(),
            scheduler_queue: Duration::ZERO,
            newly_active: false,
        }))?;
        Ok(response)
    }

    pub(super) fn submit_prefill(
        &self,
        request: PrefillRequest,
        expects_decode: bool,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        let response = Arc::new(PrefillResponse::new());
        self.send(Command::Prefill(PendingPrefill {
            request,
            response: response.clone(),
            enqueued: Instant::now(),
            scheduler_queue: Duration::ZERO,
            expects_decode,
        }))?;
        response.wait(progress)
    }

    pub(super) fn release(&self, session: uuid::Uuid) {
        let _sent = self.commands.send(Command::Release(session));
    }

    fn send(&self, command: Command) -> Result<()> {
        if self.commands.send(command).is_err() {
            return Err(super::scheduler_error("accelerator generation worker stopped"));
        }
        Ok(())
    }
}

impl Drop for GenerationCoordinator {
    fn drop(&mut self) {
        let _sent = self.commands.send(Command::Stop);
        if let Ok(worker) = self.worker.get_mut()
            && let Some(worker) = worker.take()
        {
            let _joined = worker.join();
        }
    }
}
