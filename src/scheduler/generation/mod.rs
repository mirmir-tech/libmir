mod completion;
mod worker;

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
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
    interrupt: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

pub(super) enum Command {
    Decode(PendingDecode),
    Prefill(PendingPrefill),
    Release(uuid::Uuid),
    Cancellation,
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
    pub(super) cancellation: crate::CancellationToken,
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
        if config.prefill_refill_policy == runtime::scheduler::PrefillRefillPolicy::ShortPrompt
            && config.prefill_decode_policy
                == runtime::scheduler::PrefillDecodePolicy::CompleteCohort
        {
            return Err(runtime::RuntimeError::Config(
                "short_prompt refill requires backend_default prefill_decode_policy".into(),
            )
            .into());
        }
        let prefill_profile = engine.generation_prefill_profile(
            &model,
            config.max_batch_tokens,
            cache,
            config.prefill_refill_policy,
        )?;
        let (commands, receiver) = mpsc::channel();
        let interrupt = Arc::new(AtomicBool::new(false));
        let worker_interrupt = interrupt.clone();
        let worker =
            std::thread::Builder::new().name("libmir-generation".into()).spawn(move || {
                worker::Worker::new(
                    engine, model, config, receiver, prefill_profile, worker_interrupt,
                )
                .run();
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
            interrupt,
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
        cancellation: &crate::CancellationToken,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        let response = Arc::new(PrefillResponse::new());
        self.send(Command::Prefill(PendingPrefill {
            request,
            response: response.clone(),
            enqueued: Instant::now(),
            scheduler_queue: Duration::ZERO,
            expects_decode,
            cancellation: cancellation.clone(),
        }))?;
        response.wait_cancellable(progress, cancellation, || {
            self.interrupt.store(true, Ordering::Release);
            self.send(Command::Cancellation)
        })
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
