use std::{
    sync::{Arc, Mutex, mpsc},
    thread::JoinHandle,
};

use foundation::model::ModelManifest;

use crate::{
    MetalConfig, MetalProgressEvent,
    native::{
        error::{Error, Result, WorkerFailure},
        model::LoadedModel,
    },
};

enum Task {
    Run(Box<dyn FnOnce(&mut LoadedModel) + Send + 'static>),
    Shutdown,
}

#[derive(Clone, Debug)]
pub(super) struct ModelClient {
    sender: mpsc::Sender<Task>,
    worker: Arc<Mutex<Option<JoinHandle<()>>>>,
}

enum StartEvent {
    Progress(MetalProgressEvent),
    Ready(std::result::Result<(), WorkerFailure>),
}

enum TaskEvent<T> {
    Progress(MetalProgressEvent),
    Complete(std::result::Result<T, WorkerFailure>),
}

impl ModelClient {
    pub(super) fn spawn(
        manifest: ModelManifest,
        config: Arc<MetalConfig>,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<Self> {
        let (task_sender, task_receiver) = mpsc::channel::<Task>();
        let (event_sender, event_receiver) = mpsc::channel();
        let name = format!("mirmir-metal-{}", manifest.id);
        let worker = std::thread::Builder::new().name(name).spawn(move || {
            let mut report = |event| {
                let _sent = event_sender.send(StartEvent::Progress(event));
            };
            let mut loaded = match LoadedModel::load_with_config(&manifest, config, &mut report) {
                Ok(loaded) => loaded,
                Err(error) => {
                    let _sent = event_sender.send(StartEvent::Ready(Err(error.into())));
                    return;
                },
            };
            if event_sender.send(StartEvent::Ready(Ok(()))).is_err() {
                return;
            }
            for task in task_receiver {
                match task {
                    Task::Run(task) => task(&mut loaded),
                    Task::Shutdown => break,
                }
            }
        });
        let worker = match worker {
            Ok(worker) => worker,
            Err(error) => return Err(Error::WorkerSpawn(error)),
        };
        loop {
            match event_receiver.recv()? {
                StartEvent::Progress(event) => progress(event),
                StartEvent::Ready(Ok(())) => {
                    return Ok(Self {
                        sender: task_sender,
                        worker: Arc::new(Mutex::new(Some(worker))),
                    });
                },
                StartEvent::Ready(Err(error)) => return Err(error.into()),
            }
        }
    }

    pub(super) fn run<T>(
        &self,
        run: impl FnOnce(&mut LoadedModel) -> Result<T> + Send + 'static,
    ) -> Result<T>
    where
        T: Send + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.send(Task::Run(Box::new(move |model| {
            let result = transferable(run(model));
            let _sent = sender.send(result);
        })))?;
        Ok(receiver.recv()??)
    }

    pub(super) fn run_with_progress<T>(
        &self,
        run: impl FnOnce(&mut LoadedModel, &mut dyn FnMut(MetalProgressEvent)) -> Result<T>
        + Send
        + 'static,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<T>
    where
        T: Send + 'static,
    {
        let (sender, receiver) = mpsc::channel();
        self.send(Task::Run(Box::new(move |model| {
            let mut report = |event| {
                let _sent = sender.send(TaskEvent::Progress(event));
            };
            let result = transferable(run(model, &mut report));
            let _sent = sender.send(TaskEvent::Complete(result));
        })))?;
        loop {
            match receiver.recv()? {
                TaskEvent::Progress(event) => progress(event),
                TaskEvent::Complete(result) => return Ok(result?),
            }
        }
    }

    fn send(&self, task: Task) -> Result<()> {
        if self.sender.send(task).is_err() {
            return Err(Error::WorkerClosed);
        }
        Ok(())
    }

    pub(super) fn shutdown(self) -> Result<()> {
        self.send(Task::Shutdown)?;
        let worker = self.worker.lock()?.take();
        if worker.is_some_and(|worker| worker.join().is_err()) {
            return Err(Error::WorkerJoin);
        }
        Ok(())
    }
}

fn transferable<T>(result: Result<T>) -> std::result::Result<T, WorkerFailure> {
    Ok(result?)
}
