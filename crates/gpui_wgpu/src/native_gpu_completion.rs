use crate::stream_contract::StreamImageBudgets;
use anyhow::{Result, ensure};
use std::{
    any::Any,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::ThreadId,
    time::Duration,
};

static QUARANTINE: OnceLock<Mutex<Vec<Arc<dyn Any + Send + Sync>>>> = OnceLock::new();
static POLLS: AtomicU64 = AtomicU64::new(0);

pub(crate) trait AdmittedResource: Any + Send + Sync {
    fn budgets(&self) -> StreamImageBudgets;
    fn quarantined(&self) -> &AtomicBool;
}

pub(crate) fn quarantine<R: AdmittedResource>(resources: &Arc<R>) {
    resources.budgets().revoke();
    if !resources.quarantined().swap(true, Ordering::AcqRel) {
        QUARANTINE
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(resources.clone());
    }
}

pub(crate) struct GpuCompletion<R> {
    device: Arc<wgpu::Device>,
    owning_thread: ThreadId,
    _submission: wgpu::SubmissionIndex,
    done: Arc<AtomicBool>,
    resources: Arc<R>,
}

impl<R> Clone for GpuCompletion<R> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            owning_thread: self.owning_thread,
            _submission: self._submission.clone(),
            done: self.done.clone(),
            resources: self.resources.clone(),
        }
    }
}

impl<R: AdmittedResource> GpuCompletion<R> {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: &wgpu::Queue,
        owning_thread: ThreadId,
        submission: wgpu::SubmissionIndex,
        resources: Arc<R>,
    ) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let callback_done = done.clone();
        let retained = resources.clone();
        queue.on_submitted_work_done(move || {
            callback_done.store(true, Ordering::Release);
            drop(retained);
        });
        Self { device, owning_thread, _submission: submission, done, resources }
    }

    pub fn completed(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    pub fn wait(&self) -> Result<()> {
        ensure!(
            std::thread::current().id() != self.owning_thread,
            "GPU completion cannot wait on its UI thread"
        );
        POLLS.fetch_add(1, Ordering::Relaxed);
        // Callbacks cover the newest submission at registration. A snapshot wait
        // also covers shared-window work submitted before this poll; later work
        // does not extend the target. A stuck driver honoring the timeout is unproven.
        if let Err(error) = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        }) {
            quarantine(&self.resources);
            anyhow::bail!("native completion failed; retained admission: {error}");
        }
        if !self.completed() {
            quarantine(&self.resources);
            anyhow::bail!("native completion callback was not acknowledged; retained admission");
        }
        Ok(())
    }

    pub fn retire_in_background(self) {
        let retained = self.resources.clone();
        if let Err(error) =
            std::thread::Builder::new().name("native-image-retirement".into()).spawn(move || {
                if let Err(error) = self.wait() {
                    eprintln!("native retirement: {error:#}");
                }
            })
        {
            quarantine(&retained);
            eprintln!("native retirement worker could not start: {error}");
        }
    }
}

pub(crate) fn poll_count() -> u64 {
    POLLS.load(Ordering::Acquire)
}
