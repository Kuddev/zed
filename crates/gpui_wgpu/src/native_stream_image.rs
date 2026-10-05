use crate::native_gpu_completion::{AdmittedResource, GpuCompletion, quarantine};
use crate::stream_contract::{BackgroundShaderCancellation, StreamImageBudgets, StreamImageLease};
use anyhow::{Result, ensure};
use gpui::StreamImageFrame;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::ThreadId,
};

static TEXTURES: AtomicU64 = AtomicU64::new(0);
static BUFFERS: AtomicU64 = AtomicU64::new(0);
static UPLOADS: AtomicU64 = AtomicU64::new(0);
static MAPS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
pub(crate) struct StreamDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    owning_thread: ThreadId,
    epoch: Arc<AtomicU64>,
    lost: Arc<AtomicBool>,
}

pub(crate) struct StreamFactory {
    device: StreamDevice,
    owner: u64,
    size: [u32; 2],
    pitch: u32,
    known_bytes: u64,
    budgets: StreamImageBudgets,
    cancellation: BackgroundShaderCancellation,
    epoch: u64,
}

pub(crate) struct PreparedStream {
    image: NativeStream,
    epoch: u64,
    cancellation: BackgroundShaderCancellation,
}

struct Resources {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    staging: wgpu::Buffer,
    mapped: AtomicBool,
    map_failed: AtomicBool,
    lease: StreamImageLease,
    quarantined: AtomicBool,
}

impl AdmittedResource for Resources {
    fn budgets(&self) -> StreamImageBudgets {
        self.lease.budgets()
    }
    fn quarantined(&self) -> &AtomicBool {
        &self.quarantined
    }
}

pub(crate) struct NativeStream {
    device: StreamDevice,
    owner: u64,
    size: [u32; 2],
    pitch: u32,
    resources: Arc<Resources>,
    sequence: Option<u64>,
    write: Option<Completion>,
    last_submission: Option<wgpu::SubmissionIndex>,
}

#[derive(Clone)]
pub(crate) struct Completion {
    receipt: GpuCompletion<Resources>,
    resources: Arc<Resources>,
    require_mapping: bool,
}

impl StreamDevice {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, lost: Arc<AtomicBool>) -> Self {
        Self {
            device,
            queue,
            lost,
            owning_thread: std::thread::current().id(),
            epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn invalidate(&self) -> Result<()> {
        self.epoch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| value.checked_add(1))
            .map_err(|_| anyhow::anyhow!("stream device epoch exhausted"))?;
        Ok(())
    }

    pub fn factory(
        &self,
        owner: u64,
        size: [u32; 2],
        budgets: StreamImageBudgets,
        cancellation: BackgroundShaderCancellation,
    ) -> Result<StreamFactory> {
        ensure!(
            std::thread::current().id() == self.owning_thread,
            "stream factory capture must run on UI"
        );
        ensure!(owner != 0 && owner < 0x8000_0000, "stream texture identity exhausted");
        ensure!(size.iter().all(|value| *value != 0 && *value <= self.device.limits().max_texture_dimension_2d),
            "invalid stream dimensions");
        let row = size[0].checked_mul(4).ok_or_else(|| anyhow::anyhow!("stream row overflow"))?;
        let pitch = row
            .checked_add(255)
            .map(|row| row / 256 * 256)
            .ok_or_else(|| anyhow::anyhow!("stream pitch overflow"))?;
        let buffer = u64::from(pitch)
            .checked_mul(u64::from(size[1]))
            .ok_or_else(|| anyhow::anyhow!("stream buffer overflow"))?;
        ensure!(
            buffer <= self.device.limits().max_buffer_size,
            "stream staging exceeds device limit"
        );
        let known_bytes = u64::from(row)
            .checked_mul(u64::from(size[1]))
            .and_then(|texture| texture.checked_add(buffer))
            .ok_or_else(|| anyhow::anyhow!("stream allocation overflow"))?;
        budgets.ensure_available()?;
        Ok(StreamFactory {
            device: self.clone(),
            owner,
            size,
            pitch,
            known_bytes,
            budgets,
            cancellation,
            epoch: self.epoch.load(Ordering::Acquire),
        })
    }

    pub fn adopt(&self, owner: u64, prepared: PreparedStream) -> Result<NativeStream> {
        ensure!(
            std::thread::current().id() == self.owning_thread,
            "stream adoption must run on UI"
        );
        ensure!(owner == prepared.image.owner, "prepared stream belongs to another owner");
        ensure!(!prepared.cancellation.is_cancelled(), "prepared stream source was cancelled");
        ensure!(!self.lost.load(Ordering::Acquire), "stream device lost");
        ensure!(
            prepared.epoch == self.epoch.load(Ordering::Acquire)
                && Arc::ptr_eq(&self.epoch, &prepared.image.device.epoch),
            "prepared stream has an obsolete device epoch"
        );
        prepared.image.resources.lease.budgets().ensure_available()?;
        Ok(prepared.image)
    }
}

impl StreamFactory {
    pub fn run(self) -> Result<Option<PreparedStream>> {
        self.run_with_allocation_gate(|| {})
    }

    pub fn run_with_allocation_gate(self, gate: impl FnOnce()) -> Result<Option<PreparedStream>> {
        ensure!(
            std::thread::current().id() != self.device.owning_thread,
            "stream allocation cannot run on UI"
        );
        if self.cancellation.is_cancelled() {
            return Ok(None);
        }
        ensure!(!self.device.lost.load(Ordering::Acquire), "stream device lost");
        let lease = self.budgets.reserve(self.known_bytes)?;
        let device = &self.device.device;
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("admitted BGRA stream target"),
            size: wgpu::Extent3d {
                width: self.size[0],
                height: self.size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("admitted reusable stream staging"),
            size: u64::from(self.pitch) * u64::from(self.size[1]),
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        TEXTURES.fetch_add(1, Ordering::Relaxed);
        BUFFERS.fetch_add(1, Ordering::Relaxed);
        let resources = Arc::new(Resources {
            texture,
            view,
            staging,
            lease,
            mapped: AtomicBool::new(true),
            map_failed: AtomicBool::new(false),
            quarantined: AtomicBool::new(false),
        });
        gate();
        let errors: Vec<_> = [internal, memory, validation]
            .into_iter()
            .filter_map(|scope| pollster::block_on(scope.pop()))
            .collect();
        ensure!(errors.is_empty(), "native stream preparation: {errors:?}");
        if self.cancellation.is_cancelled() {
            return Ok(None);
        }
        self.budgets.ensure_available()?;
        Ok(Some(PreparedStream {
            image: NativeStream {
                device: self.device.clone(),
                owner: self.owner,
                size: self.size,
                pitch: self.pitch,
                resources,
                sequence: None,
                write: None,
                last_submission: None,
            },
            epoch: self.epoch,
            cancellation: self.cancellation,
        }))
    }
}

impl NativeStream {
    pub fn sequence(&self) -> Option<u64> {
        self.sequence
    }
    pub fn view(&self) -> wgpu::TextureView {
        self.resources.view.clone()
    }
    pub fn texture(&self) -> &wgpu::Texture {
        &self.resources.texture
    }

    pub fn stage(&mut self, frame: &StreamImageFrame<'_>) -> Result<Option<Completion>> {
        ensure!(
            std::thread::current().id() == self.device.owning_thread,
            "stream staging must run on UI"
        );
        frame.validate()?;
        self.resources.lease.budgets().ensure_available()?;
        ensure!(!self.device.lost.load(Ordering::Acquire), "stream device lost");
        ensure!(
            [frame.size.width.0 as u32, frame.size.height.0 as u32] == self.size,
            "stream dimensions changed; create a new source owner"
        );
        if self.sequence.is_some_and(|sequence| frame.sequence <= sequence) {
            return Ok(None);
        }
        ensure!(!self.resources.map_failed.load(Ordering::Acquire), "stream staging map failed");
        if let Some(write) = &self.write {
            if !write.receipt.completed() || !self.resources.mapped.load(Ordering::Acquire) {
                return Ok(Some(write.clone()));
            }
        }
        ensure!(self.resources.mapped.load(Ordering::Acquire), "stream staging is not prepared");
        {
            let mut destination = self.resources.staging.slice(..).get_mapped_range_mut();
            let row_bytes = self.size[0] as usize * 4;
            for row in 0..self.size[1] as usize {
                let source_start = row * frame.row_stride;
                let destination_start = row * self.pitch as usize;
                destination
                    .slice(destination_start..destination_start + row_bytes)
                    .copy_from_slice(&frame.pixels[source_start..source_start + row_bytes]);
            }
        }
        self.resources.mapped.store(false, Ordering::Release);
        self.resources.staging.unmap();
        let mut encoder =
            self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stream upload encoder"),
            });
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.resources.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.pitch),
                    rows_per_image: None,
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.resources.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: self.size[0], height: self.size[1], depth_or_array_layers: 1 },
        );
        let index = self.device.queue.submit([encoder.finish()]);
        self.last_submission = Some(index.clone());
        let completion = Completion::new(&self.device, index, self.resources.clone(), true);
        let retained = self.resources.clone();
        self.resources.staging.slice(..).map_async(wgpu::MapMode::Write, move |result| {
            if result.is_ok() {
                retained.mapped.store(true, Ordering::Release);
            } else {
                retained.map_failed.store(true, Ordering::Release);
                quarantine(&retained);
            }
        });
        MAPS.fetch_add(1, Ordering::Relaxed);
        UPLOADS.fetch_add(1, Ordering::Relaxed);
        self.sequence = Some(frame.sequence);
        self.write = Some(completion.clone());
        Ok(Some(completion))
    }

    pub fn note_scene_submission(&mut self, index: wgpu::SubmissionIndex) {
        self.last_submission = Some(index);
    }

    pub fn retire(mut self) -> Option<Completion> {
        self.last_submission
            .take()
            .map(|index| Completion::new(&self.device, index, self.resources.clone(), false))
    }
}

impl Drop for NativeStream {
    fn drop(&mut self) {
        if let Some(index) = self.last_submission.take() {
            Completion::new(&self.device, index, self.resources.clone(), false)
                .receipt
                .retire_in_background();
        }
    }
}

impl Completion {
    fn new(
        device: &StreamDevice,
        index: wgpu::SubmissionIndex,
        resources: Arc<Resources>,
        require_mapping: bool,
    ) -> Self {
        Self {
            receipt: GpuCompletion::new(
                device.device.clone(),
                &device.queue,
                device.owning_thread,
                index,
                resources.clone(),
            ),
            resources,
            require_mapping,
        }
    }

    pub fn wait(self) -> Result<()> {
        self.receipt.wait()?;
        if self.require_mapping
            && (!self.resources.mapped.load(Ordering::Acquire)
                || self.resources.map_failed.load(Ordering::Acquire))
        {
            quarantine(&self.resources);
            anyhow::bail!("native staging map was not acknowledged; retained admission");
        }
        Ok(())
    }
}

pub(crate) fn statistics() -> [u64; 4] {
    [
        TEXTURES.load(Ordering::Acquire),
        BUFFERS.load(Ordering::Acquire),
        UPLOADS.load(Ordering::Acquire),
        MAPS.load(Ordering::Acquire),
    ]
}
