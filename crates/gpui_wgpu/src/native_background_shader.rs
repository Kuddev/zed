use crate::native_gpu_completion::{AdmittedResource, GpuCompletion};
use crate::stream_contract::{BackgroundShaderCancellation, StreamImageBudgets, StreamImageLease};
use anyhow::{Result, ensure};
use std::{
    borrow::Cow,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::ThreadId,
};

const VERTEX: &str = r#"
@vertex fn background_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}
"#;
static PROGRAMS_CREATED: AtomicU64 = AtomicU64::new(0);
static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static TEXTURES_CREATED: AtomicU64 = AtomicU64::new(0);
static SHADER_DRAWS: AtomicU64 = AtomicU64::new(0);
static DEVICES: OnceLock<Mutex<Vec<Weak<DevicePrograms>>>> = OnceLock::new();

struct DevicePrograms {
    device: Arc<wgpu::Device>,
    programs: Mutex<Vec<Weak<Program>>>,
}
struct Program {
    source: Arc<str>,
    entry: Arc<str>,
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    _cache: Arc<DevicePrograms>,
}
struct Resources {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    program: Arc<Program>,
    lease: StreamImageLease,
    quarantined: AtomicBool,
}

#[derive(Clone)]
pub(crate) struct ShaderDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    cache: Arc<DevicePrograms>,
    owning_thread: ThreadId,
    epoch: Arc<AtomicU64>,
    lost: Arc<AtomicBool>,
}

pub(crate) struct ShaderFactory {
    device: ShaderDevice,
    owner: u64,
    size: [u32; 2],
    source: Arc<str>,
    entry: Arc<str>,
    format: wgpu::TextureFormat,
    budgets: StreamImageBudgets,
    cancellation: BackgroundShaderCancellation,
    epoch: u64,
}

pub(crate) struct PreparedShader {
    image: NativeShader,
    epoch: u64,
    cancellation: BackgroundShaderCancellation,
}
pub(crate) struct NativeShader {
    device: ShaderDevice,
    owner: u64,
    pub size: [u32; 2],
    resources: Arc<Resources>,
    sequence: Option<u64>,
    write: Option<Completion>,
    last_submission: Option<wgpu::SubmissionIndex>,
}
#[derive(Clone)]
pub(crate) struct Completion {
    receipt: GpuCompletion<Resources>,
}

impl ShaderDevice {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        lost: Arc<AtomicBool>,
    ) -> Result<Self> {
        let mut caches = DEVICES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .map_err(|_| anyhow::anyhow!("shader device cache poisoned"))?;
        caches.retain(|cache| cache.strong_count() != 0);
        let cache = if let Some(cache) = caches
            .iter()
            .filter_map(Weak::upgrade)
            .find(|cache| Arc::ptr_eq(&cache.device, &device))
        {
            cache
        } else {
            ensure!(caches.len() < 8, "shader device cache admission exceeded");
            let cache = Arc::new(DevicePrograms {
                device: device.clone(),
                programs: Mutex::new(Vec::new()),
            });
            caches.push(Arc::downgrade(&cache));
            cache
        };
        Ok(Self {
            device,
            queue,
            cache,
            owning_thread: std::thread::current().id(),
            epoch: Arc::new(AtomicU64::new(0)),
            lost,
        })
    }
    pub fn invalidate(&self) -> Result<()> {
        self.epoch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| value.checked_add(1))
            .map_err(|_| anyhow::anyhow!("background device epoch exhausted"))?;
        Ok(())
    }
    pub fn factory(
        &self,
        owner: u64,
        size: [u32; 2],
        source: Arc<str>,
        entry: Arc<str>,
        format: wgpu::TextureFormat,
        budgets: StreamImageBudgets,
        cancellation: BackgroundShaderCancellation,
    ) -> Result<ShaderFactory> {
        ensure!(owner != 0 && owner < 0x8000_0000, "stream texture identity exhausted");
        ensure!(size.iter().all(|value| *value != 0 && *value <= self.device.limits().max_texture_dimension_2d), "invalid shader dimensions");
        ensure!(
            matches!(format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm),
            "unsupported background format"
        );
        ensure!(
            !source.is_empty() && source.len() <= 64 * 1024,
            "WGSL exceeds its 64 KiB source limit"
        );
        ensure!(!entry.is_empty() && entry.len() <= 256, "invalid WGSL entry name");
        budgets.ensure_available()?;
        Ok(ShaderFactory {
            device: self.clone(),
            owner,
            size,
            source,
            entry,
            format,
            budgets,
            cancellation,
            epoch: self.epoch.load(Ordering::Acquire),
        })
    }
    pub fn adopt(&self, owner: u64, prepared: PreparedShader) -> Result<NativeShader> {
        ensure!(
            std::thread::current().id() == self.owning_thread,
            "background adoption must run on its UI thread"
        );
        ensure!(owner == prepared.image.owner, "prepared background belongs to another owner");
        ensure!(!prepared.cancellation.is_cancelled(), "prepared background source was cancelled");
        ensure!(!self.lost.load(Ordering::Acquire), "background device lost");
        ensure!(
            prepared.epoch == self.epoch.load(Ordering::Acquire)
                && Arc::ptr_eq(&self.epoch, &prepared.image.device.epoch),
            "prepared background has an obsolete device epoch"
        );
        prepared.image.resources.lease.budgets().ensure_available()?;
        Ok(prepared.image)
    }
}

impl ShaderFactory {
    pub fn run(self) -> Result<Option<PreparedShader>> {
        self.run_with_allocation_gate(|| {})
    }
    // The private lab supplies a channel gate after actual allocation; no sleep or
    // fault injection is read from product preferences or installed source files.
    pub fn run_with_allocation_gate(self, gate: impl FnOnce()) -> Result<Option<PreparedShader>> {
        ensure!(
            std::thread::current().id() != self.device.owning_thread,
            "background shader factory ran on its UI thread"
        );
        if self.cancellation.is_cancelled() {
            return Ok(None);
        }
        ensure!(!self.device.lost.load(Ordering::Acquire), "background device lost");
        let module = validate_fragment(&self.source, &self.entry)?;
        let bytes = u64::from(self.size[0])
            .checked_mul(u64::from(self.size[1]))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| anyhow::anyhow!("shader allocation overflow"))?;
        let lease = self.budgets.reserve(bytes)?;
        let device = &self.device.device;
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let result = (|| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("admitted background target"),
                size: wgpu::Extent3d {
                    width: self.size[0],
                    height: self.size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            TEXTURES_CREATED.fetch_add(1, Ordering::Relaxed);
            gate();
            if self.cancellation.is_cancelled() {
                return Ok(None);
            }
            let program = self.device.cache.program(
                self.source.clone(),
                self.entry.clone(),
                self.format,
                module,
            )?;
            if self.cancellation.is_cancelled() {
                return Ok(None);
            }
            self.budgets.ensure_available()?;
            let resources = Arc::new(Resources {
                texture,
                view,
                program,
                lease,
                quarantined: AtomicBool::new(false),
            });
            let image = NativeShader {
                device: self.device.clone(),
                owner: self.owner,
                size: self.size,
                resources,
                sequence: None,
                write: None,
                last_submission: None,
            };
            Ok(Some(PreparedShader {
                image,
                epoch: self.epoch,
                cancellation: self.cancellation.clone(),
            }))
        })();
        check_scopes([internal, memory, validation])?;
        result
    }
}

impl DevicePrograms {
    fn program(
        self: &Arc<Self>,
        source: Arc<str>,
        entry: Arc<str>,
        format: wgpu::TextureFormat,
        module: naga::Module,
    ) -> Result<Arc<Program>> {
        let mut programs =
            self.programs.lock().map_err(|_| anyhow::anyhow!("shader program cache poisoned"))?;
        programs.retain(|program| program.strong_count() != 0);
        if let Some(program) = programs.iter().filter_map(Weak::upgrade).find(|program| {
            program.source == source && program.entry == entry && program.format == format
        }) {
            CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            return Ok(program);
        }
        ensure!(programs.len() < 64, "shader program cache admission exceeded");
        let fragment = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("validated background WGSL"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
        });
        let vertex = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("controlled background vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(VERTEX)),
        });
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("no-binding background ABI"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prepared background WGSL"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vertex,
                entry_point: Some("background_vertex"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fragment,
                entry_point: Some(&entry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        check_scopes([internal, memory, validation])?;
        let program = Arc::new(Program { source, entry, format, pipeline, _cache: self.clone() });
        PROGRAMS_CREATED.fetch_add(1, Ordering::Relaxed);
        programs.push(Arc::downgrade(&program));
        Ok(program)
    }
}

impl NativeShader {
    pub fn view(&self) -> wgpu::TextureView {
        self.resources.view.clone()
    }
    pub fn texture(&self) -> &wgpu::Texture {
        &self.resources.texture
    }
    pub fn sequence(&self) -> Option<u64> {
        self.sequence
    }
    pub fn stage(
        &mut self,
        sequence: u64,
        size: [u32; 2],
        source: &str,
        entry: &str,
    ) -> Result<Option<Completion>> {
        ensure!(
            std::thread::current().id() == self.device.owning_thread,
            "shader staging must run on its UI thread"
        );
        self.resources.lease.budgets().ensure_available()?;
        ensure!(!self.device.lost.load(Ordering::Acquire), "background device lost");
        ensure!(
            size == self.size
                && source == self.resources.program.source.as_ref()
                && entry == self.resources.program.entry.as_ref(),
            "shader changed; create a new source owner"
        );
        if self.sequence.is_some_and(|old| sequence <= old) {
            return Ok(None);
        }
        if let Some(completion) = &self.write {
            if !completion.receipt.completed() {
                return Ok(Some(completion.clone()));
            }
        }
        let mut encoder =
            self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("background encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("background pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.resources.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.resources.program.pipeline);
            pass.draw(0..3, 0..1);
        }
        let index = self.device.queue.submit([encoder.finish()]);
        self.last_submission = Some(index.clone());
        self.write = Some(Completion::new(self.device.clone(), index, self.resources.clone()));
        self.sequence = Some(sequence);
        SHADER_DRAWS.fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }
    pub fn note_scene_submission(&mut self, index: wgpu::SubmissionIndex) {
        // Updating the last reader does not allocate another completion callback per
        // repaint. Retirement records one receipt covering the most recent reader.
        self.last_submission = Some(index);
    }
    pub fn completion(&self) -> Option<Completion> {
        self.write.clone()
    }
    pub fn retire(mut self) -> Option<Completion> {
        self.last_submission
            .take()
            .map(|index| Completion::new(self.device.clone(), index, self.resources.clone()))
    }
}

impl Drop for NativeShader {
    fn drop(&mut self) {
        if let Some(index) = self.last_submission.take() {
            // Atlas reset/drop also needs quiescence when no window executor remains.
            // Unsubmitted/cancelled factories never take this path or start a task.
            Completion::new(self.device.clone(), index, self.resources.clone())
                .receipt
                .retire_in_background();
        }
    }
}

impl Completion {
    fn new(device: ShaderDevice, index: wgpu::SubmissionIndex, resources: Arc<Resources>) -> Self {
        Self {
            receipt: GpuCompletion::new(
                device.device,
                &device.queue,
                device.owning_thread,
                index,
                resources,
            ),
        }
    }
    pub fn wait(self) -> Result<()> {
        self.receipt.wait()
    }
}

impl AdmittedResource for Resources {
    fn budgets(&self) -> StreamImageBudgets {
        self.lease.budgets()
    }
    fn quarantined(&self) -> &AtomicBool {
        &self.quarantined
    }
}

fn check_scopes(scopes: [wgpu::ErrorScopeGuard; 3]) -> Result<()> {
    let errors: Vec<_> =
        scopes.into_iter().filter_map(|scope| pollster::block_on(scope.pop())).collect();
    ensure!(errors.is_empty(), "native WGSL preparation: {errors:?}");
    Ok(())
}

pub(crate) fn validate_fragment(source: &str, entry: &str) -> Result<naga::Module> {
    ensure!(source.len() <= 64 * 1024, "WGSL exceeds its 64 KiB source limit");
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| anyhow::anyhow!(error.emit_to_string(source)))?;
    ensure!(
        !module.global_variables.iter().any(|(_, global)| global.binding.is_some()),
        "background binding ABI is not admitted"
    );
    ensure!(module.entry_points.len() == 1, "one fragment entry required");
    let point =
        module.entry_points.first().ok_or_else(|| anyhow::anyhow!("fragment entry required"))?;
    ensure!(
        point.stage == naga::ShaderStage::Fragment && point.name == entry,
        "fragment entry mismatch"
    );
    let output = point
        .function
        .result
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("location0 vec4 output required"))?;
    let color = |ty, binding: &Option<naga::Binding>| {
        matches!(binding, Some(naga::Binding::Location { location: 0, .. }))
            && matches!(
                module.types[ty].inner,
                naga::TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    scalar: naga::Scalar { kind: naga::ScalarKind::Float, width: 4 }
                }
            )
    };
    let valid_output = match &module.types[output.ty].inner {
        naga::TypeInner::Struct { members, .. } => {
            members.len() == 1 && color(members[0].ty, &members[0].binding)
        },
        _ => color(output.ty, &output.binding),
    };
    ensure!(valid_output, "only one location0 vec4 color output is admitted");
    ensure!(
        point.function.arguments.iter().all(|argument| matches!(
            argument.binding,
            Some(naga::Binding::BuiltIn(naga::BuiltIn::Position { .. }))
        )),
        "only fragment position input is admitted"
    );
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|error| anyhow::anyhow!("WGSL validation: {error}"))?;
    Ok(module)
}

pub(crate) fn statistics() -> [u64; 5] {
    [
        PROGRAMS_CREATED.load(Ordering::Acquire),
        CACHE_HITS.load(Ordering::Acquire),
        TEXTURES_CREATED.load(Ordering::Acquire),
        SHADER_DRAWS.load(Ordering::Acquire),
        crate::native_gpu_completion::poll_count(),
    ]
}
