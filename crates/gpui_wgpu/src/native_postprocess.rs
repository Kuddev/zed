use crate::native_background_shader::check_scopes;
use crate::native_gpu_completion::{AdmittedResource, GpuCompletion};
use crate::stream_contract::{
    BackgroundShaderCancellation, StreamImageBudgets, StreamImageLease, WgslPostprocessDescriptor,
};
use anyhow::{Result, ensure};
use gpui::{Bounds, ScaledPixels};
use std::{
    borrow::Cow,
    num::NonZeroU64,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::ThreadId,
};

const VERTEX: &str = r#"
@vertex fn effect_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}"#;
static DEVICES: OnceLock<Mutex<Vec<Weak<DevicePrograms>>>> = OnceLock::new();

struct DevicePrograms {
    device: Arc<wgpu::Device>,
    programs: Mutex<Vec<Weak<Program>>>,
}
struct Program {
    source: Arc<str>,
    entry: Arc<str>,
    uniform_size: usize,
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    _cache: Arc<DevicePrograms>,
}
struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}
struct Pass {
    program: Arc<Program>,
    inputs: [wgpu::BindGroup; 2],
}
struct Resources {
    targets: [Target; 2],
    uniforms: wgpu::Buffer,
    passes: Vec<Pass>,
    lease: StreamImageLease,
    quarantined: AtomicBool,
}

#[derive(Clone)]
pub(crate) struct PostprocessDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    cache: Arc<DevicePrograms>,
    owning_thread: ThreadId,
    epoch: Arc<AtomicU64>,
    lost: Arc<AtomicBool>,
}
pub(crate) struct PostprocessFactory {
    device: PostprocessDevice,
    owner: u64,
    descriptor: WgslPostprocessDescriptor,
    format: wgpu::TextureFormat,
    budgets: StreamImageBudgets,
    cancellation: BackgroundShaderCancellation,
    epoch: u64,
}
pub(crate) struct PreparedPostprocess {
    image: NativePostprocess,
    cancellation: BackgroundShaderCancellation,
    epoch: u64,
}
pub(crate) struct NativePostprocess {
    device: PostprocessDevice,
    owner: u64,
    size: [u32; 2],
    uniform_size: usize,
    format: wgpu::TextureFormat,
    resources: Arc<Resources>,
    last_submission: Option<wgpu::SubmissionIndex>,
}
pub(crate) struct Completion(GpuCompletion<Resources>);
impl Completion {
    pub fn wait(self) -> Result<()> {
        self.0.wait()
    }
}

impl PostprocessDevice {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        lost: Arc<AtomicBool>,
    ) -> Result<Self> {
        let mut caches = DEVICES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .map_err(|_| anyhow::anyhow!("postprocess device cache poisoned"))?;
        caches.retain(|cache| cache.strong_count() != 0);
        let cache = if let Some(cache) = caches
            .iter()
            .filter_map(Weak::upgrade)
            .find(|cache| Arc::ptr_eq(&cache.device, &device))
        {
            cache
        } else {
            ensure!(caches.len() < 8, "postprocess device admission exceeded");
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
            .map_err(|_| anyhow::anyhow!("postprocess device epoch exhausted"))?;
        Ok(())
    }

    pub fn factory(
        &self,
        owner: u64,
        descriptor: WgslPostprocessDescriptor,
        format: wgpu::TextureFormat,
        budgets: StreamImageBudgets,
        cancellation: BackgroundShaderCancellation,
    ) -> Result<PostprocessFactory> {
        ensure!(
            std::thread::current().id() == self.owning_thread,
            "postprocess factory capture must run on UI"
        );
        descriptor.validate()?;
        ensure!(owner != 0 && owner < 0x8000_0000, "postprocess identity exhausted");
        ensure!(
            descriptor.size.width.0 as u32 <= self.device.limits().max_texture_dimension_2d
                && descriptor.size.height.0 as u32 <= self.device.limits().max_texture_dimension_2d,
            "postprocess extent exceeds device limits"
        );
        ensure!(
            descriptor.uniform_size as u64 <= self.device.limits().max_uniform_buffer_binding_size,
            "postprocess uniforms exceed device limits"
        );
        ensure!(
            matches!(format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm),
            "postprocess requires an unorm color surface"
        );
        budgets.ensure_available()?;
        Ok(PostprocessFactory {
            device: self.clone(),
            owner,
            descriptor,
            format,
            budgets,
            cancellation,
            epoch: self.epoch.load(Ordering::Acquire),
        })
    }

    pub fn adopt(&self, owner: u64, prepared: PreparedPostprocess) -> Result<NativePostprocess> {
        ensure!(
            std::thread::current().id() == self.owning_thread,
            "postprocess adoption must run on UI"
        );
        ensure!(owner == prepared.image.owner, "prepared effect belongs to another owner");
        ensure!(!prepared.cancellation.is_cancelled(), "prepared effect source was cancelled");
        ensure!(!self.lost.load(Ordering::Acquire), "postprocess device lost");
        ensure!(
            prepared.epoch == self.epoch.load(Ordering::Acquire)
                && Arc::ptr_eq(&self.epoch, &prepared.image.device.epoch),
            "prepared effect has an obsolete device epoch"
        );
        prepared.image.resources.lease.budgets().ensure_available()?;
        Ok(prepared.image)
    }
}

impl PostprocessFactory {
    pub fn run(self) -> Result<Option<PreparedPostprocess>> {
        self.run_with_allocation_gate(|| {})
    }

    pub fn run_with_allocation_gate(
        self,
        gate: impl FnOnce(),
    ) -> Result<Option<PreparedPostprocess>> {
        ensure!(
            std::thread::current().id() != self.device.owning_thread,
            "postprocess allocation must run off UI"
        );
        if self.cancellation.is_cancelled() {
            return Ok(None);
        }
        ensure!(!self.device.lost.load(Ordering::Acquire), "postprocess device lost");
        let lease = self
            .budgets
            .reserve(self.descriptor.texture_bytes()? + self.descriptor.uniform_size as u64)?;
        let device = &self.device.device;
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let result = (|| {
            let size = [self.descriptor.size.width.0 as u32, self.descriptor.size.height.0 as u32];
            let make_target = || {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("admitted postprocess target"),
                    size: wgpu::Extent3d {
                        width: size[0],
                        height: size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                Target { texture, view }
            };
            let targets = [make_target(), make_target()];
            let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("admitted postprocess uniforms"),
                size: self.descriptor.uniform_size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gate();
            if self.cancellation.is_cancelled() {
                return Ok(None);
            }
            let mut passes = Vec::new();
            for pass in self.descriptor.passes.iter() {
                let program = self.device.cache.program(
                    pass.source.clone(),
                    pass.entry.clone(),
                    self.descriptor.uniform_size,
                    self.format,
                )?;
                let inputs = targets.each_ref().map(|target| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("postprocess frame and input"),
                        layout: &program.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniforms.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&target.view),
                            },
                        ],
                    })
                });
                passes.push(Pass { program, inputs });
                if self.cancellation.is_cancelled() {
                    return Ok(None);
                }
            }
            self.budgets.ensure_available()?;
            Ok(Some(PreparedPostprocess {
                image: NativePostprocess {
                    device: self.device.clone(),
                    owner: self.owner,
                    size,
                    uniform_size: self.descriptor.uniform_size,
                    format: self.format,
                    resources: Arc::new(Resources {
                        targets,
                        uniforms,
                        passes,
                        lease,
                        quarantined: AtomicBool::new(false),
                    }),
                    last_submission: None,
                },
                cancellation: self.cancellation.clone(),
                epoch: self.epoch,
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
        uniform_size: usize,
        format: wgpu::TextureFormat,
    ) -> Result<Arc<Program>> {
        let mut programs = self
            .programs
            .lock()
            .map_err(|_| anyhow::anyhow!("postprocess program cache poisoned"))?;
        programs.retain(|program| program.strong_count() != 0);
        if let Some(program) = programs.iter().filter_map(Weak::upgrade).find(|program| {
            program.source == source
                && program.entry == entry
                && program.uniform_size == uniform_size
                && program.format == format
        }) {
            return Ok(program);
        }
        ensure!(programs.len() < 64, "postprocess program admission exceeded");
        let module = validate_fragment(&source, &entry, uniform_size)?;
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let fragment = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("validated postprocess WGSL"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
        });
        let vertex = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("controlled postprocess vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(VERTEX)),
        });
        let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postprocess ABI"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(uniform_size as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("postprocess pipeline ABI"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prepared postprocess WGSL"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex,
                entry_point: Some("effect_vertex"),
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
        let program = Arc::new(Program {
            source,
            entry,
            uniform_size,
            format,
            pipeline,
            layout,
            _cache: self.clone(),
        });
        programs.push(Arc::downgrade(&program));
        Ok(program)
    }
}

impl NativePostprocess {
    pub fn render(
        &mut self,
        screen: &wgpu::Texture,
        bounds: Bounds<ScaledPixels>,
        mask: Bounds<ScaledPixels>,
        uniforms: &[u8],
    ) -> Result<()> {
        ensure!(
            std::thread::current().id() == self.device.owning_thread,
            "postprocess rendering must run on UI"
        );
        self.resources.lease.budgets().ensure_available()?;
        ensure!(!self.device.lost.load(Ordering::Acquire), "postprocess device lost");
        ensure!(uniforms.len() == self.uniform_size, "postprocess uniform layout changed");
        ensure!(
            screen.format() == self.format
                && screen.sample_count() == 1
                && screen
                    .usage()
                    .contains(wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST),
            "postprocess requires a copyable matching surface"
        );
        ensure!(
            bounds.size.width.0 == self.size[0] as f32
                && bounds.size.height.0 == self.size[1] as f32,
            "postprocess extent changed before replacement"
        );
        let x = bounds.origin.x.0;
        let y = bounds.origin.y.0;
        ensure!(
            x.is_finite() && y.is_finite() && x.fract() == 0.0 && y.fract() == 0.0,
            "postprocess origin must use physical pixels"
        );
        ensure!(
            [mask.origin.x.0, mask.origin.y.0, mask.size.width.0, mask.size.height.0]
                .iter()
                .all(|value| value.is_finite()),
            "invalid postprocess content mask"
        );
        let window = Bounds::new(
            gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            gpui::size(ScaledPixels(screen.width() as f32), ScaledPixels(screen.height() as f32)),
        );
        let visible = bounds.intersect(&window).intersect(&mask);
        if visible.is_empty() {
            return Ok(());
        }
        let left = visible.left().0.ceil().max(0.0) as u32;
        let top = visible.top().0.ceil().max(0.0) as u32;
        let right = visible.right().0.floor().min(screen.width() as f32) as u32;
        let bottom = visible.bottom().0.floor().min(screen.height() as f32) as u32;
        if left >= right || top >= bottom {
            return Ok(());
        }
        let local =
            wgpu::Origin3d { x: (left as f32 - x) as u32, y: (top as f32 - y) as u32, z: 0 };
        let screen_origin = wgpu::Origin3d { x: left, y: top, z: 0 };
        let extent =
            wgpu::Extent3d { width: right - left, height: bottom - top, depth_or_array_layers: 1 };
        ensure!(
            local.x + extent.width <= self.size[0] && local.y + extent.height <= self.size[1],
            "postprocess copy exceeds local extent"
        );
        let mut encoder =
            self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scoped postprocess encoder"),
            });
        {
            // 排除的输入先清透明，防止用户着色器采样到相邻面板或被遮挡的内容。
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear scoped input"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.resources.targets[0].view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }
        encoder.copy_texture_to_texture(
            texture_copy(screen, screen_origin),
            texture_copy(&self.resources.targets[0].texture, local),
            extent,
        );
        // 前置场景已提交；独立提交保证同一所有者在一帧内重复使用时不被后一次 uniforms 覆盖。
        self.device.queue.write_buffer(&self.resources.uniforms, 0, uniforms);
        for (index, stage) in self.resources.passes.iter().enumerate() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scoped WGSL pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.resources.targets[(index + 1) % 2].view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&stage.program.pipeline);
            pass.set_bind_group(0, &stage.inputs[index % 2], &[]);
            pass.draw(0..3, 0..1);
        }
        let output = &self.resources.targets[self.resources.passes.len() % 2].texture;
        encoder.copy_texture_to_texture(
            texture_copy(output, local),
            texture_copy(screen, screen_origin),
            extent,
        );
        self.last_submission = Some(self.device.queue.submit([encoder.finish()]));
        Ok(())
    }

    pub fn retire(mut self) -> Option<Completion> {
        self.last_submission.take().map(|index| Completion(self.completion(index)))
    }
    fn completion(&self, index: wgpu::SubmissionIndex) -> GpuCompletion<Resources> {
        GpuCompletion::new(
            self.device.device.clone(),
            &self.device.queue,
            self.device.owning_thread,
            index,
            self.resources.clone(),
        )
    }
}

fn texture_copy(texture: &wgpu::Texture, origin: wgpu::Origin3d) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin, aspect: wgpu::TextureAspect::All }
}

impl Drop for NativePostprocess {
    fn drop(&mut self) {
        if let Some(index) = self.last_submission.take() {
            self.completion(index).retire_in_background();
        }
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

fn validate_fragment(source: &str, entry: &str, uniform_size: usize) -> Result<naga::Module> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| anyhow::anyhow!(error.emit_to_string(source)))?;
    let mut uniform = false;
    let mut surface = false;
    for (_, global) in module.global_variables.iter() {
        if global.space == naga::AddressSpace::Private && global.binding.is_none() {
            continue;
        }
        let binding = global
            .binding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("effect global requires the declared ABI"))?;
        ensure!(binding.group == 0, "effect binding group changed");
        match binding.binding {
            0 => {
                ensure!(
                    !uniform
                        && global.space == naga::AddressSpace::Uniform
                        && matches!(module.types[global.ty].inner, naga::TypeInner::Struct { span, .. } if span as usize == uniform_size),
                    "effect uniform ABI changed"
                );
                uniform = true;
            },
            1 => {
                ensure!(
                    !surface
                        && global.space == naga::AddressSpace::Handle
                        && matches!(
                            module.types[global.ty].inner,
                            naga::TypeInner::Image {
                                dim: naga::ImageDimension::D2,
                                arrayed: false,
                                class: naga::ImageClass::Sampled {
                                    kind: naga::ScalarKind::Float,
                                    multi: false
                                }
                            }
                        ),
                    "effect surface ABI changed"
                );
                surface = true;
            },
            _ => anyhow::bail!("effect binding is not admitted"),
        }
    }
    ensure!(uniform && surface, "effect frame and surface bindings are required");
    ensure!((1..=8).contains(&module.entry_points.len()), "invalid effect entry count");
    for point in &module.entry_points {
        ensure!(
            point.stage == naga::ShaderStage::Fragment
                && point.function.arguments.len() <= 1
                && point.function.arguments.iter().all(|argument| matches!(
                    argument.binding,
                    Some(naga::Binding::BuiltIn(naga::BuiltIn::Position { .. }))
                )),
            "effect input is fragment position"
        );
        let result = point
            .function
            .result
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("effect output required"))?;
        ensure!(
            matches!(result.binding, Some(naga::Binding::Location { location: 0, .. }))
                && matches!(
                    module.types[result.ty].inner,
                    naga::TypeInner::Vector {
                        size: naga::VectorSize::Quad,
                        scalar: naga::Scalar { kind: naga::ScalarKind::Float, width: 4 }
                    }
                ),
            "effect output must be location0 vec4<f32>"
        );
    }
    ensure!(module.entry_points.iter().any(|point| point.name == entry), "effect entry is missing");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)?;
    Ok(module)
}

#[cfg(test)]
mod tests;
