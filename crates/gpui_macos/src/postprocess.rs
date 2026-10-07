//! Metal 局部后处理：准备归后台，命令归渲染线程，预算归实际完成回调。
mod compiler;

use anyhow::{Context as _, Result, ensure};
use block::ConcreteBlock;
use collections::FxHashMap;
use foreign_types::{ForeignType, ForeignTypeRef};
use gpui::{
    BackgroundShaderCancellation, Bounds, PaintPostprocess, PreparedStreamImage, ScaledPixels,
    StreamImageBudgets, StreamImageId, StreamImageLease, WgslPostprocessDescriptor, point, size,
};
use metal::{CommandBufferRef, MTLPixelFormat, TextureRef};
use objc::{msg_send, sel, sel_impl};
use parking_lot::Mutex;
use std::{
    cell::{Cell, RefCell},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::ThreadId,
};

struct DeviceState {
    device: metal::Device,
    compiler: Mutex<compiler::Compiler>,
    owning_thread: ThreadId,
    epoch: AtomicU64,
    lost: AtomicBool,
}

struct Resources {
    targets: [metal::Texture; 2],
    pipelines: Vec<metal::RenderPipelineState>,
    lease: StreamImageLease,
}

struct Prepared {
    device: Arc<DeviceState>,
    epoch: u64,
    cancellation: BackgroundShaderCancellation,
    descriptor: WgslPostprocessDescriptor,
    resources: Arc<Resources>,
}

pub(crate) struct PostprocessAtlas {
    device: Arc<DeviceState>,
    // 只有拥有窗口的线程访问已采用的资源；后台编译从不持有渲染路径上的锁。
    images: RefCell<FxHashMap<StreamImageId, Prepared>>,
}

impl PostprocessAtlas {
    pub fn new(device: metal::Device) -> Self {
        Self {
            device: Arc::new(DeviceState {
                compiler: Mutex::new(compiler::Compiler::new(device.clone())),
                device,
                owning_thread: std::thread::current().id(),
                epoch: AtomicU64::new(0),
                lost: AtomicBool::new(false),
            }),
            images: RefCell::new(FxHashMap::default()),
        }
    }

    pub fn supported(&self) -> bool {
        !self.device.lost.load(Ordering::Acquire)
    }

    pub fn invalidate(&self) -> Result<()> {
        self.device
            .epoch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| anyhow::anyhow!("effect device epoch exhausted"))?;
        Ok(())
    }

    pub fn destroy(&self) {
        self.device.lost.store(true, Ordering::Release);
        self.images.borrow_mut().clear();
    }

    pub fn factory(
        &self,
        id: StreamImageId,
        budgets: &StreamImageBudgets,
        descriptor: WgslPostprocessDescriptor,
        cancellation: BackgroundShaderCancellation,
    ) -> Result<Box<dyn FnOnce() -> Result<Option<PreparedStreamImage>> + Send>> {
        self.check_thread()?;
        descriptor.validate()?;
        ensure!(self.supported(), "effect device is unavailable");
        ensure!(
            descriptor.size.width.0 <= 16384 && descriptor.size.height.0 <= 16384,
            "effect extent exceeds Metal texture limits"
        );
        budgets.ensure_available()?;
        let device = self.device.clone();
        let epoch = device.epoch.load(Ordering::Acquire);
        let budgets = budgets.clone();
        Ok(Box::new(move || {
            objc::rc::autoreleasepool(|| {
                ensure!(
                    std::thread::current().id() != device.owning_thread,
                    "effect preparation must run off UI"
                );
                if cancellation.is_cancelled() {
                    return Ok(None);
                }
                ensure!(
                    !device.lost.load(Ordering::Acquire)
                        && epoch == device.epoch.load(Ordering::Acquire),
                    "effect preparation belongs to an obsolete device"
                );
                budgets.ensure_available()?;
                // 参数沿用已有逐帧实例池，效果自身只分配两张完整分辨率纹理。
                let lease = budgets.reserve(descriptor.texture_bytes()?)?;
                let mut pipelines = Vec::with_capacity(descriptor.passes.len());
                for pass in descriptor.passes.iter() {
                    pipelines.push(
                        device
                            .compiler
                            .lock()
                            .program(pass, descriptor.uniform_size)?,
                    );
                    if cancellation.is_cancelled() {
                        return Ok(None);
                    }
                }
                let targets = [
                    target(&device.device, &descriptor)?,
                    target(&device.device, &descriptor)?,
                ];
                if cancellation.is_cancelled() {
                    return Ok(None);
                }
                budgets.ensure_available()?;
                ensure!(
                    !device.lost.load(Ordering::Acquire)
                        && epoch == device.epoch.load(Ordering::Acquire),
                    "effect device changed during preparation"
                );
                Ok(Some(PreparedStreamImage::new(
                    id,
                    Prepared {
                        device,
                        epoch,
                        cancellation,
                        descriptor,
                        resources: Arc::new(Resources {
                            targets,
                            pipelines,
                            lease,
                        }),
                    },
                )))
            })
        }))
    }

    pub fn adopt(&self, id: StreamImageId, prepared: PreparedStreamImage) -> Result<()> {
        self.check_thread()?;
        let prepared = prepared.into_native::<Prepared>(id)?;
        ensure!(
            Arc::ptr_eq(&self.device, &prepared.device),
            "effect belongs to another atlas"
        );
        ensure!(
            self.supported() && prepared.epoch == self.device.epoch.load(Ordering::Acquire),
            "effect device changed before adoption"
        );
        ensure!(
            !prepared.cancellation.is_cancelled(),
            "effect adoption cancelled"
        );
        prepared.resources.lease.budgets().ensure_available()?;
        let mut images = self.images.borrow_mut();
        ensure!(
            !images.contains_key(&id),
            "effect owner already has an adopted program"
        );
        images.insert(id, prepared);
        Ok(())
    }

    pub fn retire(&self, id: StreamImageId) -> Result<()> {
        self.check_thread()?;
        // 已提交命令的完成回调仍拥有资源及租约，因此逻辑移除不提前归还 GPU 预算。
        self.images.borrow_mut().remove(&id);
        Ok(())
    }

    pub fn render(
        &self,
        command: &CommandBufferRef,
        screen: &TextureRef,
        effect: &PaintPostprocess,
        uniforms: &metal::BufferRef,
        offset: usize,
    ) -> Result<()> {
        self.check_thread()?;
        ensure!(self.supported(), "effect device is unavailable");
        let images = self.images.borrow();
        let prepared = images
            .get(&effect.owner.id())
            .context("effect owner is not prepared")?;
        let resources = &prepared.resources;
        resources.lease.budgets().ensure_available()?;
        ensure!(
            effect.uniforms.len() == prepared.descriptor.uniform_size,
            "effect uniform layout changed"
        );
        ensure!(
            offset % 256 == 0
                && offset
                    .checked_add(effect.uniforms.len())
                    .is_some_and(|end| end as u64 <= uniforms.length()),
            "effect uniform buffer range changed"
        );
        ensure!(
            screen.device().as_ptr() == self.device.device.as_ptr()
                && uniforms.device().as_ptr() == self.device.device.as_ptr(),
            "effect device mismatch"
        );
        ensure!(
            screen.pixel_format() == MTLPixelFormat::BGRA8Unorm
                && screen.sample_count() == 1
                && !screen.framebuffer_only(),
            "effect requires a copyable BGRA surface"
        );
        ensure!(
            effect.bounds.size.width.0 == prepared.descriptor.size.width.0 as f32
                && effect.bounds.size.height.0 == prepared.descriptor.size.height.0 as f32,
            "effect extent changed before replacement"
        );
        let Some(region) = region(screen, effect.bounds, effect.content_mask.bounds)? else {
            return Ok(());
        };
        let input = &resources.targets[0];
        clear(command, input)?;
        let blit = command.new_blit_command_encoder();
        blit.copy_from_texture(
            screen,
            0,
            0,
            region.screen,
            region.size,
            input,
            0,
            0,
            region.local,
        );
        blit.end_encoding();
        for (index, pipeline) in resources.pipelines.iter().enumerate() {
            let target = &resources.targets[(index + 1) % 2];
            let descriptor = render_pass(target)?;
            let encoder = command.new_render_command_encoder(&descriptor);
            encoder.set_render_pipeline_state(pipeline);
            encoder.set_fragment_buffer(0, Some(uniforms), offset as u64);
            encoder.set_fragment_texture(0, Some(&resources.targets[index % 2]));
            encoder.set_viewport(metal::MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: target.width() as f64,
                height: target.height() as f64,
                znear: 0.0,
                zfar: 1.0,
            });
            encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 3);
            encoder.end_encoding();
        }
        let output = &resources.targets[resources.pipelines.len() % 2];
        let blit = command.new_blit_command_encoder();
        blit.copy_from_texture(
            output,
            0,
            0,
            region.local,
            region.size,
            screen,
            0,
            0,
            region.screen,
        );
        blit.end_encoding();

        let retained = Cell::new(Some(resources.clone()));
        let device = self.device.clone();
        let feedback = effect.feedback.clone();
        let block = ConcreteBlock::new(move |command: &CommandBufferRef| {
            if let Some(resources) = retained.take() {
                if command.status() != metal::MTLCommandBufferStatus::Completed {
                    resources.lease.budgets().revoke();
                    device.lost.store(true, Ordering::Release);
                    feedback.record_error(format!(
                        "Metal effect submission failed: {:?}",
                        command.status()
                    ));
                }
                // 回调运行时 GPU 已结束访问；失败也先撤销准入，再释放实际完成的资源。
                drop(resources);
            }
        })
        .copy();
        command.add_completed_handler(&block);
        Ok(())
    }

    fn check_thread(&self) -> Result<()> {
        ensure!(
            std::thread::current().id() == self.device.owning_thread,
            "effect operation must run on the owning UI thread"
        );
        Ok(())
    }
}

fn target(
    device: &metal::DeviceRef,
    descriptor: &WgslPostprocessDescriptor,
) -> Result<metal::Texture> {
    let texture = metal::TextureDescriptor::new();
    texture.set_width(descriptor.size.width.0 as u64);
    texture.set_height(descriptor.size.height.0 as u64);
    texture.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    texture.set_storage_mode(metal::MTLStorageMode::Private);
    texture.set_usage(metal::MTLTextureUsage::ShaderRead | metal::MTLTextureUsage::RenderTarget);
    // metal-rs 的便捷接口假定分配成功；这里检查原生 nil，避免预算内的驱动失败变成空指针。
    let native: *mut metal::MTLTexture =
        unsafe { msg_send![device, newTextureWithDescriptor: texture.as_ref()] };
    ensure!(!native.is_null(), "Metal effect texture allocation failed");
    Ok(unsafe { metal::Texture::from_ptr(native) })
}

fn render_pass(texture: &TextureRef) -> Result<metal::RenderPassDescriptor> {
    let descriptor = metal::RenderPassDescriptor::new().to_owned();
    let color = descriptor
        .color_attachments()
        .object_at(0)
        .context("effect color attachment missing")?;
    color.set_texture(Some(texture));
    color.set_load_action(metal::MTLLoadAction::Clear);
    color.set_store_action(metal::MTLStoreAction::Store);
    color.set_clear_color(metal::MTLClearColor::new(0.0, 0.0, 0.0, 0.0));
    Ok(descriptor)
}

fn clear(command: &CommandBufferRef, texture: &TextureRef) -> Result<()> {
    // 输入裁剪以外必须清透明；只裁剪最终写回仍会暴露相邻窗格的像素。
    command
        .new_render_command_encoder(&render_pass(texture)?)
        .end_encoding();
    Ok(())
}

struct Region {
    local: metal::MTLOrigin,
    screen: metal::MTLOrigin,
    size: metal::MTLSize,
}

fn region(
    screen: &TextureRef,
    bounds: Bounds<ScaledPixels>,
    mask: Bounds<ScaledPixels>,
) -> Result<Option<Region>> {
    let x = bounds.origin.x.0;
    let y = bounds.origin.y.0;
    ensure!(
        x.is_finite() && y.is_finite() && x.fract() == 0.0 && y.fract() == 0.0,
        "effect origin must use physical pixels"
    );
    ensure!(
        [
            mask.origin.x.0,
            mask.origin.y.0,
            mask.size.width.0,
            mask.size.height.0
        ]
        .iter()
        .all(|value| value.is_finite()),
        "effect content mask is invalid"
    );
    let window = Bounds::new(
        point(ScaledPixels(0.0), ScaledPixels(0.0)),
        size(
            ScaledPixels(screen.width() as f32),
            ScaledPixels(screen.height() as f32),
        ),
    );
    let visible = bounds.intersect(&window).intersect(&mask);
    if visible.is_empty() {
        return Ok(None);
    }
    let left = visible.left().0.ceil().max(0.0) as u64;
    let top = visible.top().0.ceil().max(0.0) as u64;
    let right = visible.right().0.floor().min(screen.width() as f32) as u64;
    let bottom = visible.bottom().0.floor().min(screen.height() as f32) as u64;
    if left >= right || top >= bottom {
        return Ok(None);
    }
    let local = metal::MTLOrigin {
        x: (left as f32 - x) as u64,
        y: (top as f32 - y) as u64,
        z: 0,
    };
    ensure!(
        local.x + right - left <= bounds.size.width.0 as u64
            && local.y + bottom - top <= bounds.size.height.0 as u64,
        "effect copy exceeds its extent"
    );
    Ok(Some(Region {
        local,
        screen: metal::MTLOrigin {
            x: left,
            y: top,
            z: 0,
        },
        size: metal::MTLSize {
            width: right - left,
            height: bottom - top,
            depth: 1,
        },
    }))
}

#[cfg(test)]
mod tests;
