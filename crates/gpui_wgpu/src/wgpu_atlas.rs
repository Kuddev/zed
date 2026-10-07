use anyhow::{Context as _, Result};
use collections::FxHashMap;
use etagere::{BucketedAtlasAllocator, size2};
use gpui::{
    AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTextureList, AtlasTile, Bounds, DevicePixels,
    PlatformAtlas, Point, Size,
};
use parking_lot::Mutex;
use std::{borrow::Cow, ops, sync::Arc};

use crate::WgpuContext;

fn device_size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    size2(size.width.0, size.height.0)
}

fn etagere_point_to_device(point: etagere::Point) -> Point<DevicePixels> {
    Point { x: DevicePixels(point.x), y: DevicePixels(point.y) }
}

pub struct WgpuAtlas(Mutex<WgpuAtlasState>);
#[cfg(not(target_family = "wasm"))]
use crate::native_background_shader::{NativeShader, PreparedShader, ShaderDevice};
#[cfg(not(target_family = "wasm"))]
use crate::native_postprocess::{NativePostprocess, PostprocessDevice, PreparedPostprocess};
#[cfg(not(target_family = "wasm"))]
use crate::native_stream_image::{NativeStream, PreparedStream, StreamDevice};
#[cfg(not(target_family = "wasm"))]
const STREAM_TEXTURE_BIT: u32 = 0x8000_0000;

struct PendingUpload {
    id: AtlasTextureId,
    bounds: Bounds<DevicePixels>,
    data: Vec<u8>,
}

struct WgpuAtlasState {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    max_texture_size: u32,
    color_texture_format: wgpu::TextureFormat,
    storage: WgpuAtlasStorage,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
    pending_uploads: Vec<PendingUpload>,
    #[cfg(not(target_family = "wasm"))]
    streams: FxHashMap<u64, NativeShader>,
    #[cfg(not(target_family = "wasm"))]
    images: FxHashMap<u64, NativeStream>,
    #[cfg(not(target_family = "wasm"))]
    stream_device: Option<StreamDevice>,
    #[cfg(not(target_family = "wasm"))]
    shader_device: Option<ShaderDevice>,
    #[cfg(not(target_family = "wasm"))]
    shader_lost: Arc<std::sync::atomic::AtomicBool>,
    #[cfg(not(target_family = "wasm"))]
    owning_thread: std::thread::ThreadId,
    #[cfg(not(target_family = "wasm"))]
    background_preparation_supported: bool,
    #[cfg(not(target_family = "wasm"))]
    postprocess_format: Option<wgpu::TextureFormat>,
    #[cfg(not(target_family = "wasm"))]
    postprocess_device: Option<PostprocessDevice>,
    #[cfg(not(target_family = "wasm"))]
    postprocesses: FxHashMap<u64, NativePostprocess>,
}

pub struct WgpuTextureInfo {
    pub view: wgpu::TextureView,
}

impl WgpuAtlas {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        color_texture_format: wgpu::TextureFormat,
    ) -> Self {
        let max_texture_size = device.limits().max_texture_dimension_2d;
        WgpuAtlas(Mutex::new(WgpuAtlasState {
            device,
            queue,
            max_texture_size,
            color_texture_format,
            storage: WgpuAtlasStorage::default(),
            tiles_by_key: Default::default(),
            pending_uploads: Vec::new(),
            #[cfg(not(target_family = "wasm"))]
            streams: Default::default(),
            #[cfg(not(target_family = "wasm"))]
            images: Default::default(),
            #[cfg(not(target_family = "wasm"))]
            stream_device: None,
            #[cfg(not(target_family = "wasm"))]
            shader_device: None,
            #[cfg(not(target_family = "wasm"))]
            shader_lost: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(not(target_family = "wasm"))]
            owning_thread: std::thread::current().id(),
            #[cfg(not(target_family = "wasm"))]
            background_preparation_supported: false,
            #[cfg(not(target_family = "wasm"))]
            postprocess_format: None,
            #[cfg(not(target_family = "wasm"))]
            postprocess_device: None,
            #[cfg(not(target_family = "wasm"))]
            postprocesses: Default::default(),
        }))
    }

    pub fn from_context(context: &WgpuContext) -> Self {
        let atlas = Self::new(
            context.device.clone(),
            context.queue.clone(),
            context.color_texture_format(),
        );
        #[cfg(not(target_family = "wasm"))]
        {
            let mut state = atlas.0.lock();
            state.shader_lost = context.device_lost_flag();
            // WSL Mesa/D3D12 GL cancellation segfaulted during actual lifetime
            // qualification. The ordinary renderer stays available; background
            // device factories need independent acceptance before enabling GL.
            state.background_preparation_supported =
                matches!(context.backend(), crate::WgpuBackend::Native(wgpu::Backend::Vulkan));
        }
        atlas
    }

    pub fn before_frame(&self) {
        let mut lock = self.0.lock();
        lock.flush_uploads();
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn configure_postprocess(&self, format: Option<wgpu::TextureFormat>) {
        self.0.lock().postprocess_format = format;
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn render_postprocess(
        &self,
        screen: &wgpu::Texture,
        effect: &gpui::PaintPostprocess,
    ) -> Result<()> {
        let mut state = self.0.lock();
        anyhow::ensure!(
            state.postprocess_format == Some(screen.format()),
            "surface post-processing is not available for this surface"
        );
        let image = state
            .postprocesses
            .get_mut(&effect.owner.id().value())
            .ok_or_else(|| anyhow::anyhow!("effect owner is not prepared"))?;
        image.render(screen, effect.bounds, effect.content_mask.bounds, &effect.uniforms)
    }

    pub(crate) fn close_background_device(&self) {
        #[cfg(not(target_family = "wasm"))]
        {
            let mut state = self.0.lock();
            state.background_preparation_supported = false;
            state.postprocess_format = None;
            if let Some(device) = state.postprocess_device.take() {
                if let Err(error) = device.invalidate() {
                    log::error!("postprocess device close: {error:#}");
                }
            }
            if let Some(device) = state.stream_device.take() {
                if let Err(error) = device.invalidate() {
                    log::error!("stream device close: {error:#}");
                }
            }
            if let Some(device) = state.shader_device.take() {
                if let Err(error) = device.invalidate() {
                    log::error!("background device close: {error:#}");
                }
            }
            // Drop schedules acknowledged retirement; it never waits on this thread.
            state.streams.clear();
            state.images.clear();
            state.postprocesses.clear();
        }
    }

    pub fn get_texture_info(&self, id: AtlasTextureId) -> Result<WgpuTextureInfo> {
        let lock = self.0.lock();
        #[cfg(not(target_family = "wasm"))]
        if id.kind == AtlasTextureKind::Polychrome && id.index & STREAM_TEXTURE_BIT != 0 {
            if let Some(image) = lock.images.get(&u64::from(id.index & !STREAM_TEXTURE_BIT)) {
                return Ok(WgpuTextureInfo { view: image.view() });
            }
            let image = lock
                .streams
                .get(&u64::from(id.index & !STREAM_TEXTURE_BIT))
                .ok_or_else(|| anyhow::anyhow!("retired stream texture"))?;
            return Ok(WgpuTextureInfo { view: image.view() });
        }
        let texture = &lock.storage[id];
        Ok(WgpuTextureInfo { view: texture.view.clone() })
    }

    pub(crate) fn note_stream_submission(&self, scene: &gpui::Scene, index: wgpu::SubmissionIndex) {
        #[cfg(not(target_family = "wasm"))]
        {
            let mut state = self.0.lock();
            for sprite in &scene.polychrome_sprites {
                if sprite.tile.texture_id.index & STREAM_TEXTURE_BIT == 0 {
                    continue;
                }
                let id = u64::from(sprite.tile.texture_id.index & !STREAM_TEXTURE_BIT);
                if let Some(image) = state.images.get_mut(&id) {
                    image.note_scene_submission(index.clone());
                }
                if let Some(image) = state.streams.get_mut(&id) {
                    image.note_scene_submission(index.clone());
                }
            }
        }
        #[cfg(target_family = "wasm")]
        {
            let _unused = (scene, index);
        }
    }

    /// Clears all cached textures and tiles, forcing them to be recreated.
    /// Use this for incremental recovery when the device is still valid.
    pub fn clear(&self) {
        let mut lock = self.0.lock();
        lock.storage = WgpuAtlasStorage::default();
        lock.tiles_by_key.clear();
        lock.pending_uploads.clear();
    }

    /// Handles device lost by clearing all textures and cached tiles.
    /// The atlas will lazily recreate textures as needed on subsequent frames.
    pub fn handle_device_lost(&self, context: &WgpuContext) {
        let mut lock = self.0.lock();
        #[cfg(not(target_family = "wasm"))]
        {
            lock.postprocess_format = None;
            if let Some(device) = lock.postprocess_device.take() {
                if let Err(error) = device.invalidate() {
                    log::error!("postprocess device invalidation: {error:#}");
                }
            }
            lock.postprocesses.clear();
            if let Some(device) = lock.shader_device.take() {
                if let Err(error) = device.invalidate() {
                    log::error!("background device invalidation: {error:#}");
                }
            }
            if let Some(device) = lock.stream_device.take() {
                if let Err(error) = device.invalidate() {
                    log::error!("stream device invalidation: {error:#}");
                }
            }
            lock.images.clear();
            lock.streams.clear();
            lock.shader_lost = context.device_lost_flag();
            lock.background_preparation_supported =
                matches!(context.backend(), crate::WgpuBackend::Native(wgpu::Backend::Vulkan));
        }
        lock.device = context.device.clone();
        lock.queue = context.queue.clone();
        lock.color_texture_format = context.color_texture_format();
        lock.max_texture_size = context.device.limits().max_texture_dimension_2d;
        lock.storage = WgpuAtlasStorage::default();
        lock.tiles_by_key.clear();
        lock.pending_uploads.clear();
    }
}

impl PlatformAtlas for WgpuAtlas {
    #[cfg(not(target_family = "wasm"))]
    fn supports_postprocess_wgsl(&self) -> bool {
        let state = self.0.lock();
        state.background_preparation_supported
            && state.postprocess_format.is_some()
            && !state.shader_lost.load(std::sync::atomic::Ordering::Acquire)
    }

    #[cfg(not(target_family = "wasm"))]
    fn postprocess_wgsl_factory(
        &self,
        id: gpui::StreamImageId,
        budgets: &gpui::StreamImageBudgets,
        descriptor: gpui::WgslPostprocessDescriptor,
        cancellation: gpui::BackgroundShaderCancellation,
    ) -> Result<Box<dyn FnOnce() -> Result<Option<gpui::PreparedStreamImage>> + Send>> {
        let mut state = self.0.lock();
        anyhow::ensure!(
            state.background_preparation_supported,
            "postprocess preparation is not qualified for this backend"
        );
        let format = state
            .postprocess_format
            .ok_or_else(|| anyhow::anyhow!("surface does not support scoped copies"))?;
        anyhow::ensure!(
            std::thread::current().id() == state.owning_thread,
            "postprocess factory capture must run on UI"
        );
        anyhow::ensure!(
            !state.postprocesses.contains_key(&id.value())
                && !state.streams.contains_key(&id.value())
                && !state.images.contains_key(&id.value()),
            "postprocess owner already published"
        );
        if state.postprocess_device.is_none() {
            state.postprocess_device = Some(PostprocessDevice::new(
                state.device.clone(),
                state.queue.clone(),
                state.shader_lost.clone(),
            )?);
        }
        let device = state
            .postprocess_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("postprocess device missing"))?;
        let work = device.factory(id.value(), descriptor, format, budgets.clone(), cancellation)?;
        Ok(Box::new(move || {
            work.run().map(|image| image.map(|image| gpui::PreparedStreamImage::new(id, image)))
        }))
    }

    #[cfg(not(target_family = "wasm"))]
    fn adopt_postprocess(
        &self,
        id: gpui::StreamImageId,
        prepared: gpui::PreparedStreamImage,
    ) -> Result<()> {
        let prepared = prepared.into_native::<PreparedPostprocess>(id)?;
        let mut state = self.0.lock();
        anyhow::ensure!(
            !state.postprocesses.contains_key(&id.value())
                && !state.streams.contains_key(&id.value())
                && !state.images.contains_key(&id.value()),
            "postprocess owner already published"
        );
        let device = state
            .postprocess_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("obsolete postprocess device"))?;
        let image = device.adopt(id.value(), prepared)?;
        state.postprocesses.insert(id.value(), image);
        Ok(())
    }
    #[cfg(not(target_family = "wasm"))]
    fn stream_image_factory(
        &self,
        id: gpui::StreamImageId,
        budgets: &gpui::StreamImageBudgets,
        size: Size<DevicePixels>,
        cancellation: gpui::BackgroundShaderCancellation,
    ) -> Result<Box<dyn FnOnce() -> Result<Option<gpui::PreparedStreamImage>> + Send>> {
        let mut state = self.0.lock();
        anyhow::ensure!(
            state.background_preparation_supported,
            "stream resource preparation is not qualified for this backend"
        );
        anyhow::ensure!(
            std::thread::current().id() == state.owning_thread,
            "stream factory capture must run on UI"
        );
        anyhow::ensure!(
            !state.images.contains_key(&id.value())
                && !state.streams.contains_key(&id.value())
                && !state.postprocesses.contains_key(&id.value()),
            "stream owner already published"
        );
        if state.stream_device.is_none() {
            state.stream_device = Some(StreamDevice::new(
                state.device.clone(),
                state.queue.clone(),
                state.shader_lost.clone(),
            ));
        }
        let device = state
            .stream_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("stream device missing"))?
            .clone();
        let work = device.factory(
            id.value(),
            [size.width.0 as u32, size.height.0 as u32],
            budgets.clone(),
            cancellation,
        )?;
        Ok(Box::new(move || {
            work.run()
                .map(|prepared| prepared.map(|image| gpui::PreparedStreamImage::new(id, image)))
        }))
    }

    #[cfg(not(target_family = "wasm"))]
    fn adopt_stream_image(
        &self,
        id: gpui::StreamImageId,
        prepared: gpui::PreparedStreamImage,
    ) -> Result<()> {
        let prepared = prepared.into_native::<PreparedStream>(id)?;
        let mut state = self.0.lock();
        anyhow::ensure!(
            !state.images.contains_key(&id.value())
                && !state.streams.contains_key(&id.value())
                && !state.postprocesses.contains_key(&id.value()),
            "stream owner already published"
        );
        let device = state
            .stream_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("obsolete stream device"))?;
        let image = device.adopt(id.value(), prepared)?;
        state.images.insert(id.value(), image);
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    fn stage_stream_image(
        &self,
        id: gpui::StreamImageId,
        budgets: &gpui::StreamImageBudgets,
        frame: &gpui::StreamImageFrame<'_>,
    ) -> Result<gpui::StreamImageUpdate> {
        frame.validate()?;
        budgets.ensure_available()?;
        let mut state = self.0.lock();
        let image = state
            .images
            .get_mut(&id.value())
            .ok_or_else(|| anyhow::anyhow!("stream must be prepared before paint"))?;
        let completion = image.stage(frame)?;
        let tile = image.sequence().map(|_| AtlasTile {
            texture_id: AtlasTextureId {
                index: STREAM_TEXTURE_BIT | id.value() as u32,
                kind: AtlasTextureKind::Polychrome,
            },
            tile_id: gpui::TileId(id.value() as u32),
            padding: 0,
            bounds: Bounds::new(gpui::point(DevicePixels(0), DevicePixels(0)), frame.size),
        });
        Ok(gpui::StreamImageUpdate {
            tile,
            sequence: image.sequence().unwrap_or(0),
            completion: completion
                .map(|receipt| gpui::StreamImageCompletion(Box::new(move || receipt.wait()))),
        })
    }

    #[cfg(not(target_family = "wasm"))]
    fn background_wgsl_factory(
        &self,
        id: gpui::StreamImageId,
        budgets: &gpui::StreamImageBudgets,
        size: Size<DevicePixels>,
        source: Arc<str>,
        entry: Arc<str>,
        cancellation: gpui::BackgroundShaderCancellation,
    ) -> Result<Box<dyn FnOnce() -> Result<Option<gpui::PreparedBackgroundShader>> + Send>> {
        let mut state = self.0.lock();
        anyhow::ensure!(
            state.background_preparation_supported,
            "background shader preparation is not qualified for this backend"
        );
        anyhow::ensure!(
            std::thread::current().id() == state.owning_thread,
            "factory capture must run on its UI thread"
        );
        anyhow::ensure!(
            !state.streams.contains_key(&id.value())
                && !state.images.contains_key(&id.value())
                && !state.postprocesses.contains_key(&id.value()),
            "owner already published"
        );
        if state.shader_device.is_none() {
            state.shader_device = Some(ShaderDevice::new(
                state.device.clone(),
                state.queue.clone(),
                state.shader_lost.clone(),
            )?);
        }
        let device = state
            .shader_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing shader device"))?
            .clone();
        let work = device.factory(
            id.value(),
            [size.width.0 as u32, size.height.0 as u32],
            source,
            entry,
            state.color_texture_format,
            budgets.clone(),
            cancellation,
        )?;
        Ok(Box::new(move || {
            work.run().map(|prepared| {
                prepared.map(|prepared| gpui::PreparedBackgroundShader::new(id, prepared))
            })
        }))
    }
    #[cfg(not(target_family = "wasm"))]
    fn adopt_background_shader(
        &self,
        id: gpui::StreamImageId,
        prepared: gpui::PreparedBackgroundShader,
    ) -> Result<()> {
        let prepared = prepared.into_native::<PreparedShader>(id)?;
        let mut state = self.0.lock();
        anyhow::ensure!(
            !state.streams.contains_key(&id.value())
                && !state.images.contains_key(&id.value())
                && !state.postprocesses.contains_key(&id.value()),
            "background owner already published"
        );
        let device = state
            .shader_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("obsolete background device"))?;
        let image = device.adopt(id.value(), prepared)?;
        state.streams.insert(id.value(), image);
        Ok(())
    }
    #[cfg(not(target_family = "wasm"))]
    fn stage_background_wgsl(
        &self,
        id: gpui::StreamImageId,
        budgets: &gpui::StreamImageBudgets,
        frame: &gpui::BackgroundWgslFrame<'_>,
    ) -> Result<gpui::StreamImageUpdate> {
        frame.validate()?;
        budgets.ensure_available()?;
        let mut state = self.0.lock();
        let image = state
            .streams
            .get_mut(&id.value())
            .ok_or_else(|| anyhow::anyhow!("background must be prepared before paint"))?;
        let completion = image.stage(
            frame.sequence,
            [frame.size.width.0 as u32, frame.size.height.0 as u32],
            frame.source,
            frame.entry,
        )?;
        let tile = image.sequence().map(|_| AtlasTile {
            texture_id: AtlasTextureId {
                index: STREAM_TEXTURE_BIT | id.value() as u32,
                kind: AtlasTextureKind::Polychrome,
            },
            tile_id: gpui::TileId(id.value() as u32),
            padding: 0,
            bounds: Bounds::new(gpui::point(DevicePixels(0), DevicePixels(0)), frame.size),
        });
        Ok(gpui::StreamImageUpdate {
            tile,
            sequence: image.sequence().unwrap_or(0),
            completion: completion
                .map(|completion| gpui::StreamImageCompletion(Box::new(move || completion.wait()))),
        })
    }
    #[cfg(not(target_family = "wasm"))]
    fn retire_stream_image(
        &self,
        id: gpui::StreamImageId,
    ) -> Result<Option<gpui::StreamImageCompletion>> {
        let mut state = self.0.lock();
        if let Some(image) = state.postprocesses.remove(&id.value()) {
            return Ok(image
                .retire()
                .map(|receipt| gpui::StreamImageCompletion(Box::new(move || receipt.wait()))));
        }
        if let Some(image) = state.images.remove(&id.value()) {
            return Ok(image
                .retire()
                .map(|receipt| gpui::StreamImageCompletion(Box::new(move || receipt.wait()))));
        }
        Ok(state
            .streams
            .remove(&id.value())
            .and_then(NativeShader::retire)
            .map(|completion| gpui::StreamImageCompletion(Box::new(move || completion.wait()))))
    }
    #[cfg(not(target_family = "wasm"))]
    fn invalidate_background_preparations_for_test(&self) -> Result<()> {
        let state = self.0.lock();
        anyhow::ensure!(state.streams.is_empty(), "injection requires no published streams");
        state
            .shader_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing background device"))?
            .invalidate()
    }

    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        let mut lock = self.0.lock();
        if let Some(tile) = lock.tiles_by_key.get(key) {
            Ok(Some(*tile))
        } else {
            profiling::scope!("new tile");
            let Some((size, bytes)) = build()? else {
                return Ok(None);
            };
            let tile = lock.allocate(size, key.texture_kind()).context("failed to allocate")?;
            lock.upload_texture(tile.texture_id, tile.bounds, &bytes);
            lock.tiles_by_key.insert(key.clone(), tile);
            Ok(Some(tile))
        }
    }

    fn remove(&self, key: &AtlasKey) {
        let mut lock = self.0.lock();

        let Some(tile) = lock.tiles_by_key.remove(key) else {
            return;
        };
        let id = tile.texture_id;

        let Some(texture_slot) = lock.storage[id.kind].textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                lock.pending_uploads.retain(|upload| upload.id != texture.id);
                lock.storage[id.kind].free_list.push(texture.id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

impl WgpuAtlasState {
    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = &mut self.storage[texture_kind];

            if let Some(tile) = textures.iter_mut().rev().find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut WgpuAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> =
            Size { width: DevicePixels(1024), height: DevicePixels(1024) };
        let max_texture_size = self.max_texture_size as i32;
        let max_atlas_size =
            Size { width: DevicePixels(max_texture_size), height: DevicePixels(max_texture_size) };

        let size = min_size.min(&max_atlas_size).max(&DEFAULT_ATLAS_SIZE);
        let format = match kind {
            AtlasTextureKind::Monochrome => wgpu::TextureFormat::R8Unorm,
            AtlasTextureKind::Subpixel | AtlasTextureKind::Polychrome => self.color_texture_format,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d {
                width: size.width.0 as u32,
                height: size.height.0 as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_list = &mut self.storage[kind];
        let index = texture_list.free_list.pop();

        let atlas_texture = WgpuAtlasTexture {
            id: AtlasTextureId { index: index.unwrap_or(texture_list.textures.len()) as u32, kind },
            allocator: BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            format,
            texture,
            view,
            live_atlas_keys: 0,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix).and_then(|t| t.as_mut()).expect("texture must exist")
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut().and_then(|t| t.as_mut()).expect("texture must exist")
        }
    }

    fn upload_texture(&mut self, id: AtlasTextureId, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let data = self
            .storage
            .get(id)
            .map(|texture| swizzle_upload_data(bytes, texture.format))
            .unwrap_or_else(|| bytes.to_vec());

        self.pending_uploads.push(PendingUpload { id, bounds, data });
    }

    fn flush_uploads(&mut self) {
        for upload in self.pending_uploads.drain(..) {
            let Some(texture) = self.storage.get(upload.id) else {
                continue;
            };
            let bytes_per_pixel = texture.bytes_per_pixel();

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: upload.bounds.origin.x.0 as u32,
                        y: upload.bounds.origin.y.0 as u32,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &upload.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.bounds.size.width.0 as u32 * bytes_per_pixel as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: upload.bounds.size.width.0 as u32,
                    height: upload.bounds.size.height.0 as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

#[derive(Default)]
struct WgpuAtlasStorage {
    monochrome_textures: AtlasTextureList<WgpuAtlasTexture>,
    subpixel_textures: AtlasTextureList<WgpuAtlasTexture>,
    polychrome_textures: AtlasTextureList<WgpuAtlasTexture>,
}

impl ops::Index<AtlasTextureKind> for WgpuAtlasStorage {
    type Output = AtlasTextureList<WgpuAtlasTexture>;
    fn index(&self, kind: AtlasTextureKind) -> &Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        }
    }
}

impl ops::IndexMut<AtlasTextureKind> for WgpuAtlasStorage {
    fn index_mut(&mut self, kind: AtlasTextureKind) -> &mut Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        }
    }
}

impl WgpuAtlasStorage {
    fn get(&self, id: AtlasTextureId) -> Option<&WgpuAtlasTexture> {
        self[id.kind].textures.get(id.index as usize).and_then(|t| t.as_ref())
    }
}

impl ops::Index<AtlasTextureId> for WgpuAtlasStorage {
    type Output = WgpuAtlasTexture;
    fn index(&self, id: AtlasTextureId) -> &Self::Output {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        };
        textures[id.index as usize].as_ref().expect("texture must exist")
    }
}

struct WgpuAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    live_atlas_keys: u32,
}

impl WgpuAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(device_size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            padding: 0,
            bounds: Bounds { origin: etagere_point_to_device(allocation.rectangle.min), size },
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn bytes_per_pixel(&self) -> u8 {
        match self.format {
            wgpu::TextureFormat::R8Unorm => 1,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm => 4,
            _ => 4,
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn swizzle_upload_data(bytes: &[u8], format: wgpu::TextureFormat) -> Vec<u8> {
    match format {
        wgpu::TextureFormat::Rgba8Unorm => {
            let mut data = bytes.to_vec();
            for pixel in data.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            data
        },
        _ => bytes.to_vec(),
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use gpui::block_on;
    use gpui::{ImageId, RenderImageParams};
    use std::sync::Arc;

    fn test_device_and_queue() -> anyhow::Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|error| anyhow::anyhow!("failed to request adapter: {error}"))?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("wgpu_atlas_test_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults()
                        .using_resolution(adapter.limits())
                        .using_alignment(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                })
                .await
                .map_err(|error| anyhow::anyhow!("failed to request device: {error}"))?;
            Ok((Arc::new(device), Arc::new(queue)))
        })
    }

    #[test]
    fn before_frame_skips_uploads_for_removed_texture() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;

        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let key = AtlasKey::Image(RenderImageParams { image_id: ImageId(1), frame_index: 0 });
        let size = Size { width: DevicePixels(1), height: DevicePixels(1) };
        let mut build = || Ok(Some((size, Cow::Owned(vec![0, 0, 0, 255]))));

        // Regression test: before the fix, this panicked in flush_uploads
        atlas.get_or_insert_with(&key, &mut build)?.expect("tile should be created");
        atlas.remove(&key);
        atlas.before_frame();
        Ok(())
    }

    #[test]
    fn remove_deallocates_tile_space_for_reuse() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);

        let small = Size { width: DevicePixels(64), height: DevicePixels(64) };
        let big = Size { width: DevicePixels(700), height: DevicePixels(700) };

        let make_key = |image_id: usize| {
            AtlasKey::Image(RenderImageParams { image_id: ImageId(image_id), frame_index: 0 })
        };
        let insert = |key: &AtlasKey, size: Size<DevicePixels>| {
            let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
            atlas
                .get_or_insert_with(key, &mut || {
                    Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
                })
                .expect("allocation should succeed")
                .expect("callback returns Some")
        };

        let keeper_key = make_key(1);
        let big_key_a = make_key(2);
        let big_key_b = make_key(3);

        let keeper_tile = insert(&keeper_key, small);
        let tile_a = insert(&big_key_a, big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);
        let tile_b = insert(&big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
        Ok(())
    }

    #[test]
    fn swizzle_upload_data_preserves_bgra_uploads() {
        let input = vec![0x10, 0x20, 0x30, 0x40];
        assert_eq!(swizzle_upload_data(&input, wgpu::TextureFormat::Bgra8Unorm), input);
    }

    #[test]
    fn swizzle_upload_data_converts_bgra_to_rgba() {
        let input = vec![0x10, 0x20, 0x30, 0x40, 0xAA, 0xBB, 0xCC, 0xDD];
        assert_eq!(
            swizzle_upload_data(&input, wgpu::TextureFormat::Rgba8Unorm),
            vec![0x30, 0x20, 0x10, 0x40, 0xCC, 0xBB, 0xAA, 0xDD]
        );
    }
}
