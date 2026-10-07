use collections::FxHashMap;
#[path = "postprocess.rs"]
mod postprocess;
#[path = "postprocess_wgsl.rs"]
mod postprocess_wgsl;
#[path = "stream_image.rs"]
mod stream_image;
use etagere::BucketedAtlasAllocator;
use parking_lot::Mutex;
use postprocess::NativePostprocess;
use std::sync::Arc;
use stream_image::{NativeStreamImage, STREAM_TEXTURE_BIT};
use windows::Win32::Graphics::{
    Direct3D11::{
        D3D11_BIND_SHADER_RESOURCE, D3D11_BOX, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
        ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
    },
    Dxgi::Common::*,
};
use windows::Win32::System::Threading::GetCurrentThreadId;

use gpui::{
    AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTextureList, AtlasTile, Bounds, DevicePixels,
    PlatformAtlas, Point, Size,
};

pub(crate) struct DirectXAtlas(Mutex<DirectXAtlasState>, u32);

struct DirectXAtlasState {
    device: ID3D11Device,
    device_context: ID3D11DeviceContext,
    monochrome_textures: AtlasTextureList<DirectXAtlasTexture>,
    polychrome_textures: AtlasTextureList<DirectXAtlasTexture>,
    subpixel_textures: AtlasTextureList<DirectXAtlasTexture>,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
    streams: FxHashMap<u64, NativeStreamImage>,
    effects: FxHashMap<u64, NativePostprocess>,
    device_epoch: u64,
}
struct PreparedNativeBackground {
    image: NativeStreamImage,
    device_epoch: u64,
}
enum PostprocessInput {
    Bytecode(gpui::PostprocessDescriptor),
    Wgsl(gpui::WgslPostprocessDescriptor),
}

struct PreparedNativePostprocess {
    effect: NativePostprocess,
    device_epoch: u64,
    cancellation: gpui::BackgroundShaderCancellation,
}

struct DirectXAtlasTexture {
    id: AtlasTextureId,
    bytes_per_pixel: u32,
    allocator: BucketedAtlasAllocator,
    texture: ID3D11Texture2D,
    view: [Option<ID3D11ShaderResourceView>; 1],
    live_atlas_keys: u32,
}

impl DirectXAtlas {
    pub(crate) fn new(device: &ID3D11Device, device_context: &ID3D11DeviceContext) -> Self {
        DirectXAtlas(
            Mutex::new(DirectXAtlasState {
                device: device.clone(),
                device_context: device_context.clone(),
                monochrome_textures: Default::default(),
                polychrome_textures: Default::default(),
                subpixel_textures: Default::default(),
                tiles_by_key: Default::default(),
                streams: Default::default(),
                effects: Default::default(),
                device_epoch: 0,
            }),
            unsafe { GetCurrentThreadId() },
        )
    }

    fn postprocess_preparation(
        &self,
        id: gpui::StreamImageId,
        budgets: &gpui::StreamImageBudgets,
        input: PostprocessInput,
        cancellation: gpui::BackgroundShaderCancellation,
    ) -> anyhow::Result<Box<dyn FnOnce() -> anyhow::Result<Option<gpui::PreparedStreamImage>> + Send>>
    {
        let state = self.0.lock();
        anyhow::ensure!(
            !state.streams.contains_key(&id.value()) && !state.effects.contains_key(&id.value()),
            "effect source owner is already occupied"
        );
        let device = state.device.clone();
        let context = state.device_context.clone();
        let epoch = state.device_epoch;
        let thread = self.1;
        let budgets = budgets.clone();
        Ok(Box::new(move || {
            anyhow::ensure!(
                unsafe { GetCurrentThreadId() } != thread,
                "effect factory ran on its UI thread"
            );
            if cancellation.is_cancelled() {
                return Ok(None);
            }
            let descriptor = match input {
                PostprocessInput::Bytecode(descriptor) => descriptor,
                PostprocessInput::Wgsl(descriptor) => {
                    let Some(descriptor) = postprocess_wgsl::compile(&descriptor, &cancellation)?
                    else {
                        return Ok(None);
                    };
                    descriptor
                },
            };
            let effect = NativePostprocess::new(&device, &context, id, &budgets, descriptor)?;
            if cancellation.is_cancelled() {
                return Ok(None);
            }
            budgets.ensure_available()?;
            Ok(Some(gpui::PreparedStreamImage::new(
                id,
                PreparedNativePostprocess { effect, device_epoch: epoch, cancellation },
            )))
        }))
    }

    pub(crate) fn get_texture_view(
        &self,
        id: AtlasTextureId,
    ) -> [Option<ID3D11ShaderResourceView>; 1] {
        let lock = self.0.lock();
        if id.kind == AtlasTextureKind::Polychrome && id.index & STREAM_TEXTURE_BIT != 0 {
            return lock
                .streams
                .get(&u64::from(id.index & !STREAM_TEXTURE_BIT))
                .map(NativeStreamImage::view)
                .unwrap_or([None]);
        }
        let tex = lock.texture(id);
        tex.view.clone()
    }

    pub(crate) fn handle_device_lost(
        &self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
    ) -> anyhow::Result<()> {
        let mut lock = self.0.lock();
        anyhow::ensure!(
            (lock.streams.is_empty() && lock.effects.is_empty())
                || unsafe { lock.device.GetDeviceRemovedReason() }.is_err(),
            "live stream resources require completion before a healthy device reset"
        );
        lock.streams.clear();
        lock.effects.clear();
        lock.device_epoch = lock
            .device_epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("background device epoch exhausted"))?;
        stream_image::recover_quarantined();
        lock.device = device.clone();
        lock.device_context = device_context.clone();
        lock.monochrome_textures = AtlasTextureList::default();
        lock.polychrome_textures = AtlasTextureList::default();
        lock.subpixel_textures = AtlasTextureList::default();
        lock.tiles_by_key.clear();
        Ok(())
    }

    pub(crate) fn render_postprocess(
        &self,
        screen: &ID3D11Texture2D,
        paint: &gpui::PaintPostprocess,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            unsafe { GetCurrentThreadId() } == self.1,
            "effect draw must run on its UI thread"
        );
        let mut state = self.0.lock();
        state
            .effects
            .get_mut(&paint.owner.id().value())
            .ok_or_else(|| anyhow::anyhow!("surface effect is not prepared"))?
            .render(screen, paint)
    }

    pub(crate) fn finish_stream_frame(&self, scene: &gpui::Scene) -> anyhow::Result<()> {
        let mut lock = self.0.lock();
        let ids: collections::FxHashSet<u64> = scene
            .polychrome_sprites
            .iter()
            .filter(|sprite| sprite.tile.texture_id.index & STREAM_TEXTURE_BIT != 0)
            .map(|sprite| u64::from(sprite.tile.texture_id.index & !STREAM_TEXTURE_BIT))
            .collect();
        for id in ids {
            if let Some(stream) = lock.streams.get_mut(&id) {
                stream.signal()?;
            }
        }
        // Completion must progress even when the window stops presenting next.
        if !lock.streams.is_empty() {
            unsafe { lock.device_context.Flush() };
        }
        Ok(())
    }
}

impl PlatformAtlas for DirectXAtlas {
    fn supports_postprocess_wgsl(&self) -> bool {
        true
    }

    fn invalidate_background_preparations_for_test(&self) -> anyhow::Result<()> {
        let mut lock = self.0.lock();
        anyhow::ensure!(
            lock.streams.is_empty() && lock.effects.is_empty(),
            "epoch fault injection requires no published streams"
        );
        lock.device_epoch = lock
            .device_epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("background epoch exhausted"))?;
        Ok(())
    }
    fn stage_stream_image(
        &self,
        id: gpui::StreamImageId,
        budgets: &gpui::StreamImageBudgets,
        frame: &gpui::StreamImageFrame<'_>,
    ) -> anyhow::Result<gpui::StreamImageUpdate> {
        let mut lock = self.0.lock();
        if !lock.streams.contains_key(&id.value()) {
            let stream = NativeStreamImage::new(
                &lock.device,
                &lock.device_context,
                id,
                budgets,
                frame.size,
            )?;
            lock.streams.insert(id.value(), stream);
        }
        lock.streams
            .get_mut(&id.value())
            .ok_or_else(|| anyhow::anyhow!("missing stream owner"))?
            .stage(frame)
    }
    fn retire_stream_image(
        &self,
        id: gpui::StreamImageId,
    ) -> anyhow::Result<Option<gpui::StreamImageCompletion>> {
        let mut state = self.0.lock();
        if let Some(effect) = state.effects.remove(&id.value()) {
            return effect.retire().map(Some);
        }
        let stream = state.streams.remove(&id.value());
        stream.map(NativeStreamImage::retire).transpose()
    }
    fn postprocess_factory(
        &self,
        id: gpui::StreamImageId,
        budgets: &gpui::StreamImageBudgets,
        descriptor: gpui::PostprocessDescriptor,
        cancellation: gpui::BackgroundShaderCancellation,
    ) -> anyhow::Result<Box<dyn FnOnce() -> anyhow::Result<Option<gpui::PreparedStreamImage>> + Send>>
    {
        self.postprocess_preparation(
            id,
            budgets,
            PostprocessInput::Bytecode(descriptor),
            cancellation,
        )
    }
    fn postprocess_wgsl_factory(
        &self,
        id: gpui::StreamImageId,
        budgets: &gpui::StreamImageBudgets,
        descriptor: gpui::WgslPostprocessDescriptor,
        cancellation: gpui::BackgroundShaderCancellation,
    ) -> anyhow::Result<Box<dyn FnOnce() -> anyhow::Result<Option<gpui::PreparedStreamImage>> + Send>>
    {
        descriptor.validate()?;
        self.postprocess_preparation(id, budgets, PostprocessInput::Wgsl(descriptor), cancellation)
    }
    fn adopt_postprocess(
        &self,
        id: gpui::StreamImageId,
        prepared: gpui::PreparedStreamImage,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            unsafe { GetCurrentThreadId() } == self.1,
            "effect adoption must run on its UI thread"
        );
        let prepared = prepared.into_native::<PreparedNativePostprocess>(id)?;
        anyhow::ensure!(!prepared.cancellation.is_cancelled(), "effect preparation was cancelled");
        let mut state = self.0.lock();
        unsafe { state.device.GetDeviceRemovedReason() }?;
        anyhow::ensure!(
            prepared.device_epoch == state.device_epoch
                && prepared.effect.belongs_to_device(&state.device),
            "effect preparation belongs to an obsolete device"
        );
        anyhow::ensure!(
            !state.streams.contains_key(&id.value()) && !state.effects.contains_key(&id.value()),
            "effect source owner is already occupied"
        );
        state.effects.insert(id.value(), prepared.effect);
        Ok(())
    }
    fn stage_background_shader(
        &self,
        id: gpui::StreamImageId,
        budgets: &gpui::StreamImageBudgets,
        frame: &gpui::BackgroundShaderFrame<'_>,
    ) -> anyhow::Result<gpui::StreamImageUpdate> {
        let mut lock = self.0.lock();
        budgets.ensure_available()?;
        lock.streams
            .get_mut(&id.value())
            .ok_or_else(|| anyhow::anyhow!("missing shader owner"))?
            .render_shader(frame)
    }
    fn background_shader_factory(
        &self,
        id: gpui::StreamImageId,
        budgets: &gpui::StreamImageBudgets,
        size: Size<DevicePixels>,
        bytecode: Arc<[u8]>,
        cancellation: gpui::BackgroundShaderCancellation,
    ) -> anyhow::Result<
        Box<dyn FnOnce() -> anyhow::Result<Option<gpui::PreparedBackgroundShader>> + Send>,
    > {
        let lock = self.0.lock();
        anyhow::ensure!(
            !lock.streams.contains_key(&id.value()),
            "owner already contains a prepared source"
        );
        let device = lock.device.clone();
        let context = lock.device_context.clone();
        let epoch = lock.device_epoch;
        let owning_thread = self.1;
        let budgets = budgets.clone();
        Ok(Box::new(move || {
            anyhow::ensure!(
                unsafe { GetCurrentThreadId() } != owning_thread,
                "background shader factory ran on its UI thread"
            );
            if cancellation.is_cancelled() {
                return Ok(None);
            }
            let mut image = NativeStreamImage::new_shader(&device, &context, id, &budgets, size)?;
            // Qualification injection pauses after actual allocation, so close/cancel
            // tests observe held bytes until the worker resumes and releases them.
            if let Some(delay) = std::env::var_os("PEBREL_SHADER_AFTER_ALLOCATION_MS") {
                let delay: u64 = delay.to_string_lossy().parse()?;
                anyhow::ensure!(delay <= 5000, "qualification delay exceeds its finite limit");
                std::thread::sleep(std::time::Duration::from_millis(delay));
            }
            if cancellation.is_cancelled() {
                return Ok(None);
            }
            image.prepare_shader(&bytecode)?;
            if cancellation.is_cancelled() {
                return Ok(None);
            }
            budgets.ensure_available()?;
            Ok(Some(gpui::PreparedBackgroundShader::new(
                id,
                PreparedNativeBackground { image, device_epoch: epoch },
            )))
        }))
    }
    fn adopt_background_shader(
        &self,
        id: gpui::StreamImageId,
        prepared: gpui::PreparedBackgroundShader,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            unsafe { GetCurrentThreadId() } == self.1,
            "background adoption must run on its UI thread"
        );
        let prepared = prepared.into_native::<PreparedNativeBackground>(id)?;
        let mut lock = self.0.lock();
        unsafe { lock.device.GetDeviceRemovedReason() }?;
        anyhow::ensure!(
            prepared.device_epoch == lock.device_epoch
                && prepared.image.belongs_to_device(&lock.device),
            "prepared background has an obsolete device epoch"
        );
        anyhow::ensure!(
            !lock.streams.contains_key(&id.value()),
            "background owner already published"
        );
        lock.streams.insert(id.value(), prepared.image);
        Ok(())
    }
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> anyhow::Result<
            Option<(Size<DevicePixels>, std::borrow::Cow<'a, [u8]>)>,
        >,
    ) -> anyhow::Result<Option<AtlasTile>> {
        let mut lock = self.0.lock();
        if let Some(tile) = lock.tiles_by_key.get(key) {
            Ok(Some(*tile))
        } else {
            let Some((size, bytes)) = build()? else {
                return Ok(None);
            };
            let tile = lock
                .allocate(size, key.texture_kind())
                .ok_or_else(|| anyhow::anyhow!("failed to allocate"))?;
            let texture = lock.texture(tile.texture_id);
            texture.upload(&lock.device_context, tile.bounds, &bytes);
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

        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &mut lock.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut lock.polychrome_textures,
            AtlasTextureKind::Subpixel => &mut lock.subpixel_textures,
        };

        let Some(texture_slot) = textures.textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                textures.free_list.push(texture.id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

impl DirectXAtlasState {
    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = match texture_kind {
                AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
                AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
                AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            };

            if let Some(tile) = textures.iter_mut().rev().find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind)?;
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> Option<&mut DirectXAtlasTexture> {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> =
            Size { width: DevicePixels(1024), height: DevicePixels(1024) };
        // Max texture size for DirectX. See:
        // https://learn.microsoft.com/en-us/windows/win32/direct3d11/overviews-direct3d-11-resources-limits
        const MAX_ATLAS_SIZE: Size<DevicePixels> =
            Size { width: DevicePixels(16384), height: DevicePixels(16384) };
        let size = min_size.min(&MAX_ATLAS_SIZE).max(&DEFAULT_ATLAS_SIZE);
        let pixel_format;
        let bind_flag;
        let bytes_per_pixel;
        match kind {
            AtlasTextureKind::Monochrome => {
                pixel_format = DXGI_FORMAT_R8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 1;
            },
            AtlasTextureKind::Polychrome => {
                pixel_format = DXGI_FORMAT_B8G8R8A8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 4;
            },
            AtlasTextureKind::Subpixel => {
                pixel_format = DXGI_FORMAT_R8G8B8A8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 4;
            },
        }
        let texture_desc = D3D11_TEXTURE2D_DESC {
            Width: size.width.0 as u32,
            Height: size.height.0 as u32,
            MipLevels: 1,
            ArraySize: 1,
            Format: pixel_format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind_flag.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture: Option<ID3D11Texture2D> = None;
        unsafe {
            // This only returns None if the device is lost, which we will recreate later.
            // So it's ok to return None here.
            self.device.CreateTexture2D(&texture_desc, None, Some(&mut texture)).ok()?;
        }
        let texture = texture.unwrap();

        let texture_list = match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
        };
        let index = texture_list.free_list.pop();
        let view = unsafe {
            let mut view = None;
            self.device.CreateShaderResourceView(&texture, None, Some(&mut view)).ok()?;
            [view]
        };
        let atlas_texture = DirectXAtlasTexture {
            id: AtlasTextureId { index: index.unwrap_or(texture_list.textures.len()) as u32, kind },
            bytes_per_pixel,
            allocator: etagere::BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            texture,
            view,
            live_atlas_keys: 0,
        };
        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix).unwrap().as_mut()
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut().unwrap().as_mut()
        }
    }

    fn texture(&self, id: AtlasTextureId) -> &DirectXAtlasTexture {
        match id.kind {
            AtlasTextureKind::Monochrome => {
                &self.monochrome_textures[id.index as usize].as_ref().unwrap()
            },
            AtlasTextureKind::Polychrome => {
                &self.polychrome_textures[id.index as usize].as_ref().unwrap()
            },
            AtlasTextureKind::Subpixel => {
                &self.subpixel_textures[id.index as usize].as_ref().unwrap()
            },
        }
    }
}

impl DirectXAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(device_size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds { origin: etagere_point_to_device(allocation.rectangle.min), size },
            padding: 0,
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn upload(
        &self,
        device_context: &ID3D11DeviceContext,
        bounds: Bounds<DevicePixels>,
        bytes: &[u8],
    ) {
        // `UpdateSubresource` reads `row_pitch * height` bytes from `bytes` based on the
        // `D3D11_BOX` below. If the caller hands us a slice shorter than that, the driver would
        // over-read past the end of the source buffer (potentially by multiple megabytes), so bail
        // out instead. This is a first-insert path rather than a per-frame one, so the check is
        // effectively free.
        let row_bytes = bounds.size.width.to_bytes(self.bytes_per_pixel as u8) as usize;
        let expected = row_bytes * bounds.size.height.0.max(0) as usize;
        if bytes.len() < expected {
            log::error!(
                "DirectXAtlasTexture::upload: source slice is {} bytes but the {}x{} region \
                 requires {} bytes; skipping upload to avoid a driver over-read",
                bytes.len(),
                bounds.size.width.0,
                bounds.size.height.0,
                expected,
            );
            return;
        }
        unsafe {
            device_context.UpdateSubresource(
                &self.texture,
                0,
                Some(&D3D11_BOX {
                    left: bounds.left().0 as u32,
                    top: bounds.top().0 as u32,
                    front: 0,
                    right: bounds.right().0 as u32,
                    bottom: bounds.bottom().0 as u32,
                    back: 1,
                }),
                bytes.as_ptr() as _,
                bounds.size.width.to_bytes(self.bytes_per_pixel as u8),
                0,
            );
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&mut self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn device_size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    etagere::Size::new(size.width.into(), size.height.into())
}

fn etagere_point_to_device(value: etagere::Point) -> Point<DevicePixels> {
    Point { x: DevicePixels::from(value.x), y: DevicePixels::from(value.y) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{ImageId, RenderImageParams};
    use std::borrow::Cow;
    use windows::Win32::{
        Foundation::HMODULE,
        Graphics::{
            Direct3D::D3D_DRIVER_TYPE_WARP,
            Direct3D11::{D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11CreateDevice},
        },
    };

    fn create_atlas() -> Option<DirectXAtlas> {
        let mut device: Option<ID3D11Device> = None;
        let mut device_context: Option<ID3D11DeviceContext> = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_WARP,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut device_context),
            )
        }
        .ok()?;
        Some(DirectXAtlas::new(&device?, &device_context?))
    }

    fn make_image_key(image_id: usize) -> AtlasKey {
        AtlasKey::Image(RenderImageParams { image_id: ImageId(image_id), frame_index: 0 })
    }

    fn insert_tile(atlas: &DirectXAtlas, key: &AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
        atlas
            .get_or_insert_with(key, &mut || {
                let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
                Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
            })
            .expect("allocation should succeed")
            .expect("callback returns Some")
    }

    #[test]
    fn test_remove_deallocates_tile_space_for_reuse() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size { width: DevicePixels(64), height: DevicePixels(64) };
        let big = Size { width: DevicePixels(700), height: DevicePixels(700) };

        let keeper_key = make_image_key(1);
        let big_key_a = make_image_key(2);
        let big_key_b = make_image_key(3);

        let keeper_tile = insert_tile(&atlas, &keeper_key, small);
        let tile_a = insert_tile(&atlas, &big_key_a, big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);

        let tile_b = insert_tile(&atlas, &big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
    }
}
