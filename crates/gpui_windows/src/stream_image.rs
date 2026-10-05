use gpui::{
    AtlasTextureId, AtlasTextureKind, AtlasTile, Bounds, DevicePixels, Size, StreamImageBudgets,
    StreamImageCompletion, StreamImageFrame, StreamImageId, StreamImageLease, StreamImageUpdate,
    TileId, point,
};
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use windows::{
    Win32::{
        Foundation::{CloseHandle, WAIT_OBJECT_0, WAIT_TIMEOUT},
        Graphics::{Direct3D11::*, Dxgi::Common::*},
        System::Threading::{CreateEventW, WaitForSingleObject},
    },
    core::Interface,
};
#[path = "background_shader.rs"]
pub(super) mod background_shader;
use background_shader::NativeBackgroundShader;

pub(super) const STREAM_TEXTURE_BIT: u32 = 0x8000_0000;
static CREATED: AtomicU64 = AtomicU64::new(0);
static UPLOADS: AtomicU64 = AtomicU64::new(0);
static BUSY: AtomicU64 = AtomicU64::new(0);
static SUBMISSIONS: AtomicU64 = AtomicU64::new(0);
static RETIRED: AtomicU64 = AtomicU64::new(0);
static ACKNOWLEDGED: AtomicU64 = AtomicU64::new(0);
static WAIT_TIMEOUTS: AtomicU64 = AtomicU64::new(0);
static QUARANTINED: AtomicU64 = AtomicU64::new(0);
static SHADER_PIPELINES: AtomicU64 = AtomicU64::new(0);
static SHADER_DRAWS: AtomicU64 = AtomicU64::new(0);
static QUARANTINE: OnceLock<Mutex<Vec<NativeStreamImage>>> = OnceLock::new();
const COMPLETION_TIMEOUT_MS: u32 = 5000;

pub(super) struct NativeStreamImage {
    id: StreamImageId,
    size: Size<DevicePixels>,
    texture: ID3D11Texture2D,
    staging: Option<ID3D11Texture2D>,
    view: [Option<ID3D11ShaderResourceView>; 1],
    context: ID3D11DeviceContext4,
    device: ID3D11Device,
    fence: ID3D11Fence,
    fence_value: u64,
    sequence: u64,
    uploaded: bool,
    shader: Option<NativeBackgroundShader>,
    retained: Option<Box<dyn Send>>,
    // Last field releases admission only after native owners have been dropped.
    _lease: StreamImageLease,
}

fn wait_fence(fence: &ID3D11Fence, device: &ID3D11Device, value: u64) -> anyhow::Result<()> {
    // A qualification-only error injection exercises ownership without hanging a GPU.
    static INJECTED: AtomicBool = AtomicBool::new(false);
    if std::env::var_os("PEBREL_STREAM_INJECT_WAIT_TIMEOUT").is_some()
        && !INJECTED.swap(true, Ordering::AcqRel)
    {
        WAIT_TIMEOUTS.fetch_add(1, Ordering::Relaxed);
        anyhow::bail!("injected stream GPU completion timeout; no hardware hang requested");
    }
    if unsafe { fence.GetCompletedValue() } >= value {
        unsafe { device.GetDeviceRemovedReason() }?;
        return Ok(());
    }
    let event = unsafe { CreateEventW(None, true, false, None) }?;
    let result = (|| {
        unsafe { fence.SetEventOnCompletion(value, event) }?;
        let status = unsafe { WaitForSingleObject(event, COMPLETION_TIMEOUT_MS) };
        if status == WAIT_TIMEOUT {
            WAIT_TIMEOUTS.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!("stream GPU completion exceeded {COMPLETION_TIMEOUT_MS} ms");
        }
        anyhow::ensure!(status == WAIT_OBJECT_0, "stream GPU completion wait failed");
        unsafe { device.GetDeviceRemovedReason() }?;
        Ok(())
    })();
    if let Err(error) = unsafe { CloseHandle(event) } {
        log::error!("stream completion event close failed: {error}");
    }
    result
}

impl NativeStreamImage {
    pub fn new(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        id: StreamImageId,
        budgets: &StreamImageBudgets,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<Self> {
        Self::allocate(device, context, id, budgets, size, false, 0)
    }
    pub fn new_shader(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        id: StreamImageId,
        budgets: &StreamImageBudgets,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<Self> {
        Self::allocate(device, context, id, budgets, size, true, 16)
    }
    pub fn new_effect_input(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        id: StreamImageId,
        budgets: &StreamImageBudgets,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<Self> {
        Self::allocate(device, context, id, budgets, size, true, 0)
    }
    fn allocate(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        id: StreamImageId,
        budgets: &StreamImageBudgets,
        size: Size<DevicePixels>,
        shader_only: bool,
        uniform_bytes: u64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            id.value() < u64::from(STREAM_TEXTURE_BIT),
            "stream texture identity exhausted"
        );
        let device5: ID3D11Device5 = device.cast()?;
        let context4: ID3D11DeviceContext4 = context.cast()?;
        let width = size.width.0 as u32;
        let height = size.height.0 as u32;
        let row = u64::from(width)
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("stream row overflow"))?;
        let staging_pitch = if shader_only { 0 } else { row.div_ceil(256) * 256 };
        let admitted = (row + staging_pitch)
            .checked_mul(u64::from(height))
            .and_then(|bytes| bytes.checked_add(uniform_bytes))
            .ok_or_else(|| anyhow::anyhow!("stream allocation overflow"))?;
        let lease = budgets.reserve(admitted)?;
        let description = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: if shader_only {
                (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32
            } else {
                D3D11_BIND_SHADER_RESOURCE.0 as u32
            },
            ..Default::default()
        };
        let mut texture = None;
        unsafe { device.CreateTexture2D(&description, None, Some(&mut texture)) }?;
        let texture = texture.ok_or_else(|| anyhow::anyhow!("missing stream texture"))?;
        let mut view = None;
        unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut view)) }?;
        let staging_description = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
            ..description
        };
        let mut staging = None;
        if !shader_only {
            unsafe { device.CreateTexture2D(&staging_description, None, Some(&mut staging)) }?;
            anyhow::ensure!(staging.is_some(), "missing stream upload texture");
        }
        let mut fence: Option<ID3D11Fence> = None;
        unsafe { device5.CreateFence(0, D3D11_FENCE_FLAG_NONE, &mut fence) }?;
        let fence = fence.ok_or_else(|| anyhow::anyhow!("missing stream fence"))?;
        CREATED.fetch_add(if shader_only { 1 } else { 2 }, Ordering::Relaxed);
        Ok(Self {
            id,
            size,
            texture,
            staging,
            view: [view],
            context: context4,
            device: device.clone(),
            fence,
            fence_value: 0,
            sequence: 0,
            uploaded: false,
            shader: None,
            retained: None,
            _lease: lease,
        })
    }
    fn tile(&self) -> Option<AtlasTile> {
        self.uploaded.then_some(AtlasTile {
            texture_id: AtlasTextureId {
                index: STREAM_TEXTURE_BIT | self.id.value() as u32,
                kind: AtlasTextureKind::Polychrome,
            },
            tile_id: TileId(self.id.value() as u32),
            padding: 0,
            bounds: Bounds::new(point(DevicePixels(0), DevicePixels(0)), self.size),
        })
    }
    pub fn prepare_shader(&mut self, bytecode: &[u8]) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.staging.is_none() && !self.uploaded,
            "shader preparation requires an unused shader owner"
        );
        self._lease.budgets().ensure_available()?;
        unsafe { self.device.GetDeviceRemovedReason() }?;
        self.shader = Some(NativeBackgroundShader::new(&self.device, &self.texture, bytecode)?);
        SHADER_PIPELINES.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    pub fn belongs_to_device(&self, device: &ID3D11Device) -> bool {
        self.device.as_raw() == device.as_raw()
    }
    pub fn view(&self) -> [Option<ID3D11ShaderResourceView>; 1] {
        self.view.clone()
    }
    pub fn texture(&self) -> &ID3D11Texture2D {
        &self.texture
    }
    pub fn ready_for_effect(&self) -> anyhow::Result<bool> {
        self._lease.budgets().ensure_available()?;
        unsafe { self.device.GetDeviceRemovedReason() }?;
        let ready = unsafe { self.fence.GetCompletedValue() } >= self.fence_value;
        if !ready {
            unsafe { self.context.Flush() };
        }
        Ok(ready)
    }
    pub fn retain_for_retirement(&mut self, resources: impl Send + 'static) {
        // 多段效果与输入纹理共用一次 fence 确认和隔离；不会提前释放其余纹理/许可证。
        self.retained = Some(Box::new(resources));
    }
    pub fn signal(&mut self) -> anyhow::Result<()> {
        self.fence_value = self
            .fence_value
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("stream fence exhausted"))?;
        unsafe { self.context.Signal(&self.fence, self.fence_value) }?;
        SUBMISSIONS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn completion(&self) -> StreamImageCompletion {
        let fence = self.fence.clone();
        let device = self.device.clone();
        let value = self.fence_value;
        let budgets = self._lease.budgets();
        StreamImageCompletion(Box::new(move || {
            let result = wait_fence(&fence, &device, value);
            if result.is_err() {
                budgets.revoke();
            }
            result
        }))
    }
    pub fn stage(&mut self, frame: &StreamImageFrame<'_>) -> anyhow::Result<StreamImageUpdate> {
        self._lease.budgets().ensure_available()?;
        anyhow::ensure!(
            self.shader.is_none(),
            "a shader source cannot become a pixel source without a new owner"
        );
        anyhow::ensure!(
            frame.size == self.size,
            "stream dimensions changed; create a new admitted source owner"
        );
        unsafe { self.device.GetDeviceRemovedReason() }?;
        if self.uploaded && frame.sequence <= self.sequence {
            return Ok(StreamImageUpdate {
                tile: self.tile(),
                sequence: self.sequence,
                completion: None,
            });
        }
        if unsafe { self.fence.GetCompletedValue() } < self.fence_value {
            BUSY.fetch_add(1, Ordering::Relaxed);
            unsafe { self.context.Flush() };
            return Ok(StreamImageUpdate {
                tile: self.tile(),
                sequence: self.sequence,
                completion: Some(self.completion()),
            });
        }
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        let staging = self
            .staging
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("shader source has no upload slot; create a new pixel source owner")
            })?
            .clone();
        let mapped_result = unsafe {
            self.context.Map(
                &staging,
                0,
                D3D11_MAP_WRITE,
                D3D11_MAP_FLAG_DO_NOT_WAIT.0 as u32,
                Some(&mut mapped),
            )
        };
        if let Err(error) = mapped_result {
            if error.code() == windows::Win32::Graphics::Dxgi::DXGI_ERROR_WAS_STILL_DRAWING {
                // Creation/residency work can still be pending before the first upload fence.
                self.signal()?;
                unsafe { self.context.Flush() };
                BUSY.fetch_add(1, Ordering::Relaxed);
                return Ok(StreamImageUpdate {
                    tile: self.tile(),
                    sequence: self.sequence,
                    completion: Some(self.completion()),
                });
            }
            return Err(error.into());
        }
        let row = self.size.width.0 as usize * 4;
        let pitch = mapped.RowPitch as usize;
        let admitted_pitch = row.div_ceil(256) * 256;
        let copied = (|| -> anyhow::Result<()> {
            anyhow::ensure!(
                !mapped.pData.is_null() && pitch >= row && pitch <= admitted_pitch,
                "native staging pitch exceeds its allocation admission"
            );
            for y in 0..self.size.height.0 as usize {
                let source = &frame.pixels[y * frame.row_stride..y * frame.row_stride + row];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        source.as_ptr(),
                        mapped.pData.cast::<u8>().add(y * pitch),
                        row,
                    );
                }
            }
            Ok(())
        })();
        unsafe { self.context.Unmap(&staging, 0) };
        copied?;
        unsafe { self.context.CopyResource(&self.texture, &staging) };
        self.signal()?;
        self.sequence = frame.sequence;
        self.uploaded = true;
        UPLOADS.fetch_add(1, Ordering::Relaxed);
        Ok(StreamImageUpdate { tile: self.tile(), sequence: self.sequence, completion: None })
    }
    pub fn render_shader(
        &mut self,
        frame: &gpui::BackgroundShaderFrame<'_>,
    ) -> anyhow::Result<StreamImageUpdate> {
        self._lease.budgets().ensure_available()?;
        anyhow::ensure!(
            self.staging.is_none(),
            "pixel source cannot become a shader source without a new owner"
        );
        anyhow::ensure!(
            frame.size == self.size,
            "shader target changed; create a new admitted source owner"
        );
        anyhow::ensure!(
            self.shader.is_some() || !self.uploaded,
            "a pixel source cannot become a shader without a new owner"
        );
        unsafe { self.device.GetDeviceRemovedReason() }?;
        if self.uploaded && frame.sequence <= self.sequence {
            return Ok(StreamImageUpdate {
                tile: self.tile(),
                sequence: self.sequence,
                completion: None,
            });
        }
        if unsafe { self.fence.GetCompletedValue() } < self.fence_value {
            BUSY.fetch_add(1, Ordering::Relaxed);
            unsafe { self.context.Flush() };
            return Ok(StreamImageUpdate {
                tile: self.tile(),
                sequence: self.sequence,
                completion: Some(self.completion()),
            });
        }
        let shader = self
            .shader
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("background shader must be prepared before paint"))?;
        anyhow::ensure!(
            shader.bytecode.as_ref() == frame.directx_bytecode,
            "shader changed; create a new source owner"
        );
        let viewport = D3D11_VIEWPORT {
            Width: self.size.width.0 as f32,
            Height: self.size.height.0 as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
            ..Default::default()
        };
        unsafe {
            // Keep GPUI's rasterizer contract; restore it before its next scene.
            let rasterizer = self.context.RSGetState()?;
            let mut previous_constants = [None];
            self.context.PSGetConstantBuffers(0, Some(&mut previous_constants));
            self.context.PSSetShaderResources(0, Some(&[None]));
            self.context.OMSetRenderTargets(Some(&[Some(shader.target.clone())]), None);
            self.context.ClearRenderTargetView(&shader.target, &[0.0; 4]);
            self.context.OMSetBlendState(None, None, u32::MAX);
            self.context.RSSetState(None);
            self.context.RSSetViewports(Some(&[viewport]));
            self.context.IASetInputLayout(None);
            self.context.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            self.context.VSSetShader(&shader.vertex, None);
            self.context.PSSetShader(&shader.fragment, None);
            if let Some(values) = frame.uniforms {
                // 上次提交的 fence 已完成，复用 b0 不会改写仍在执行的帧。
                self.context.UpdateSubresource(
                    &shader.uniforms,
                    0,
                    None,
                    values.as_ptr().cast(),
                    0,
                    0,
                );
                self.context.PSSetConstantBuffers(0, Some(&[Some(shader.uniforms.clone())]));
            } else {
                self.context.PSSetConstantBuffers(0, Some(&[None]));
            }
            self.context.Draw(3, 0);
            self.context.PSSetConstantBuffers(0, Some(&previous_constants));
            self.context.OMSetRenderTargets(None, None);
            self.context.RSSetState(&rasterizer);
        }
        self.signal()?;
        self.sequence = frame.sequence;
        self.uploaded = true;
        SHADER_DRAWS.fetch_add(1, Ordering::Relaxed);
        Ok(StreamImageUpdate { tile: self.tile(), sequence: self.sequence, completion: None })
    }
    pub fn retire(mut self) -> anyhow::Result<StreamImageCompletion> {
        // Immediate-context state also owns COM references. Retiring a Rust owner
        // must release those bindings before acknowledging and dropping its lease.
        // The renderer installs resources/shaders again before every later draw.
        unsafe {
            self.context.PSSetShaderResources(0, Some(&[None]));
            if self.shader.is_some() {
                self.context.PSSetShader(None::<&ID3D11PixelShader>, None);
                self.context.VSSetShader(None::<&ID3D11VertexShader>, None);
            }
        }
        if let Err(error) = self.signal() {
            if unsafe { self.device.GetDeviceRemovedReason() }.is_ok() {
                quarantine(self);
            }
            return Err(error);
        }
        unsafe { self.context.Flush() };
        RETIRED.fetch_add(1, Ordering::Relaxed);
        Ok(StreamImageCompletion(Box::new(move || {
            let result = wait_fence(&self.fence, &self.device, self.fence_value);
            // Device removal also terminates that device's submitted work.
            if result.is_ok() || unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                ACKNOWLEDGED.fetch_add(1, Ordering::Relaxed);
                drop(self);
                report();
                result
            } else {
                quarantine(self);
                result
            }
        })))
    }
}

fn quarantine(image: NativeStreamImage) {
    image._lease.budgets().revoke();
    // Admission also bounds allocation owners. Revocation prevents repeated failures
    // from admitting fresh textures or creating an unbounded queue of waiters.
    QUARANTINE
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .expect("stream quarantine mutex poisoned")
        .push(image);
    QUARANTINED.fetch_add(1, Ordering::Relaxed);
    report();
}

pub(super) fn recover_quarantined() {
    let Some(quarantine) = QUARANTINE.get() else {
        return;
    };
    let mut retained = quarantine.lock().expect("stream quarantine mutex poisoned");
    retained.retain(|image| {
        let completed = unsafe { image.fence.GetCompletedValue() } >= image.fence_value;
        let removed = unsafe { image.device.GetDeviceRemovedReason() }.is_err();
        if completed || removed {
            QUARANTINED.fetch_sub(1, Ordering::Relaxed);
            ACKNOWLEDGED.fetch_add(1, Ordering::Relaxed);
            false
        } else {
            true
        }
    });
    // Explicit recovery only. Idle/static sources start no retirement poll.
}

pub(super) fn report() {
    let Some(directory) = std::env::var_os("PEBREL_MEDIA_ATLAS_DIR") else {
        return;
    };
    let (programs_created, program_cache_hits) = background_shader::statistics();
    let report = format!(
        "{{\"created_textures\":{},\"uploads\":{},\"backpressure\":{},\"fence_signals\":{},\"retired\":{},\"acknowledged\":{},\"completion_timeouts\":{},\"quarantined\":{},\"wait_deadline_ms\":{},\"shader_pipelines\":{},\"shader_draws\":{},\"native_programs_created\":{},\"program_cache_hits\":{}}}",
        CREATED.load(Ordering::Acquire),
        UPLOADS.load(Ordering::Acquire),
        BUSY.load(Ordering::Acquire),
        SUBMISSIONS.load(Ordering::Acquire),
        RETIRED.load(Ordering::Acquire),
        ACKNOWLEDGED.load(Ordering::Acquire),
        WAIT_TIMEOUTS.load(Ordering::Acquire),
        QUARANTINED.load(Ordering::Acquire),
        COMPLETION_TIMEOUT_MS,
        SHADER_PIPELINES.load(Ordering::Acquire),
        SHADER_DRAWS.load(Ordering::Acquire),
        programs_created,
        program_cache_hits
    );
    if let Err(error) = std::fs::write(
        std::path::PathBuf::from(directory).join(format!("streams-{}.json", std::process::id())),
        report,
    ) {
        log::error!("stream metrics write failed: {error}");
    }
}
