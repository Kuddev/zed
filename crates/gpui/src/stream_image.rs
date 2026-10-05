use crate::{
    AtlasTile, BackgroundExecutor, BackgroundShaderCancellation, DevicePixels, PlatformAtlas,
    Result, Size, StreamImageBudgets, StreamImagePreparationLease,
};
use std::{
    any::Any,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// An unsubmitted, backend-owned result. Adopt on the UI thread before painting.
pub struct PreparedStreamImage {
    id: StreamImageId,
    native: Box<dyn Any + Send>,
}
impl PreparedStreamImage {
    /// The backend result owns its admitted resources; it has submitted no draw/copy.
    pub fn new<T: Any + Send>(id: StreamImageId, native: T) -> Self {
        Self { id, native: Box::new(native) }
    }
    /// Rejects mismatched owner/backend before native state publication.
    pub fn into_native<T: Any + Send>(self, id: StreamImageId) -> Result<T> {
        anyhow::ensure!(self.id == id, "prepared background belongs to another owner");
        self.native
            .downcast::<T>()
            .map(|native| *native)
            .map_err(|_| anyhow::anyhow!("prepared background belongs to another backend"))
    }
}

/// An unsubmitted shader resource using the same owner/backend adoption contract.
pub type PreparedBackgroundShader = PreparedStreamImage;

/// Native factory work; it performs device operations only, never immediate-context draws.
pub struct StreamImagePreparation {
    work: Box<dyn FnOnce() -> Result<Option<PreparedStreamImage>> + Send>,
    _lease: StreamImagePreparationLease,
}
impl StreamImagePreparation {
    /// Execute only on a background worker. The permit stays held throughout this call.
    pub fn run(self) -> Result<Option<PreparedStreamImage>> {
        (self.work)()
    }
}
/// A shader factory uses the same outstanding-job admission as stream preparation.
pub type BackgroundShaderPreparation = StreamImagePreparation;
/// Source identity within a stream. Changing source creates a new identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamImageId(u64);
impl StreamImageId {
    /// Stable numeric identity for a platform adapter.
    pub fn value(self) -> u64 {
        self.0
    }
}

/// Immutable borrowed frame description. BGRA channels and source row stride are explicit.
pub struct StreamImageFrame<'a> {
    /// Increasing source sequence; unchanged sequences never upload again.
    pub sequence: u64,
    /// Encoded source's admitted output size, independent of placement geometry.
    pub size: Size<DevicePixels>,
    /// Bytes between source rows.
    pub row_stride: usize,
    /// Prepared pixels; the backend copies synchronously before returning.
    pub pixels: &'a [u8],
}

/// One-pass background shader. Its optional b0 uniform never captures text/UI.
pub struct BackgroundShaderFrame<'a> {
    /// Stable until the program's output needs to change.
    pub sequence: u64,
    /// Admitted target dimensions; placement changes do not recreate this target.
    pub size: Size<DevicePixels>,
    /// Previously compiled DirectX fragment bytecode. Other backends fail explicitly.
    pub directx_bytecode: &'a [u8],
    /// Optional 16-byte b0: source width, source height, media time, reserved zero.
    /// The media owner freezes time while inactive; geometry does not change this size.
    pub uniforms: Option<[f32; 4]>,
}
impl BackgroundShaderFrame<'_> {
    /// Bytecode is bounded before native shader creation. This is not a GPU-time sandbox.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.size.width.0 > 0 && self.size.height.0 > 0, "empty shader target");
        anyhow::ensure!(
            self.directx_bytecode.len() <= 64 * 1024 && self.directx_bytecode.starts_with(b"DXBC"),
            "background bytecode is invalid or exceeds 64 KiB"
        );
        if let Some(values) = self.uniforms {
            anyhow::ensure!(
                values.iter().all(|value| value.is_finite())
                    && values[0] == self.size.width.0 as f32
                    && values[1] == self.size.height.0 as f32
                    && values[2] >= 0.0
                    && values[3] == 0.0,
                "background uniform does not match the admitted target/clock"
            );
        }
        Ok(())
    }
}

/// Prepared WGSL background identity; parsing and pipeline creation happen off UI.
pub struct BackgroundWgslFrame<'a> {
    /// Stable until an admitted output needs to change.
    pub sequence: u64,
    /// Source target dimensions, independent of placement geometry.
    pub size: Size<DevicePixels>,
    /// Previously prepared source, never read from a path during paint.
    pub source: &'a str,
    /// Prepared fragment entry point.
    pub entry: &'a str,
}
impl BackgroundWgslFrame<'_> {
    /// Structural checks only; native preparation owns WGSL/ABI validation.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.size.width.0 > 0 && self.size.height.0 > 0, "empty shader target");
        anyhow::ensure!(
            !self.source.is_empty() && self.source.len() <= 64 * 1024,
            "WGSL source exceeds its limit"
        );
        anyhow::ensure!(
            !self.entry.is_empty() && self.entry.len() <= 256,
            "invalid WGSL entry name"
        );
        Ok(())
    }
}
impl StreamImageFrame<'_> {
    /// Checks every byte range before a platform maps/copies pixels.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.size.width.0 > 0 && self.size.height.0 > 0, "empty stream frame");
        let row = (self.size.width.0 as usize)
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("stream row overflow"))?;
        let needed = self
            .row_stride
            .checked_mul(self.size.height.0 as usize - 1)
            .and_then(|bytes| bytes.checked_add(row))
            .ok_or_else(|| anyhow::anyhow!("stream span overflow"))?;
        anyhow::ensure!(
            self.row_stride >= row && needed <= self.pixels.len(),
            "stream row outside prepared pixels"
        );
        Ok(())
    }
}

/// A native completion wait. Invoke only on a background worker, never while painting.
pub struct StreamImageCompletion(pub Box<dyn FnOnce() -> Result<()> + Send>);
impl StreamImageCompletion {
    /// Waits for the recorded submission, without starting a recurring timer.
    pub fn wait(self) -> Result<()> {
        (self.0)()
    }
}

/// Native image state. A previous tile can remain visible under backpressure.
pub struct StreamImageUpdate {
    /// The persistent texture tile, if any frame has been uploaded.
    pub tile: Option<AtlasTile>,
    /// Actual sequence stored in that texture.
    pub sequence: u64,
    /// The last submission must finish before another upload may use the slots.
    pub completion: Option<StreamImageCompletion>,
}

struct Owner {
    id: StreamImageId,
    atlas: Arc<dyn PlatformAtlas>,
    executor: BackgroundExecutor,
    budgets: StreamImageBudgets,
}
impl Drop for Owner {
    fn drop(&mut self) {
        match self.atlas.retire_stream_image(self.id) {
            Ok(Some(retirement)) => self
                .executor
                .spawn(async move {
                    if let Err(error) = retirement.wait() {
                        log::error!("stream image retirement failed: {error}");
                    }
                })
                .detach(),
            Ok(None) => {},
            Err(error) => log::error!("stream image retirement not accepted: {error}"),
        }
    }
}

/// Per-window renderer owner. Scenes retain clones until their references are gone.
#[derive(Clone)]
pub struct StreamImageHandle(Arc<Owner>);
impl fmt::Debug for StreamImageHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("StreamImageHandle").field(&self.0.id).finish()
    }
}
impl StreamImageHandle {
    /// Binds an image owner to an explicitly supplied native renderer atlas.
    /// Platform hosts retain this handle in every referencing scene; allocation,
    /// adoption and staging still enforce that atlas's owning device/thread.
    pub fn from_platform_atlas(
        atlas: Arc<dyn PlatformAtlas>,
        executor: BackgroundExecutor,
        budgets: StreamImageBudgets,
    ) -> Self {
        Self::new(atlas, executor, budgets)
    }

    pub(crate) fn new(
        atlas: Arc<dyn PlatformAtlas>,
        executor: BackgroundExecutor,
        budgets: StreamImageBudgets,
    ) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(Arc::new(Owner {
            id: StreamImageId(NEXT.fetch_add(1, Ordering::Relaxed)),
            atlas,
            executor,
            budgets,
        }))
    }
    /// Identity used for source/device ownership checks.
    pub fn id(&self) -> StreamImageId {
        self.0.id
    }
    /// Captures exact-size native stream allocation; execute the result off UI.
    pub fn prepare_stream_image(
        &self,
        size: Size<DevicePixels>,
        cancellation: BackgroundShaderCancellation,
    ) -> Result<StreamImagePreparation> {
        anyhow::ensure!(size.width.0 > 0 && size.height.0 > 0, "empty stream target");
        let lease = self.0.budgets.reserve_preparation()?;
        let work =
            self.0.atlas.stream_image_factory(self.0.id, &self.0.budgets, size, cancellation)?;
        Ok(StreamImagePreparation { work, _lease: lease })
    }
    /// Publishes only an owning-device result on its UI thread, before staging pixels.
    pub fn adopt_stream_image(&self, prepared: PreparedStreamImage) -> Result<()> {
        self.0.atlas.adopt_stream_image(self.0.id, prepared)
    }
    /// Captures a bounded native factory without creating a renderer object on the UI thread.
    pub fn prepare_background_shader(
        &self,
        size: Size<DevicePixels>,
        bytecode: Arc<[u8]>,
        cancellation: BackgroundShaderCancellation,
    ) -> Result<BackgroundShaderPreparation> {
        BackgroundShaderFrame { sequence: 0, size, directx_bytecode: &bytecode, uniforms: None }
            .validate()?;
        let lease = self.0.budgets.reserve_preparation()?;
        let work = self.0.atlas.background_shader_factory(
            self.0.id,
            &self.0.budgets,
            size,
            bytecode,
            cancellation,
        )?;
        Ok(BackgroundShaderPreparation { work, _lease: lease })
    }
    /// Must run on the owning UI thread, after source-generation/device checks.
    pub fn adopt_background_shader(&self, prepared: PreparedBackgroundShader) -> Result<()> {
        self.0.atlas.adopt_background_shader(self.0.id, prepared)
    }
    /// Captures a WGSL native factory; unsupported backends return an explicit error.
    pub fn prepare_background_wgsl(
        &self,
        size: Size<DevicePixels>,
        source: Arc<str>,
        entry: Arc<str>,
        cancellation: BackgroundShaderCancellation,
    ) -> Result<BackgroundShaderPreparation> {
        BackgroundWgslFrame { sequence: 0, size, source: &source, entry: &entry }.validate()?;
        let lease = self.0.budgets.reserve_preparation()?;
        let work = self.0.atlas.background_wgsl_factory(
            self.0.id,
            &self.0.budgets,
            size,
            source,
            entry,
            cancellation,
        )?;
        Ok(BackgroundShaderPreparation { work, _lease: lease })
    }
    /// Revokes only preparation receipts for a deterministic qualification test.
    pub fn invalidate_background_preparations_for_test(&self) -> Result<()> {
        self.0.atlas.invalidate_background_preparations_for_test()
    }
    pub(crate) fn belongs_to(&self, atlas: &Arc<dyn PlatformAtlas>) -> bool {
        Arc::ptr_eq(&self.0.atlas, atlas)
    }
    pub(crate) fn stage(&self, frame: &StreamImageFrame<'_>) -> Result<StreamImageUpdate> {
        frame.validate()?;
        self.0.atlas.stage_stream_image(self.0.id, &self.0.budgets, frame)
    }
    pub(crate) fn stage_shader(
        &self,
        frame: &BackgroundShaderFrame<'_>,
    ) -> Result<StreamImageUpdate> {
        frame.validate()?;
        self.0.atlas.stage_background_shader(self.0.id, &self.0.budgets, frame)
    }
    pub(crate) fn stage_wgsl(&self, frame: &BackgroundWgslFrame<'_>) -> Result<StreamImageUpdate> {
        frame.validate()?;
        self.0.atlas.stage_background_wgsl(self.0.id, &self.0.budgets, frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DevicePixels, StreamImageBudget, size};

    #[test]
    fn shader_time_uniforms_match_the_admitted_target_and_are_finite() {
        let frame = BackgroundShaderFrame {
            sequence: 1,
            size: size(DevicePixels(960), DevicePixels(540)),
            directx_bytecode: b"DXBC",
            uniforms: Some([960.0, 540.0, 1.0, 0.0]),
        };
        assert!(frame.validate().is_ok());
        for values in [
            [960.0, 540.0, f32::NAN, 0.0],
            [960.0, 540.0, -1.0, 0.0],
            [1280.0, 540.0, 1.0, 0.0],
            [960.0, 540.0, 1.0, 1.0],
        ] {
            assert!(BackgroundShaderFrame { uniforms: Some(values), ..frame }.validate().is_err());
        }
    }
    #[test]
    fn local_and_global_limits_are_reserved_before_allocation() {
        let global = StreamImageBudget::new(16);
        let local = StreamImageBudget::new(12);
        let limits = StreamImageBudgets::new(local.clone(), global.clone());
        let lease = limits.reserve(12).expect("admission");
        assert!(limits.reserve(1).is_err());
        assert_eq!((local.used(), global.used()), (12, 12));
        drop(lease);
        assert_eq!((local.used(), global.used()), (0, 0));
    }
    #[test]
    fn global_failure_rolls_back_local_but_retirement_keeps_old_charge() {
        let global = StreamImageBudget::new(10);
        let first = StreamImageBudgets::new(StreamImageBudget::new(10), global.clone());
        let second_local = StreamImageBudget::new(10);
        let second = StreamImageBudgets::new(second_local.clone(), global.clone());
        let retired = first.reserve(8).expect("admission");
        assert!(second.reserve(3).is_err());
        assert_eq!(second_local.used(), 0);
        assert_eq!(global.used(), 8);
        drop(retired);
        assert!(second.reserve(10).is_ok());
    }
    #[test]
    fn source_stride_padding_is_bounded_without_charging_placement_size() {
        let frame = StreamImageFrame {
            sequence: 1,
            size: size(DevicePixels(2), DevicePixels(2)),
            row_stride: 12,
            pixels: &[0; 20],
        };
        assert!(frame.validate().is_ok());
        assert!(StreamImageFrame { pixels: &[0; 19], ..frame }.validate().is_err());
        assert!(StreamImageFrame { row_stride: 4, ..frame }.validate().is_err());
    }
}
