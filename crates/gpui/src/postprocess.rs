use crate::{Bounds, ContentMask, DevicePixels, DrawOrder, Result, ScaledPixels, Size};
use std::sync::Arc;

/// Prepared native programs for an ordered, exact-resolution surface effect.
#[derive(Clone)]
pub struct PostprocessDescriptor {
    /// Physical size of the surface, before clipping against the window.
    pub size: Size<DevicePixels>,
    /// Constant-buffer size shared by all passes; must be a multiple of 16 bytes.
    pub uniform_size: usize,
    /// Already compiled fragment programs, in application-defined execution order.
    pub directx_passes: Arc<[Arc<[u8]>]>,
}

impl PostprocessDescriptor {
    /// Bounds descriptor costs before a native factory allocates any resource.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.size.width.0 > 0 && self.size.height.0 > 0, "empty effect surface");
        anyhow::ensure!(
            (16..=16 * 1024).contains(&self.uniform_size) && self.uniform_size % 16 == 0,
            "effect uniforms must contain 16-byte blocks within 16 KiB"
        );
        anyhow::ensure!(
            (1..=8).contains(&self.directx_passes.len()),
            "an effect requires between one and eight passes"
        );
        for code in self.directx_passes.iter() {
            anyhow::ensure!(
                code.len() <= 64 * 1024 && code.starts_with(b"DXBC"),
                "invalid or oversized native effect program"
            );
        }
        self.texture_bytes()?;
        Ok(())
    }

    /// Exact reusable texture storage; never substitutes a lower text resolution.
    pub fn texture_bytes(&self) -> Result<u64> {
        let width = u64::try_from(self.size.width.0)?;
        let height = u64::try_from(self.size.height.0)?;
        // 所有段交替使用两张纹理，最后复制回原区域；这样每段都使用局部像素坐标。
        width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(8))
            .ok_or_else(|| anyhow::anyhow!("effect texture size overflow"))
    }
}

/// A render failure belongs to its effect owner, not to unrelated window content.
#[derive(Clone, Debug, Default)]
pub struct PostprocessFeedback(Arc<parking_lot::Mutex<Option<String>>>);

impl PostprocessFeedback {
    /// Records the first failure until the owning view observes it.
    pub fn record_error(&self, message: String) {
        self.0.lock().get_or_insert(message);
    }

    /// Consumes a native failure so the owning view can disable/report this effect.
    pub fn take_error(&self) -> Option<String> {
        self.0.lock().take()
    }
}

/// A scoped effect evaluated after preceding content and before later overlays.
#[derive(Clone)]
pub struct PaintPostprocess {
    /// Assigned by the scene's ordering barrier.
    pub order: DrawOrder,
    /// Full surface geometry, in physical pixels.
    pub bounds: Bounds<ScaledPixels>,
    /// Final copy is restricted to the currently visible surface region.
    pub content_mask: ContentMask<ScaledPixels>,
    /// Retains prepared resources through scene caching, replay and native completion.
    pub owner: crate::StreamImageHandle,
    /// Owned immutable data; renderers never borrow live application state.
    pub uniforms: std::sync::Arc<[u8]>,
    /// Native failures are observed by this surface's owning view.
    pub feedback: crate::PostprocessFeedback,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::size;

    fn descriptor(passes: usize) -> PostprocessDescriptor {
        PostprocessDescriptor {
            size: size(DevicePixels(1920), DevicePixels(1080)),
            uniform_size: 64,
            directx_passes: vec![Arc::from(&b"DXBC"[..]); passes].into(),
        }
    }

    #[test]
    fn chain_storage_does_not_grow_per_pass() {
        assert_eq!(descriptor(1).texture_bytes().unwrap(), 1920 * 1080 * 8);
        assert_eq!(descriptor(2).texture_bytes().unwrap(), 1920 * 1080 * 8);
        assert_eq!(descriptor(8).texture_bytes().unwrap(), 1920 * 1080 * 8);
    }

    #[test]
    fn descriptors_reject_invalid_admission_before_native_work() {
        assert!(descriptor(1).validate().is_ok());
        assert!(descriptor(0).validate().is_err());
        assert!(descriptor(9).validate().is_err());
        for size in [0, 15, 17, 16 * 1024 + 16] {
            let mut invalid = descriptor(1);
            invalid.uniform_size = size;
            assert!(invalid.validate().is_err());
        }
        let mut invalid = descriptor(1);
        invalid.size.width = DevicePixels(-1);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn feedback_keeps_the_first_failure_without_an_unbounded_queue() {
        let feedback = PostprocessFeedback::default();
        feedback.record_error("first".into());
        feedback.record_error("later".into());
        assert_eq!(feedback.take_error().as_deref(), Some("first"));
        assert!(feedback.take_error().is_none());
    }
}
