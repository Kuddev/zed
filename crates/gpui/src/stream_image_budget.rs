use anyhow::Result;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

/// Allocation admission shared by sources or windows. The caller selects the limit.
pub struct StreamImageBudget {
    limit: u64,
    used: AtomicU64,
    peak: AtomicU64,
    allocations: AtomicU64,
    allocation_limit: u64,
    revoked: AtomicBool,
    preparations: AtomicU64,
    preparations_peak: AtomicU64,
}

impl StreamImageBudget {
    /// Creates an empty budget. No renderer resource or task is started.
    pub fn new(limit: u64) -> Arc<Self> {
        Self::with_allocation_limit(limit, 64)
    }
    /// Bounds completion owners as well as bytes. Limits are caller-owned policy.
    pub fn with_allocation_limit(limit: u64, allocation_limit: u64) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            allocations: AtomicU64::new(0),
            allocation_limit,
            revoked: AtomicBool::new(false),
            preparations: AtomicU64::new(0),
            preparations_peak: AtomicU64::new(0),
        })
    }
    /// Current charged bytes, including resources awaiting GPU completion.
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Acquire)
    }
    /// Peak charged bytes.
    pub fn peak(&self) -> u64 {
        self.peak.load(Ordering::Acquire)
    }
    /// Accepted preparation jobs, including jobs whose result was cancelled.
    pub fn preparations(&self) -> u64 {
        self.preparations.load(Ordering::Acquire)
    }
    /// Maximum concurrent preparation admission.
    pub fn preparations_peak(&self) -> u64 {
        self.preparations_peak.load(Ordering::Acquire)
    }
    fn acquire_preparation(&self) -> Result<()> {
        anyhow::ensure!(
            !self.revoked.load(Ordering::Acquire),
            "stream preparation admission revoked"
        );
        let previous = self
            .preparations
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                count.checked_add(1).filter(|next| *next <= self.allocation_limit)
            })
            .map_err(|_| anyhow::anyhow!("stream preparation admission exceeded"))?;
        self.preparations_peak.fetch_max(previous + 1, Ordering::AcqRel);
        Ok(())
    }
    fn acquire(&self, bytes: u64) -> Result<()> {
        anyhow::ensure!(
            !self.revoked.load(Ordering::Acquire),
            "stream budget revoked after native failure"
        );
        self.allocations
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                count.checked_add(1).filter(|next| *next <= self.allocation_limit)
            })
            .map_err(|_| anyhow::anyhow!("stream completion owner admission exceeded"))?;
        let previous = self
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|next| *next <= self.limit)
            })
            .map_err(|_| {
                self.allocations.fetch_sub(1, Ordering::AcqRel);
                anyhow::anyhow!("stream image allocation admission exceeded")
            })?;
        self.peak.fetch_max(previous + bytes, Ordering::AcqRel);
        Ok(())
    }
    fn release(&self, bytes: u64) {
        self.used.fetch_sub(bytes, Ordering::AcqRel);
        self.allocations.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Two-level admission. A lease stays alive through actual GPU retirement.
#[derive(Clone)]
pub struct StreamImageBudgets {
    local: Arc<StreamImageBudget>,
    global: Arc<StreamImageBudget>,
}
impl StreamImageBudgets {
    /// Local may be shared by all placements of a source; global by all sources.
    pub fn new(local: Arc<StreamImageBudget>, global: Arc<StreamImageBudget>) -> Self {
        Self { local, global }
    }
    /// Must be called before native allocation, including staging allocation.
    pub fn reserve(&self, bytes: u64) -> Result<StreamImageLease> {
        anyhow::ensure!(bytes > 0, "empty stream allocation");
        self.local.acquire(bytes)?;
        if let Err(error) = self.global.acquire(bytes) {
            self.local.release(bytes);
            return Err(error);
        }
        Ok(StreamImageLease { budgets: self.clone(), bytes })
    }
    /// Stops fresh GPU work after an unacknowledged native failure.
    pub fn revoke(&self) {
        self.local.revoked.store(true, Ordering::Release);
        self.global.revoked.store(true, Ordering::Release);
    }
    /// Existing allocations stay charged while new work is refused.
    pub fn ensure_available(&self) -> Result<()> {
        anyhow::ensure!(
            !self.local.revoked.load(Ordering::Acquire)
                && !self.global.revoked.load(Ordering::Acquire),
            "stream budget revoked after native failure"
        );
        Ok(())
    }
    /// A job's admission remains held until its worker actually exits.
    pub fn reserve_preparation(&self) -> Result<StreamImagePreparationLease> {
        self.local.acquire_preparation()?;
        if let Err(error) = self.global.acquire_preparation() {
            self.local.preparations.fetch_sub(1, Ordering::AcqRel);
            return Err(error);
        }
        Ok(StreamImagePreparationLease(self.clone()))
    }
}

/// Admission for one queued/running factory; cancellation does not drop this lease.
pub struct StreamImagePreparationLease(StreamImageBudgets);
impl Drop for StreamImagePreparationLease {
    fn drop(&mut self) {
        self.0.local.preparations.fetch_sub(1, Ordering::AcqRel);
        self.0.global.preparations.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Explicit cancellation token, checked between fallible native preparation steps.
#[derive(Clone, Default)]
pub struct BackgroundShaderCancellation(Arc<AtomicBool>);
impl BackgroundShaderCancellation {
    /// Cancels publication; a driver call already in progress must still finish.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    /// Native adapters check this without a UI callback or lock.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Backend-owned allocation lease; clones are deliberately unavailable.
pub struct StreamImageLease {
    budgets: StreamImageBudgets,
    bytes: u64,
}
impl StreamImageLease {
    /// Shared admission controller, without cloning or releasing this allocation.
    pub fn budgets(&self) -> StreamImageBudgets {
        self.budgets.clone()
    }
}
impl Drop for StreamImageLease {
    fn drop(&mut self) {
        self.budgets.local.release(self.bytes);
        self.budgets.global.release(self.bytes);
    }
}
