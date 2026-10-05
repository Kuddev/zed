use std::time::{Duration, Instant};

/// One configured generation and one latest request; requests own no GPU object.
pub(crate) struct SurfaceChange<T> {
    configured: T,
    pending: Option<T>,
    pending_since: Option<Instant>,
    invalidated: bool,
    closed: bool,
}

impl<T: Copy + Eq> SurfaceChange<T> {
    pub fn new(configured: T) -> Self {
        Self { configured, pending: None, pending_since: None, invalidated: false, closed: false }
    }

    pub fn desired(&self) -> T {
        self.pending.unwrap_or(self.configured)
    }

    pub fn pending(&self) -> bool {
        self.pending.is_some()
    }

    pub fn invalidate(&mut self, now: Instant) {
        if !self.closed {
            self.invalidated = true;
            if self.pending.is_none() {
                self.pending = Some(self.configured);
                self.pending_since = Some(now);
            }
        }
    }

    pub fn request(&mut self, value: T, now: Instant) -> bool {
        if self.closed || value == self.desired() {
            return false;
        }
        if value == self.configured && !self.invalidated {
            self.pending = None;
            self.pending_since = None;
        } else {
            self.pending = Some(value);
            self.pending_since.get_or_insert(now);
        }
        true
    }

    /// The caller supplies actual native queue quiescence, never paint/present success.
    pub fn take_ready(&mut self, queue_empty: bool) -> Option<T> {
        if self.closed || !queue_empty {
            return None;
        }
        let value = self.pending.take()?;
        self.configured = value;
        self.invalidated = false;
        self.pending_since = None;
        Some(value)
    }

    pub fn expired(&self, now: Instant, limit: Duration) -> bool {
        self.pending_since.is_some_and(|start| now.saturating_duration_since(start) >= limit)
    }

    pub fn close(&mut self) {
        self.closed = true;
        self.invalidated = false;
        self.pending = None;
        self.pending_since = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn busy_queue_preserves_old_generation_and_latest_request() {
        let mut change = SurfaceChange::new((320, 180));
        let now = Instant::now();
        for index in 0..1000 {
            change.request((640 + index, 360), now);
            assert_eq!(change.take_ready(false), None);
            assert_eq!(change.configured, (320, 180));
        }
        assert_eq!(change.take_ready(true), Some((1639, 360)));
        assert_eq!(change.take_ready(true), None);
    }

    #[test]
    fn identical_sizes_add_no_pending_work() {
        let mut change = SurfaceChange::new((320, 180));
        for _ in 0..1000 {
            assert!(!change.request((320, 180), Instant::now()));
        }
        assert!(!change.pending());
    }

    #[test]
    fn invalid_surface_requires_quiescence_even_with_unchanged_dimensions() {
        let mut change = SurfaceChange::new(1);
        change.invalidate(Instant::now());
        assert_eq!(change.take_ready(false), None);
        assert_eq!(change.take_ready(true), Some(1));
        assert_eq!(change.take_ready(true), None);
    }

    #[test]
    fn invalidation_survives_resize_reversion_and_keeps_its_deadline() {
        let mut change = SurfaceChange::new(1);
        let start = Instant::now();
        change.invalidate(start);
        change.request(2, start + Duration::from_secs(4));
        change.request(1, start + Duration::from_secs(4));
        assert!(change.pending());
        assert!(change.expired(start + Duration::from_secs(5), Duration::from_secs(5)));
        assert_eq!(change.take_ready(false), None);
        assert_eq!(change.take_ready(true), Some(1));
    }

    #[test]
    fn reverting_to_configured_size_cancels_rebuild() {
        let mut change = SurfaceChange::new(1);
        change.request(2, Instant::now());
        assert!(change.request(1, Instant::now()));
        assert!(!change.pending());
        assert_eq!(change.take_ready(true), None);
    }

    #[test]
    fn repeated_requests_do_not_extend_quiescence_deadline() {
        let mut change = SurfaceChange::new(1);
        let now = Instant::now();
        change.request(2, now);
        change.request(3, now + Duration::from_secs(4));
        assert!(!change.expired(now + Duration::from_secs(4), Duration::from_secs(5)));
        assert!(change.expired(now + Duration::from_secs(5), Duration::from_secs(5)));
        assert_eq!(change.take_ready(false), None);
    }

    #[test]
    fn close_rejects_late_ready_and_later_requests() {
        let mut change = SurfaceChange::new(1);
        change.request(2, Instant::now());
        change.close();
        assert_eq!(change.take_ready(true), None);
        assert!(!change.request(3, Instant::now()));
        assert_eq!(change.desired(), 1);
    }

    #[test]
    fn replacement_controller_has_an_independent_generation() {
        let mut old = SurfaceChange::new(1);
        old.request(2, Instant::now());
        old.close();
        let mut replacement = SurfaceChange::new(3);
        replacement.request(4, Instant::now());
        assert_eq!(old.take_ready(true), None);
        assert_eq!(replacement.take_ready(true), Some(4));
    }
}
