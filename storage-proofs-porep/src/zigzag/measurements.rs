//! Optional operation boundaries for a process-wide ZigZag measurement observer.
//!
//! The observer owns the clock, CPU counters and sampler. Normal FFI/Curio calls leave it
//! unset. IDs and layer numbers remain meaningful if operations later run concurrently;
//! process measurements in those windows must not be summed as worker-exclusive usage.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

static OBSERVER: OnceLock<fn(OperationEvent)> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug)]
pub struct OperationEvent {
    pub id: u64,
    pub name: &'static str,
    /// Zero-based encoding layer; TreeD has no encoding layer.
    pub layer: Option<usize>,
    pub boundary: OperationBoundary,
    pub details: Option<OperationDetails>,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct OperationDetails {
    pub partition_start: Option<usize>,
    pub partition_count: Option<usize>,
    pub query_family: Option<&'static str>,
    pub query_points: Option<usize>,
    pub encoded_bytes: Option<u64>,
    pub decoded_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub enum OperationBoundary {
    Start,
    End { completed: bool },
}

/// Install once, before sealing starts. The callback must not panic or call ZigZag APIs.
pub fn set_operation_observer(observer: fn(OperationEvent)) -> Result<(), fn(OperationEvent)> {
    OBSERVER.set(observer)
}

#[must_use]
pub struct OperationGuard {
    event: Option<OperationEvent>,
    completed: bool,
}

impl OperationGuard {
    pub fn enter(name: &'static str, layer: Option<usize>) -> Self {
        Self::enter_optional_details(name, layer, None)
    }

    pub fn enter_with_details(
        name: &'static str,
        layer: Option<usize>,
        details: OperationDetails,
    ) -> Self {
        Self::enter_optional_details(name, layer, Some(details))
    }

    fn enter_optional_details(
        name: &'static str,
        layer: Option<usize>,
        details: Option<OperationDetails>,
    ) -> Self {
        let event = OBSERVER.get().map(|observer| {
            let event = OperationEvent {
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                name,
                layer,
                boundary: OperationBoundary::Start,
                details,
            };
            observer(event);
            event
        });
        Self {
            event,
            completed: false,
        }
    }

    pub fn finish(mut self) {
        self.completed = true;
    }
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        if let (Some(mut event), Some(observer)) = (self.event, OBSERVER.get()) {
            event.boundary = OperationBoundary::End {
                completed: self.completed,
            };
            observer(event);
        }
    }
}
