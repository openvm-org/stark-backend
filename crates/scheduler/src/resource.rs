/// What one node occupies while it is resident.
///
/// Profiles are `const`-constructible so a caller can declare them as constants
/// next to the code that consumes the resource.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceProfile {
    pub gpu_bytes: u64,
    pub host_bytes: u64,
    pub cpu_threads: u32,
}

impl ResourceProfile {
    pub const ZERO: Self = Self::new(0, 0, 0);

    pub const fn new(gpu_bytes: u64, host_bytes: u64, cpu_threads: u32) -> Self {
        Self {
            gpu_bytes,
            host_bytes,
            cpu_threads,
        }
    }
}

/// The ceilings admission never exceeds, one per axis of a [`ResourceProfile`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Budget {
    pub gpu_bytes: u64,
    pub host_bytes: u64,
    pub cpu_threads: u32,
}

impl Budget {
    pub const fn new(gpu_bytes: u64, host_bytes: u64, cpu_threads: u32) -> Self {
        Self {
            gpu_bytes,
            host_bytes,
            cpu_threads,
        }
    }

    /// Whether `want` still fits once `in_use` is already resident.
    ///
    /// Every admission passes through here, which is what keeps the sum of the
    /// admitted profiles inside the budget on every axis.
    pub(crate) fn has_room_for(&self, in_use: &ResourceProfile, want: &ResourceProfile) -> bool {
        fits(in_use.gpu_bytes, want.gpu_bytes, self.gpu_bytes)
            && fits(in_use.host_bytes, want.host_bytes, self.host_bytes)
            && fits(
                u64::from(in_use.cpu_threads),
                u64::from(want.cpu_threads),
                u64::from(self.cpu_threads),
            )
    }
}

/// Checked, so a profile near `u64::MAX` answers "no room" instead of wrapping
/// into a fit.
fn fits(in_use: u64, want: u64, ceiling: u64) -> bool {
    in_use
        .checked_add(want)
        .is_some_and(|total| total <= ceiling)
}
