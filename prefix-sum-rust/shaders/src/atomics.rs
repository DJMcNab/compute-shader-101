use core::sync::atomic::{AtomicU32, Ordering};

pub fn atomic_add_relaxed(atomic: &AtomicU32, amount: u32) -> u32 {
    {
        #[cfg(not(target_arch = "spirv"))]
        atomic.fetch_add(amount, Ordering::Relaxed)
    }
    #[cfg(target_arch = "spirv")]
    unsafe {
        // SAFE: dst is atomic
        core::intrinsics::atomic_xadd_relaxed(atomic.as_mut_ptr(), amount)
    }
}

pub fn atomic_store_relaxed(atomic: &AtomicU32, value: u32) {
    #[cfg(not(target_arch = "spirv"))]
    atomic.store(value, Ordering::Relaxed);
    #[cfg(target_arch = "spirv")]
    unsafe {
        // SAFE: dst is atomic
        core::intrinsics::atomic_store_relaxed(atomic.as_mut_ptr(), value)
    }
}

pub fn atomic_or_relaxed(atomic: &AtomicU32, value: u32) -> u32 {
    {
        #[cfg(not(target_arch = "spirv"))]
        atomic.fetch_or(value, Ordering::Relaxed)
    }
    #[cfg(target_arch = "spirv")]
    unsafe {
        // SAFE: dst is atomic
        core::intrinsics::atomic_or_relaxed(atomic.as_mut_ptr(), value)
    }
}
