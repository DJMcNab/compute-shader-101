use core::sync::atomic::AtomicU32;
#[cfg(not(target_arch = "spirv"))]
use core::sync::atomic::Ordering;

#[cfg(target_arch = "spirv")]
use spirv_std::memory::{Scope, Semantics};

pub fn atomic_add_relaxed_storage(atomic: &AtomicU32, amount: u32) -> u32 {
    {
        #[cfg(not(target_arch = "spirv"))]
        atomic.fetch_add(amount, Ordering::Relaxed)
    }
    #[cfg(target_arch = "spirv")]
    unsafe {
        // SAFE: dst is atomic
        atomic_i_add_u32(atomic.as_mut_ptr(), amount)
    }
}

pub fn atomic_store_relaxed_storage(atomic: &AtomicU32, value: u32) {
    #[cfg(not(target_arch = "spirv"))]
    atomic.store(value, Ordering::Relaxed);
    #[cfg(target_arch = "spirv")]
    unsafe {
        // SAFE: dst is atomic
        atomic_store_u32(atomic.as_mut_ptr(), value)
    }
}

pub fn atomic_or_relaxed_storage(atomic: &AtomicU32, value: u32) -> u32 {
    {
        #[cfg(not(target_arch = "spirv"))]
        atomic.fetch_or(value, Ordering::Relaxed)
    }
    #[cfg(target_arch = "spirv")]
    unsafe {
        // SAFE: dst is atomic
        atomic_or_u32(atomic.as_mut_ptr(), value)
    }
}

#[spirv_std::macros::gpu_only]
#[inline]
pub unsafe fn atomic_i_add_u32(ptr: *mut u32, value: u32) -> u32 {
    let mut old = 0;

    asm! {
        "%u32 = OpTypeInt 32 0",
        "%scope = OpConstant %u32 {scope}",
        "%semantics = OpConstant %u32 {semantics}",
        "%value = OpLoad _ {value}",
        "%old = OpAtomicIAdd _ {ptr} %scope %semantics %value",
        "OpStore {old} %old",
        scope = const {Scope::Device as u8},
        semantics = const {Semantics::UNIFORM_MEMORY.bits()},
        ptr = in(reg) ptr,
        old = in(reg) &mut old,
        value = in(reg) &value,
    }

    old
}
#[spirv_std::macros::gpu_only]
#[inline]
pub unsafe fn atomic_store_u32(ptr: *mut u32, value: u32) {
    asm! {
        "%u32 = OpTypeInt 32 0",
        "%scope = OpConstant %u32 {scope}",
        "%semantics = OpConstant %u32 {semantics}",
        "%value = OpLoad _ {value}",
        "OpAtomicStore {ptr} %scope %semantics %value",
        scope = const {Scope::Device as u8},
        semantics = const {Semantics::UNIFORM_MEMORY.bits() },
        ptr = in(reg) ptr,
        value = in(reg) &value,
    }
}

#[spirv_std::macros::gpu_only]
#[inline]
pub unsafe fn atomic_or_u32(ptr: *mut u32, value: u32) -> u32 {
    let mut old = 0;

    asm! {
        "%u32 = OpTypeInt 32 0",
        "%scope = OpConstant %u32 {scope}",
        "%semantics = OpConstant %u32 {semantics}",
        "%value = OpLoad _ {value}",
        "%old = OpAtomicOr _ {ptr} %scope %semantics %value",
        "OpStore {old} %old",
        scope = const {Scope::Device as u8},
        semantics = const {Semantics::UNIFORM_MEMORY.bits()},
        ptr = in(reg) ptr,
        old = in(reg) &mut old,
        value = in(reg) &value,
    }

    old
}
