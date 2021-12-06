#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    feature(core_intrinsics),
    feature(atomic_mut_ptr),
    register_attr(spirv)
)]

mod atomics;

use core::sync::atomic::AtomicU32;

use spirv_std::glam::UVec3;
#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use rust_gpu_prefix_shared::WORKGROUP_SIZE;

const FLAG_NOT_READY: u32 = 0;
const FLAG_AGGREGATE_READY: u32 = 1;
const FLAG_PREFIX_READY: u32 = 2;
const N_SEQ: u32 = 8;

#[spirv(compute(threads(512)))]
pub fn main(
    #[spirv(workgroup)] part_id: &mut u32,
    #[spirv(workgroup)] scratch: &mut [u32; WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_prefix: &mut u32,
    #[spirv(workgroup)] shared_flag: &mut u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] main_buf: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] state_buf: &mut [AtomicU32],
    #[spirv(local_invocation_id)] local_id: UVec3,
) {
}
