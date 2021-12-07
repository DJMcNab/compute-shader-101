#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    feature(core_intrinsics),
    feature(atomic_mut_ptr),
    feature(asm),
    feature(asm_const),
    feature(asm_experimental_arch),
    register_attr(spirv)
)]
#![feature(int_log)]

mod atomics;

use core::sync::atomic::AtomicU32;

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;
use spirv_std::{
    arch::{control_barrier, workgroup_memory_barrier_with_group_sync},
    glam::UVec3,
    memory,
};

use rust_gpu_prefix_shared::WORKGROUP_SIZE;

use crate::atomics::{
    atomic_add_relaxed_storage, atomic_or_relaxed_storage, atomic_store_relaxed_storage,
};

const FLAG_NOT_READY: u32 = 0;
const FLAG_AGGREGATE_READY: u32 = 1;
const FLAG_PREFIX_READY: u32 = 2;
const N_SEQ: u32 = 8;

/// https://www.w3.org/TR/WGSL/#sync-builtin-functions
unsafe fn storage_barrier() {
    control_barrier::<
        { memory::Scope::Workgroup as u32 },
        { memory::Scope::Workgroup as u32 },
        { memory::Semantics::UNIFORM_MEMORY.bits() | memory::Semantics::ACQUIRE_RELEASE.bits() },
    >()
}

#[spirv(compute(threads(512)))]
pub fn main(
    #[spirv(workgroup)] part_id: &mut u32,
    #[spirv(workgroup)] scratch: &mut [u32; WORKGROUP_SIZE as usize],
    #[spirv(workgroup)] shared_prefix: &mut u32,
    #[spirv(workgroup)] shared_flag: &mut u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] main_buf: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] state_buf: &mut [AtomicU32],
    #[spirv(local_invocation_id)] local_id: UVec3,
) {
    if local_id.x == 0 {
        *part_id = atomic_add_relaxed_storage(&state_buf[0], 1);
    }
    unsafe {
        workgroup_memory_barrier_with_group_sync();
    }

    let my_part_id = *part_id;
    let mem_base = my_part_id * WORKGROUP_SIZE;
    const ARRAY: [u32; N_SEQ as usize] = [0u32; N_SEQ as usize];
    let mut local = ARRAY;
    let mut el = main_buf[((mem_base + local_id.x) * N_SEQ) as usize];
    local[0] = el;
    for i in 1..N_SEQ {
        el = el + main_buf[((mem_base + local_id.x) * N_SEQ + i) as usize];
        local[i as usize] = el;
    }
    scratch[local_id.x as usize] = el;

    const LOG_WG_SIZE: u32 = WORKGROUP_SIZE.log(2);
    for i in 0..LOG_WG_SIZE {
        unsafe {
            workgroup_memory_barrier_with_group_sync();
        }

        if local_id.x >= (1 << i) {
            el = el + scratch[(local_id.x - (1 << i)) as usize];
        }
        unsafe {
            workgroup_memory_barrier_with_group_sync();
        }

        scratch[local_id.x as usize] = el;
    }
    let mut exclusive_prefix = 0;

    let mut flag = FLAG_AGGREGATE_READY;
    if local_id.x == WORKGROUP_SIZE - 1 {
        atomic_store_relaxed_storage(&state_buf[(my_part_id * 3 + 2) as usize], el);
        if my_part_id == 0 {
            atomic_store_relaxed_storage(&state_buf[(my_part_id * 3 + 3) as usize], el);
            flag = FLAG_PREFIX_READY;
        }
    }
    // make sure these barriers are in uniform control flow
    unsafe {
        storage_barrier();
    }
    if local_id.x == WORKGROUP_SIZE - 1 {
        atomic_store_relaxed_storage(&state_buf[(my_part_id * 3 + 1) as usize], flag);
    }

    if my_part_id != 0 {
        // decoupled look-back
        let mut look_back_ix = my_part_id - 1;
        loop {
            if local_id.x == WORKGROUP_SIZE - 1 {
                *shared_flag =
                    atomic_or_relaxed_storage(&state_buf[(look_back_ix * 3 + 1) as usize], 0);
            }
            unsafe {
                workgroup_memory_barrier_with_group_sync();
            }
            flag = *shared_flag;
            unsafe {
                workgroup_memory_barrier_with_group_sync();
                storage_barrier()
            }
            if flag == FLAG_PREFIX_READY {
                if local_id.x == WORKGROUP_SIZE - 1 {
                    let their_prefix =
                        atomic_or_relaxed_storage(&state_buf[(look_back_ix * 3 + 3) as usize], 0);
                    exclusive_prefix = their_prefix + exclusive_prefix;
                }
                break;
            } else if flag == FLAG_AGGREGATE_READY {
                if local_id.x == WORKGROUP_SIZE - 1 {
                    let their_agg =
                        atomic_or_relaxed_storage(&state_buf[(look_back_ix * 3 + 2) as usize], 0);
                    exclusive_prefix = their_agg + exclusive_prefix;
                }
                look_back_ix = look_back_ix - 1;
            }
            // else spin
        }

        // compute inclusive prefix
        if local_id.x == WORKGROUP_SIZE - 1 {
            let inclusive_prefix = exclusive_prefix + el;
            *shared_prefix = exclusive_prefix;
            atomic_store_relaxed_storage(
                &state_buf[(my_part_id * 3 + 3) as usize],
                inclusive_prefix,
            );
        }
        unsafe {
            storage_barrier();
        }
        if local_id.x == WORKGROUP_SIZE - 1 {
            atomic_store_relaxed_storage(
                &state_buf[(my_part_id * 3 + 1) as usize],
                FLAG_PREFIX_READY,
            );
        }
    }
    let mut prefix = 0;
    unsafe {
        workgroup_memory_barrier_with_group_sync();
    }
    if my_part_id != 0 {
        prefix = *shared_prefix;
    }

    // do final output
    for i in 0..N_SEQ {
        let mut old = 0;
        if local_id.x > 0 {
            old = scratch[(local_id.x - 1) as usize];
        }
        main_buf[((mem_base + local_id.x) * N_SEQ + i) as usize] = prefix + old + local[i as usize];
    }
}
