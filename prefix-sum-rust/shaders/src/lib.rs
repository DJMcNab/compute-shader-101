#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv)
)]

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use spirv_std::{
    glam::{vec2, vec4, UVec3, Vec2, Vec4},
    image, Sampler,
};

#[spirv(compute(threads(16, 16)))]
pub fn main() {}
