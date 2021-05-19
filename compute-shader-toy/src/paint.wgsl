// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

[[group(0), binding(0)]] var outputTex: [[access(write)]] texture_storage_2d<rgba8unorm>;

[[stage(compute), workgroup_size(16, 16)]]
fn main([[builtin(global_invocation_id)]] global_ix: vec3<u32>) {
    let rgba: vec4<f32> = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    let write_ix = vec2<i32>(i32(global_ix.x), i32(global_ix.y));
    textureStore(outputTex, write_ix, rgba);
}
