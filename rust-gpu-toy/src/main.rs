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

//! A simple compute shader example that draws into a window, based on wgpu.

use std::time::Instant;

use rust_gpu_toy_shared::Config;
use wgpu::{Extent3d, ShaderModule};

use winit::dpi::PhysicalSize;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use spirv_builder::{CompileResult, SpirvBuilder};

struct Pipelines {
    compute_pipeline: wgpu::ComputePipeline,

    copy_pipeline: wgpu::RenderPipeline,
}

struct Layouts {
    compute_bg_layout: wgpu::BindGroupLayout,
    copy_bg_layout: wgpu::BindGroupLayout,

    compute_pipeline_layout: wgpu::PipelineLayout,
    copy_pipeline_layout: wgpu::PipelineLayout,
}

/// State of the runner, inspired by
/// https://sotrh.github.io/learn-wgpu/
struct State {
    pipelines: Pipelines,
    layouts: Layouts,

    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,

    surface_config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    sampler: wgpu::Sampler,

    compute_bind_group: wgpu::BindGroup,
    copy_bind_group: wgpu::BindGroup,

    start_time: Instant,
}

impl State {
    async fn new(window: &Window, compilation: CompileResult) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: Default::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("error finding adapter");
        let features = wgpu::Features::PUSH_CONSTANTS;
        let limits = wgpu::Limits {
            max_push_constant_size: std::mem::size_of::<Config>() as u32,
            ..Default::default()
        };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features,
                    limits,
                },
                None,
            )
            .await
            .expect("error creating device");
        let preferred_format = surface
            .get_preferred_format(&adapter)
            .expect("Surface is compatible with adapter, as adapter creation succeeded");
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: preferred_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
        };

        surface.configure(&device, &surface_config);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let layouts = Self::create_layouts(&device);
        let pipelines = Self::create_pipelines(&device, &surface_config, &layouts, &compilation);
        let (compute_group, copy_group) = Self::bind_for_size(&device, &sampler, size, &layouts);
        let start_time = Instant::now();
        State {
            pipelines,
            layouts,
            surface,
            device,
            queue,
            surface_config,
            size,
            sampler,
            compute_bind_group: compute_group,
            copy_bind_group: copy_group,
            start_time,
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width != 0 && new_size.height != 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.size = new_size;
            self.surface.configure(&self.device, &self.surface_config);

            let (compute_bind_group, copy_bind_group) =
                State::bind_for_size(&self.device, &self.sampler, new_size, &self.layouts);
            self.compute_bind_group = compute_bind_group;
            self.copy_bind_group = copy_bind_group;
        }
    }

    fn bind_for_size(
        device: &wgpu::Device,
        sampler: &wgpu::Sampler,
        size: PhysicalSize<u32>,
        layouts: &Layouts,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let img = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let img_view = img.create_view(&Default::default());
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layouts.compute_bg_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&img_view),
            }],
        });

        let copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layouts.copy_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&img_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        (compute_bind_group, copy_bind_group)
    }

    fn create_layouts(device: &wgpu::Device) -> Layouts {
        let copy_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        // Should filterable be false if we want nearest-neighbor?
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        filtering: false,
                        comparison: false,
                    },
                    count: None,
                },
            ],
        });
        let copy_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&copy_bg_layout],
            push_constant_ranges: &[],
        });
        let compute_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&compute_bg_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..std::mem::size_of::<Config>() as u32,
                }],
            });
        Layouts {
            compute_bg_layout,
            copy_bg_layout,
            compute_pipeline_layout,
            copy_pipeline_layout,
        }
    }

    fn replace_pipelines(&mut self, compilation: &CompileResult) {
        let new_pipelines = State::create_pipelines(
            &self.device,
            &self.surface_config,
            &self.layouts,
            compilation,
        );
        self.pipelines = new_pipelines;
    }

    fn create_pipelines(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        layouts: &Layouts,
        compilation: &CompileResult,
    ) -> Pipelines {
        let module = State::create_shader_module(device, compilation);
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layouts.compute_pipeline_layout),
            module: &module,
            entry_point: "main",
        });
        let copy_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layouts.copy_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_main",
                targets: &[surface_config.format.into()],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
        });
        Pipelines {
            compute_pipeline,
            copy_pipeline,
        }
    }

    fn create_shader_module(device: &wgpu::Device, compilation: &CompileResult) -> ShaderModule {
        let spirv_path = compilation.module.unwrap_single();
        let data = std::fs::read(spirv_path).expect("spirv should exist");
        let source = wgpu::util::make_spirv(&data);
        device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source,
        })
    }

    fn render_frame(&self) {
        let frame = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(err) => {
                eprintln!("get_current_texture error: {:?}", err);
                match err {
                    wgpu::SurfaceError::Lost => {
                        self.surface.configure(&self.device, &self.surface_config);
                    }
                    wgpu::SurfaceError::OutOfMemory => {
                        panic!();
                    }
                    _ => (),
                }
                return;
            }
        };
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let i_time: f32 = 0.5 + self.start_time.elapsed().as_micros() as f32 * 1e-6;
        let size = self.size;
        let config = Config {
            width: size.width,
            height: size.height,
            time: i_time,
        };

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&self.pipelines.compute_pipeline);
            cpass.set_push_constants(0, bytemuck::bytes_of(&config));
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            // Round up to next multiple
            cpass.dispatch((size.width + 16 - 1) / 16, (size.height + 16 - 1) / 16, 1);
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(&self.pipelines.copy_pipeline);
            rpass.set_bind_group(0, &self.copy_bind_group, &[]);
            rpass.draw(0..3, 0..2);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

async fn run(
    event_loop: EventLoop<CompileResult>,
    window: Window,
    initial_compilation: CompileResult,
) {
    let mut state = State::new(&window, initial_compilation).await;

    event_loop.run(move |event, _, control_flow| {
        // TODO: this may be excessive polling. It really should be synchronized with
        // swapchain presentation, but that's currently underbaked in wgpu.
        *control_flow = ControlFlow::Poll;
        match event {
            Event::RedrawRequested(_) => state.render_frame(),
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::UserEvent(result) => state.replace_pipelines(&result),
            Event::WindowEvent { window_id, event } if window_id == window.id() => match event {
                WindowEvent::Resized(new_size) => {
                    state.resize(new_size);
                    window.request_redraw();
                }
                _ => (),
            },
            _ => (),
        }
    });
}

fn main() {
    let event_loop = EventLoop::with_user_event();
    let proxy = event_loop.create_proxy();
    // Watch for changes on a background thread
    let initial_result = SpirvBuilder::new("./shaders", "spirv-unknown-spv1.5")
        .print_metadata(spirv_builder::MetadataPrintout::None)
        .watch(move |result| proxy.send_event(result).unwrap())
        .expect("Correctly setup");

    let window = Window::new(&event_loop).unwrap();
    pollster::block_on(run(event_loop, window, initial_result));
}
