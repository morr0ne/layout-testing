use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use winit::window::Window;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TexturedVertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ColoredVertex {
    position: [f32; 2],
    color: [f32; 3],
}

pub struct LayoutWindow {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub texture_id: u32,
    pub is_focused: bool,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    textured_pipeline: wgpu::RenderPipeline,
    colored_pipeline: wgpu::RenderPipeline,
    textures: HashMap<u32, WindowTexture>,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    viewport_width: f32,
    viewport_height: f32,
}

struct WindowTexture {
    _texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .context("Failed to create surface")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("Failed to find adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .context("Failed to create device")?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/windows.wgsl"
            ))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let textured_pipeline =
            Self::create_textured_pipeline(&device, &shader, &bind_group_layout, surface_format);
        let colored_pipeline = Self::create_colored_pipeline(&device, &shader, surface_format);

        Ok(Self {
            device,
            queue,
            surface,
            config,
            size,
            textured_pipeline,
            colored_pipeline,
            textures: HashMap::new(),
            bind_group_layout,
            sampler,
            viewport_width: size.width as f32,
            viewport_height: size.height as f32,
        })
    }

    fn create_textured_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        bind_group_layout: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Textured Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Textured Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_textured"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_textured"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn create_colored_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Colored Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Colored Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_colored"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ColoredVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_colored"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.viewport_width = new_size.width as f32;
            self.viewport_height = new_size.height as f32;
        }
    }

    pub fn load_texture(&mut self, path: &str, texture_id: u32) -> Result<()> {
        let img = image::open(path)?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Texture {}", texture_id)),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            texture.as_image_copy(),
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Bind Group {}", texture_id)),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.textures.insert(
            texture_id,
            WindowTexture {
                _texture: texture,
                bind_group,
            },
        );
        Ok(())
    }

    pub fn render(&mut self, windows: &[LayoutWindow]) -> Result<(), wgpu::SurfaceError> {
        let mut textured_vertices = Vec::new();
        let mut textured_indices = Vec::new();
        let mut colored_vertices = Vec::new();
        let mut colored_indices = Vec::new();

        // Build geometry
        for window in windows {
            let (left, top, right, bottom) =
                self.to_ndc(window.x, window.y, window.width, window.height);

            // Add window quad
            let base = textured_vertices.len() as u16;
            textured_vertices.extend_from_slice(&[
                TexturedVertex {
                    position: [left, top],
                    tex_coords: [0.0, 0.0],
                },
                TexturedVertex {
                    position: [right, top],
                    tex_coords: [1.0, 0.0],
                },
                TexturedVertex {
                    position: [right, bottom],
                    tex_coords: [1.0, 1.0],
                },
                TexturedVertex {
                    position: [left, bottom],
                    tex_coords: [0.0, 1.0],
                },
            ]);
            textured_indices.extend_from_slice(&[
                base,
                base + 1,
                base + 2,
                base,
                base + 2,
                base + 3,
            ]);

            // Add border if focused
            if window.is_focused {
                self.add_border(
                    left,
                    top,
                    right,
                    bottom,
                    &mut colored_vertices,
                    &mut colored_indices,
                );
            }
        }

        // Render
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Draw windows
            if !textured_indices.is_empty() {
                let textured_vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Textured Vertex Buffer"),
                    size: (textured_vertices.len() * std::mem::size_of::<TexturedVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.queue.write_buffer(
                    &textured_vertex_buffer,
                    0,
                    bytemuck::cast_slice(&textured_vertices),
                );

                let textured_index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Textured Index Buffer"),
                    size: (textured_indices.len() * std::mem::size_of::<u16>()) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.queue.write_buffer(
                    &textured_index_buffer,
                    0,
                    bytemuck::cast_slice(&textured_indices),
                );

                pass.set_pipeline(&self.textured_pipeline);
                pass.set_vertex_buffer(0, textured_vertex_buffer.slice(..));
                pass.set_index_buffer(textured_index_buffer.slice(..), wgpu::IndexFormat::Uint16);

                for (i, window) in windows.iter().enumerate() {
                    if let Some(texture) = self.textures.get(&window.texture_id) {
                        pass.set_bind_group(0, &texture.bind_group, &[]);
                        let start = (i * 6) as u32;
                        pass.draw_indexed(start..start + 6, 0, 0..1);
                    }
                }
            }

            // Draw borders
            if !colored_indices.is_empty() {
                let colored_vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Colored Vertex Buffer"),
                    size: (colored_vertices.len() * std::mem::size_of::<ColoredVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.queue.write_buffer(
                    &colored_vertex_buffer,
                    0,
                    bytemuck::cast_slice(&colored_vertices),
                );

                let colored_index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Colored Index Buffer"),
                    size: (colored_indices.len() * std::mem::size_of::<u16>()) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.queue.write_buffer(
                    &colored_index_buffer,
                    0,
                    bytemuck::cast_slice(&colored_indices),
                );

                pass.set_pipeline(&self.colored_pipeline);
                pass.set_vertex_buffer(0, colored_vertex_buffer.slice(..));
                pass.set_index_buffer(colored_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..colored_indices.len() as u32, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn to_ndc(&self, x: f32, y: f32, width: f32, height: f32) -> (f32, f32, f32, f32) {
        let left = (x / self.viewport_width) * 2.0 - 1.0;
        let top = -((y / self.viewport_height) * 2.0 - 1.0);
        let right = ((x + width) / self.viewport_width) * 2.0 - 1.0;
        let bottom = -(((y + height) / self.viewport_height) * 2.0 - 1.0);
        (left, top, right, bottom)
    }

    fn add_border(
        &self,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
        vertices: &mut Vec<ColoredVertex>,
        indices: &mut Vec<u16>,
    ) {
        let border_width = 4.0;
        let gap_ext = 10.0;

        // Convert to NDC
        let bw_h = (border_width / self.viewport_width) * 2.0;
        let bw_v = (border_width / self.viewport_height) * 2.0;
        let ge_h = (gap_ext / self.viewport_width) * 2.0;
        let ge_v = (gap_ext / self.viewport_height) * 2.0;

        let ol = left - ge_h;
        let ot = top + ge_v;
        let or = right + ge_h;
        let ob = bottom - ge_v;
        let il = ol + bw_h;
        let it = ot - bw_v;
        let ir = or - bw_h;
        let ib = ob + bw_v;

        let color = [1.0, 1.0, 1.0];

        // 4 border quads: top, bottom, left, right
        for corners in [
            [[ol, ot], [or, ot], [ir, it], [il, it]], // Top
            [[il, ib], [ir, ib], [or, ob], [ol, ob]], // Bottom
            [[ol, ot], [il, it], [il, ib], [ol, ob]], // Left
            [[ir, it], [or, ot], [or, ob], [ir, ib]], // Right
        ] {
            let base = vertices.len() as u16;
            for pos in corners {
                vertices.push(ColoredVertex {
                    position: pos,
                    color,
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
}
