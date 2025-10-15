use crate::layouts::{LayoutWindow, Viewport};
use anyhow::{Context, Result};
use std::sync::Arc;
use winit::window::Window;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

pub struct WgpuState<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count_to_render: u32,
    vertex_buffer_capacity: wgpu::BufferAddress,
    index_buffer_capacity: wgpu::BufferAddress,
    // Rendering configuration
    border_width_pixels: f32,
    border_gap_extension_pixels: f32,
}

impl<'window> WgpuState<'window> {
    pub async fn new(window: Arc<Window>) -> Result<WgpuState<'window>> {
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
            .context("Failed to find suitable GPU adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .context("Failed to request GPU device")?;

        let surface_caps = surface.get_capabilities(&adapter);

        // Choose surface format - prefer sRGB for correct color display
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .or_else(|| surface_caps.formats.first().copied())
            .context("No supported surface formats available")?;

        let alpha_mode = surface_caps
            .alpha_modes
            .first()
            .copied()
            .context("No supported alpha modes available")?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    out.color = model.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#
                .into(),
            ),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Initial buffer sizes - calculated to handle a reasonable number of windows
        // Each window requires 4 vertices (quad) + potentially 16 more for borders (4 quads)
        // = 20 vertices max per window. 100 windows = 2000 vertices comfortably.
        const INITIAL_VERTEX_COUNT: usize = 2048;
        const INITIAL_INDEX_COUNT: usize = INITIAL_VERTEX_COUNT * 2; // Approximately 1.5x vertices in practice

        let vertex_buffer_capacity =
            (INITIAL_VERTEX_COUNT * std::mem::size_of::<Vertex>()) as wgpu::BufferAddress;
        let index_buffer_capacity =
            (INITIAL_INDEX_COUNT * std::mem::size_of::<u16>()) as wgpu::BufferAddress;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: vertex_buffer_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: index_buffer_capacity,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            index_count_to_render: 0,
            vertex_buffer_capacity,
            index_buffer_capacity,
            border_width_pixels: 4.0,
            border_gap_extension_pixels: 10.0,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn render_windows(&mut self, windows: &[LayoutWindow], viewport: Viewport) {
        // Pre-allocate with capacity hint for better performance
        // Each window = 4 vertices, potentially 16 more for borders
        let estimated_vertex_count = windows.len() * 20;
        let estimated_index_count = estimated_vertex_count * 2;

        let mut vertices = Vec::with_capacity(estimated_vertex_count);
        let mut indices = Vec::with_capacity(estimated_index_count);
        let mut next_vertex_index = 0u16;

        for window in windows.iter() {
            // Window coordinates are already viewport-relative from get_visible_windows()
            // Convert directly to NDC (Normalized Device Coordinates: -1.0 to 1.0)
            let ndc_left = (window.rect.x / viewport.width) * 2.0 - 1.0;
            let ndc_top = -((window.rect.y / viewport.height) * 2.0 - 1.0);
            let ndc_right = ((window.rect.x + window.rect.width) / viewport.width) * 2.0 - 1.0;
            let ndc_bottom =
                -(((window.rect.y + window.rect.height) / viewport.height) * 2.0 - 1.0);

            let window_color = self.compute_window_color(window.id, window.workspace_id);

            // Draw main window rectangle as 2 triangles (quad)
            let quad_base_index = next_vertex_index;
            vertices.extend_from_slice(&[
                Vertex {
                    position: [ndc_left, ndc_top],
                    color: window_color,
                },
                Vertex {
                    position: [ndc_right, ndc_top],
                    color: window_color,
                },
                Vertex {
                    position: [ndc_right, ndc_bottom],
                    color: window_color,
                },
                Vertex {
                    position: [ndc_left, ndc_bottom],
                    color: window_color,
                },
            ]);
            indices.extend_from_slice(&[
                quad_base_index,
                quad_base_index + 1,
                quad_base_index + 2,
                quad_base_index,
                quad_base_index + 2,
                quad_base_index + 3,
            ]);
            next_vertex_index += 4;

            // Draw border if focused
            if window.is_focused {
                next_vertex_index = self.render_border(
                    ndc_left,
                    ndc_top,
                    ndc_right,
                    ndc_bottom,
                    viewport,
                    &mut vertices,
                    &mut indices,
                    next_vertex_index,
                );
            }
        }

        if !vertices.is_empty() {
            let vertex_data = bytemuck::cast_slice(&vertices);
            let index_data = bytemuck::cast_slice(&indices);

            // Check if we need to resize buffers (they grew beyond capacity)
            if vertex_data.len() as wgpu::BufferAddress > self.vertex_buffer_capacity {
                let new_capacity = (vertex_data.len() as wgpu::BufferAddress).next_power_of_two();
                self.vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Vertex Buffer (Resized)"),
                    size: new_capacity,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.vertex_buffer_capacity = new_capacity;
                tracing::warn!(
                    "Vertex buffer resized to {} bytes ({} vertices)",
                    new_capacity,
                    new_capacity / std::mem::size_of::<Vertex>() as u64
                );
            }

            if index_data.len() as wgpu::BufferAddress > self.index_buffer_capacity {
                let new_capacity = (index_data.len() as wgpu::BufferAddress).next_power_of_two();
                self.index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Index Buffer (Resized)"),
                    size: new_capacity,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.index_buffer_capacity = new_capacity;
                tracing::warn!(
                    "Index buffer resized to {} bytes ({} indices)",
                    new_capacity,
                    new_capacity / std::mem::size_of::<u16>() as u64
                );
            }

            // Write buffer data - write_buffer is the recommended approach for dynamic updates
            self.queue.write_buffer(&self.vertex_buffer, 0, vertex_data);
            self.queue.write_buffer(&self.index_buffer, 0, index_data);
            self.index_count_to_render = indices.len() as u32;
        } else {
            self.index_count_to_render = 0;
        }
    }

    fn compute_window_color(&self, window_id: u32, workspace_id: usize) -> [f32; 3] {
        let hue = ((window_id as f32 * 0.618) + (workspace_id as f32 * 0.1)) % 1.0;
        hsv_to_rgb(hue, 0.6, 0.9)
    }

    /// Renders a border around a window by adding 4 rectangular strips (top, bottom, left, right)
    /// The border extends into the gaps between windows for visual continuity
    /// Returns the updated next_vertex_index
    fn render_border(
        &self,
        ndc_left: f32,
        ndc_top: f32,
        ndc_right: f32,
        ndc_bottom: f32,
        viewport: Viewport,
        vertices: &mut Vec<Vertex>,
        indices: &mut Vec<u16>,
        mut next_vertex_index: u16,
    ) -> u16 {
        // Convert pixel measurements to NDC space
        let border_width_ndc_horizontal = (self.border_width_pixels / viewport.width) * 2.0;
        let border_width_ndc_vertical = (self.border_width_pixels / viewport.height) * 2.0;
        let gap_extension_ndc_horizontal =
            (self.border_gap_extension_pixels / viewport.width) * 2.0;
        let gap_extension_ndc_vertical = (self.border_gap_extension_pixels / viewport.height) * 2.0;

        // Outer edge of border (extends into gaps)
        let border_outer_left = ndc_left - gap_extension_ndc_horizontal;
        let border_outer_top = ndc_top + gap_extension_ndc_vertical;
        let border_outer_right = ndc_right + gap_extension_ndc_horizontal;
        let border_outer_bottom = ndc_bottom - gap_extension_ndc_vertical;

        // Inner edge of border (at window boundary minus border width)
        let border_inner_left = border_outer_left + border_width_ndc_horizontal;
        let border_inner_top = border_outer_top - border_width_ndc_vertical;
        let border_inner_right = border_outer_right - border_width_ndc_horizontal;
        let border_inner_bottom = border_outer_bottom + border_width_ndc_vertical;

        let border_color = [1.0, 1.0, 1.0]; // White border

        // Each border edge is a quad defined by 4 corners (outer and inner edges)
        let border_edge_quads = [
            // Top border strip
            [
                [border_outer_left, border_outer_top],
                [border_outer_right, border_outer_top],
                [border_inner_right, border_inner_top],
                [border_inner_left, border_inner_top],
            ],
            // Bottom border strip
            [
                [border_inner_left, border_inner_bottom],
                [border_inner_right, border_inner_bottom],
                [border_outer_right, border_outer_bottom],
                [border_outer_left, border_outer_bottom],
            ],
            // Left border strip
            [
                [border_outer_left, border_outer_top],
                [border_inner_left, border_inner_top],
                [border_inner_left, border_inner_bottom],
                [border_outer_left, border_outer_bottom],
            ],
            // Right border strip
            [
                [border_inner_right, border_inner_top],
                [border_outer_right, border_outer_top],
                [border_outer_right, border_outer_bottom],
                [border_inner_right, border_inner_bottom],
            ],
        ];

        for quad_corners in &border_edge_quads {
            let quad_base_index = next_vertex_index;
            for &corner_position in quad_corners {
                vertices.push(Vertex {
                    position: corner_position,
                    color: border_color,
                });
            }
            indices.extend_from_slice(&[
                quad_base_index,
                quad_base_index + 1,
                quad_base_index + 2,
                quad_base_index,
                quad_base_index + 2,
                quad_base_index + 3,
            ]);
            next_vertex_index += 4;
        }

        next_vertex_index
    }

    pub fn present(&mut self) -> Result<(), wgpu::SurfaceError> {
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
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

            if self.index_count_to_render > 0 {
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.index_count_to_render, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h * 6.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    [r + m, g + m, b + m]
}
