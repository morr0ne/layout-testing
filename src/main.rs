use std::sync::Arc;
use std::time::Instant;
use tracing::{error, info};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Layout constants
const GAP_SIZE: f32 = 20.0;
const LERP_FACTOR: f32 = 20.0;
const SNAP_THRESHOLD: f32 = 0.5;
const BORDER_WIDTH: f32 = 4.0;
const BORDER_GAP_EXTEND: f32 = 10.0;

#[derive(Clone, Debug)]
struct ScrollableLayout {
    workspaces: Vec<Workspace>,
    current_workspace: usize,
    viewport: Viewport,
    scroll_offset_x: f32,
    scroll_offset_y: f32,
    target_scroll_x: f32,
    target_scroll_y: f32,
    workspace_height: f32,
    focused_window: Option<(usize, usize)>, // (workspace_idx, column_idx)
}

#[derive(Clone, Debug)]
struct Workspace {
    id: usize,
    columns: Vec<Column>,
    y_position: f32,
    focused_column: Option<usize>,
}

#[derive(Clone, Debug)]
struct Column {
    windows: Vec<WindowData>,
    x_position: f32,
    width: f32,
}

#[derive(Clone, Debug)]
struct WindowData {
    id: u32,
    height: f32,
}

#[derive(Clone, Debug, Copy)]
struct Viewport {
    width: f32,
    height: f32,
}

#[derive(Clone, Debug, Copy)]
struct RenderedRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    window_id: u32,
    workspace_id: usize,
    is_focused: bool,
}

impl ScrollableLayout {
    fn set_focus(&mut self, workspace_idx: usize, col_idx: usize) {
        if workspace_idx < self.workspaces.len() {
            let workspace = &mut self.workspaces[workspace_idx];
            if col_idx < workspace.columns.len() {
                workspace.focused_column = Some(col_idx);
                self.focused_window = Some((workspace_idx, col_idx));
            }
        }
    }

    fn new(viewport_width: f32, viewport_height: f32) -> Self {
        Self {
            workspaces: vec![Workspace {
                id: 0,
                columns: vec![],
                y_position: 0.0,
                focused_column: None,
            }],
            current_workspace: 0,
            viewport: Viewport {
                width: viewport_width,
                height: viewport_height,
            },
            scroll_offset_x: 0.0,
            scroll_offset_y: 0.0,
            target_scroll_x: 0.0,
            target_scroll_y: 0.0,
            workspace_height: viewport_height,
            focused_window: None,
        }
    }

    fn add_window(&mut self, window_id: u32) {
        let workspace = &mut self.workspaces[self.current_workspace];

        let x_position = if let Some(last_col) = workspace.columns.last() {
            last_col.x_position + last_col.width + GAP_SIZE
        } else {
            GAP_SIZE // Start with gap from left edge
        };

        // Default window width is 1/3 of viewport, accounting for gaps on all sides
        // With 3 windows: 4 gaps (left, between1, between2, right) = 4 * 20px
        let default_width = (self.viewport.width - 4.0 * GAP_SIZE) / 3.0;

        workspace.columns.push(Column {
            windows: vec![WindowData {
                id: window_id,
                height: self.viewport.height - 2.0 * GAP_SIZE, // Gap on top and bottom
            }],
            x_position,
            width: default_width,
        });

        // Auto-focus the new window and scroll to it
        let new_col_idx = workspace.columns.len() - 1;
        self.set_focus(self.current_workspace, new_col_idx);
        self.scroll_to_focused_window();
    }

    fn add_workspace(&mut self) {
        let new_id = self.workspaces.len();
        let y_position = new_id as f32 * self.workspace_height;
        self.workspaces.push(Workspace {
            id: new_id,
            columns: vec![],
            y_position,
            focused_column: None,
        });
    }

    fn scroll_horizontal(&mut self, delta: f32) {
        let workspace = &mut self.workspaces[self.current_workspace];

        if workspace.columns.is_empty() {
            return;
        }

        let new_col_idx = match (delta < 0.0, workspace.focused_column) {
            // Scrolling left with focus - go to previous column
            (true, Some(col_idx)) if col_idx > 0 => Some(col_idx - 1),
            // Scrolling right with focus - go to next column
            (false, Some(col_idx)) if col_idx + 1 < workspace.columns.len() => Some(col_idx + 1),
            // No focus - default to first column
            (_, None) => Some(0),
            // Can't move in desired direction
            _ => return,
        };

        if let Some(col_idx) = new_col_idx {
            self.set_focus(self.current_workspace, col_idx);
            self.scroll_to_focused_window();
        }
    }

    fn scroll_vertical(&mut self, delta: f32) {
        self.target_scroll_y += delta;
        let workspace_idx = (self.target_scroll_y / self.workspace_height).round() as i32;
        let clamped_idx = workspace_idx.clamp(0, self.workspaces.len() as i32 - 1);
        self.target_scroll_y = clamped_idx as f32 * self.workspace_height;
        self.current_workspace = clamped_idx as usize;

        // Restore the workspace's remembered focused window
        let workspace = &self.workspaces[self.current_workspace];
        if let Some(col_idx) = workspace.focused_column {
            if col_idx < workspace.columns.len() {
                self.set_focus(self.current_workspace, col_idx);
                self.scroll_to_focused_window();
            } else {
                self.focused_window = None;
            }
        } else if !workspace.columns.is_empty() {
            // No remembered focus, focus first window
            self.set_focus(self.current_workspace, 0);
            self.scroll_to_focused_window();
        } else {
            self.focused_window = None;
        }
    }

    fn update_animation(&mut self, delta_time: f32) {
        // We are lerping chat
        // https://en.wikipedia.org/wiki/Linear_interpolation
        let lerp_factor = LERP_FACTOR * delta_time;
        self.scroll_offset_x += (self.target_scroll_x - self.scroll_offset_x) * lerp_factor;
        self.scroll_offset_y += (self.target_scroll_y - self.scroll_offset_y) * lerp_factor;

        // Snap to target when very close to avoid weird driftting
        if (self.target_scroll_x - self.scroll_offset_x).abs() < SNAP_THRESHOLD {
            self.scroll_offset_x = self.target_scroll_x;
        }
        if (self.target_scroll_y - self.scroll_offset_y).abs() < SNAP_THRESHOLD {
            self.scroll_offset_y = self.target_scroll_y;
        }
    }

    fn get_visible_rects(&self) -> Vec<RenderedRect> {
        let mut rects = Vec::new();
        let viewport_top = self.scroll_offset_y;
        let viewport_bottom = self.scroll_offset_y + self.viewport.height;

        for (ws_idx, workspace) in self.workspaces.iter().enumerate() {
            let ws_top = workspace.y_position;
            let ws_bottom = workspace.y_position + self.workspace_height;
            if ws_bottom < viewport_top || ws_top > viewport_bottom {
                continue;
            }

            let viewport_left = self.scroll_offset_x;
            let viewport_right = self.scroll_offset_x + self.viewport.width;

            for (col_idx, column) in workspace.columns.iter().enumerate() {
                let col_left = column.x_position;
                let col_right = column.x_position + column.width;
                if col_right < viewport_left || col_left > viewport_right {
                    continue;
                }

                // Check if this column is focused
                let is_focused = if let Some((focused_ws, focused_col)) = self.focused_window {
                    focused_ws == ws_idx && focused_col == col_idx
                } else {
                    false
                };

                let mut y_offset = GAP_SIZE; // Start with top gap
                for window in &column.windows {
                    rects.push(RenderedRect {
                        x: column.x_position - self.scroll_offset_x,
                        y: workspace.y_position + y_offset - self.scroll_offset_y,
                        width: column.width,
                        height: window.height,
                        window_id: window.id,
                        workspace_id: workspace.id,
                        is_focused,
                    });
                    y_offset += window.height + GAP_SIZE;
                }
            }
        }
        rects
    }

    fn resize_viewport(&mut self, width: f32, height: f32) {
        let old_width = self.viewport.width;
        self.viewport.width = width;
        self.viewport.height = height;

        // Update workspace height to match viewport
        self.workspace_height = height;

        // Update all workspace y positions
        for (idx, workspace) in self.workspaces.iter_mut().enumerate() {
            workspace.y_position = idx as f32 * self.workspace_height;
        }

        // Update window heights in all columns (accounting for top and bottom gaps)
        for workspace in &mut self.workspaces {
            for column in &mut workspace.columns {
                for window in &mut column.windows {
                    window.height = height - 2.0 * GAP_SIZE;
                }
            }
        }

        // Proportionally scale all window widths
        let scale_factor = width / old_width;
        for workspace in &mut self.workspaces {
            for column in &mut workspace.columns {
                column.width *= scale_factor;
            }
            // Also scale column positions
            for column in &mut workspace.columns {
                column.x_position *= scale_factor;
            }
        }

        // Scale scroll positions
        self.scroll_offset_x *= scale_factor;
        self.target_scroll_x *= scale_factor;

        // Recalculate vertical scroll to match new workspace heights
        self.scroll_offset_y = self.current_workspace as f32 * self.workspace_height;
        self.target_scroll_y = self.current_workspace as f32 * self.workspace_height;
    }

    fn resize_focused_window(&mut self, delta: f32) {
        if let Some((ws_idx, col_idx)) = self.focused_window {
            if ws_idx < self.workspaces.len() {
                let workspace = &mut self.workspaces[ws_idx];
                if col_idx < workspace.columns.len() {
                    let column = &mut workspace.columns[col_idx];

                    let min_width = (self.viewport.width - 7.0 * GAP_SIZE) / 6.0;
                    let max_width = self.viewport.width - 2.0 * GAP_SIZE;

                    column.width = (column.width + delta).clamp(min_width, max_width);

                    // Reposition all columns after this one
                    for i in (col_idx + 1)..workspace.columns.len() {
                        let prev_col_right = if i > 0 {
                            workspace.columns[i - 1].x_position
                                + workspace.columns[i - 1].width
                                + GAP_SIZE
                        } else {
                            GAP_SIZE
                        };
                        workspace.columns[i].x_position = prev_col_right;
                    }
                }
            }
        }
    }

    fn remove_focused_window(&mut self) {
        if let Some((ws_idx, col_idx)) = self.focused_window {
            if ws_idx < self.workspaces.len() {
                let workspace = &mut self.workspaces[ws_idx];
                if col_idx < workspace.columns.len() {
                    workspace.columns.remove(col_idx);

                    // Reposition all remaining columns
                    let mut x_pos = GAP_SIZE; // Start with left gap
                    for column in &mut workspace.columns {
                        column.x_position = x_pos;
                        x_pos += column.width + GAP_SIZE;
                    }

                    // Calculate the total width of all windows
                    let total_width = if let Some(last_col) = workspace.columns.last() {
                        last_col.x_position + last_col.width + GAP_SIZE
                    } else {
                        0.0
                    };

                    // Adjust scroll to fill viewport if there's empty space
                    if total_width > self.viewport.width {
                        // More content than viewport; ensure we don't show empty space on the right
                        let max_scroll = total_width - self.viewport.width;
                        if self.target_scroll_x > max_scroll {
                            self.target_scroll_x = max_scroll;
                        }
                    } else {
                        // All content fits in viewport - scroll to start
                        self.target_scroll_x = 0.0;
                    }

                    // Update focus to next window (right) or previous (left) if no windows left
                    if workspace.columns.is_empty() {
                        workspace.focused_column = None;
                        self.focused_window = None;
                    } else if col_idx < workspace.columns.len() {
                        // Focus the window that took the closed window's position (was on the right)
                        self.set_focus(ws_idx, col_idx);
                    } else if col_idx > 0 {
                        // Closed the last window, focus the new last window
                        self.set_focus(ws_idx, col_idx - 1);
                    }
                }
            }
        }
    }

    fn scroll_to_focused_window(&mut self) {
        if let Some((ws_idx, col_idx)) = self.focused_window {
            if ws_idx < self.workspaces.len() {
                // Scroll vertically to the workspace
                self.current_workspace = ws_idx;
                self.target_scroll_y = ws_idx as f32 * self.workspace_height;

                // Scroll horizontally to the window (accounting for left gap)
                let workspace = &self.workspaces[ws_idx];
                if col_idx < workspace.columns.len() {
                    let column = &workspace.columns[col_idx];
                    // Scroll to show the window with the left gap visible
                    self.target_scroll_x = (column.x_position - GAP_SIZE).max(0.0);

                    // Clamp to valid range
                    let max_scroll = if let Some(last_col) = workspace.columns.last() {
                        (last_col.x_position + last_col.width + GAP_SIZE - self.viewport.width)
                            .max(0.0)
                    } else {
                        0.0
                    };
                    self.target_scroll_x = self.target_scroll_x.clamp(0.0, max_scroll);
                }
            }
        }
    }
}

/*

Stuff below is mostly related to the renderer

*/

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

struct WgpuState<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

impl<'window> WgpuState<'window> {
    async fn new(window: Arc<Window>) -> WgpuState<'window> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

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
            .unwrap();

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
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
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

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: 65536,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: 65536,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices: 0,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn update_layout(&mut self, rects: &[RenderedRect], viewport: Viewport) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut vertex_count = 0u16;

        for rect in rects.iter() {
            let x1 = (rect.x / viewport.width) * 2.0 - 1.0;
            let y1 = -((rect.y / viewport.height) * 2.0 - 1.0);
            let x2 = ((rect.x + rect.width) / viewport.width) * 2.0 - 1.0;
            let y2 = -(((rect.y + rect.height) / viewport.height) * 2.0 - 1.0);

            let hue = ((rect.window_id as f32 * 0.618) + (rect.workspace_id as f32 * 0.1)) % 1.0;
            let color = hsv_to_rgb(hue, 0.6, 0.9);

            // Draw main window
            let base_index = vertex_count;
            vertices.extend_from_slice(&[
                Vertex {
                    position: [x1, y1],
                    color,
                },
                Vertex {
                    position: [x2, y1],
                    color,
                },
                Vertex {
                    position: [x2, y2],
                    color,
                },
                Vertex {
                    position: [x1, y2],
                    color,
                },
            ]);
            indices.extend_from_slice(&[
                base_index,
                base_index + 1,
                base_index + 2,
                base_index,
                base_index + 2,
                base_index + 3,
            ]);
            vertex_count += 4;

            // Draw border if focused, the border "spills" into the gaps so we don't need to recalculate things
            if rect.is_focused {
                let border_width_ndc_x = (BORDER_WIDTH / viewport.width) * 2.0;
                let border_width_ndc_y = (BORDER_WIDTH / viewport.height) * 2.0;

                // Spill border into gaps
                let gap_extend_ndc_x = (BORDER_GAP_EXTEND / viewport.width) * 2.0;
                let gap_extend_ndc_y = (BORDER_GAP_EXTEND / viewport.height) * 2.0;

                let outer_x1 = x1 - gap_extend_ndc_x;
                let outer_y1 = y1 + gap_extend_ndc_y;
                let outer_x2 = x2 + gap_extend_ndc_x;
                let outer_y2 = y2 - gap_extend_ndc_y;

                let inner_x1 = outer_x1 + border_width_ndc_x;
                let inner_y1 = outer_y1 - border_width_ndc_y;
                let inner_x2 = outer_x2 - border_width_ndc_x;
                let inner_y2 = outer_y2 + border_width_ndc_y;

                let border_color = [1.0, 1.0, 1.0]; // White border

                // Top border
                let base = vertex_count;
                vertices.extend_from_slice(&[
                    Vertex {
                        position: [outer_x1, outer_y1],
                        color: border_color,
                    },
                    Vertex {
                        position: [outer_x2, outer_y1],
                        color: border_color,
                    },
                    Vertex {
                        position: [inner_x2, inner_y1],
                        color: border_color,
                    },
                    Vertex {
                        position: [inner_x1, inner_y1],
                        color: border_color,
                    },
                ]);
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                vertex_count += 4;

                // Bottom border
                let base = vertex_count;
                vertices.extend_from_slice(&[
                    Vertex {
                        position: [inner_x1, inner_y2],
                        color: border_color,
                    },
                    Vertex {
                        position: [inner_x2, inner_y2],
                        color: border_color,
                    },
                    Vertex {
                        position: [outer_x2, outer_y2],
                        color: border_color,
                    },
                    Vertex {
                        position: [outer_x1, outer_y2],
                        color: border_color,
                    },
                ]);
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                vertex_count += 4;

                // Left border
                let base = vertex_count;
                vertices.extend_from_slice(&[
                    Vertex {
                        position: [outer_x1, outer_y1],
                        color: border_color,
                    },
                    Vertex {
                        position: [inner_x1, inner_y1],
                        color: border_color,
                    },
                    Vertex {
                        position: [inner_x1, inner_y2],
                        color: border_color,
                    },
                    Vertex {
                        position: [outer_x1, outer_y2],
                        color: border_color,
                    },
                ]);
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                vertex_count += 4;

                // Right border
                let base = vertex_count;
                vertices.extend_from_slice(&[
                    Vertex {
                        position: [inner_x2, inner_y1],
                        color: border_color,
                    },
                    Vertex {
                        position: [outer_x2, outer_y1],
                        color: border_color,
                    },
                    Vertex {
                        position: [outer_x2, outer_y2],
                        color: border_color,
                    },
                    Vertex {
                        position: [inner_x2, inner_y2],
                        color: border_color,
                    },
                ]);
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                vertex_count += 4;
            }
        }

        if !vertices.is_empty() {
            self.queue
                .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
            self.queue
                .write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
            self.num_indices = indices.len() as u32;
        } else {
            self.num_indices = 0;
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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

            if self.num_indices > 0 {
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
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

/*

This is basically the compositor but in this case it's a winit app. Obv this needs to be integrated in verdi as separate bits

*/

struct App<'window> {
    window: Option<Arc<Window>>,
    state: Option<WgpuState<'window>>,
    layout: ScrollableLayout,
    last_frame: Instant,
    next_window_id: u32,
}

impl<'window> Default for App<'window> {
    fn default() -> Self {
        Self {
            window: None,
            state: None,
            layout: ScrollableLayout::new(1920.0, 1080.0),
            last_frame: Instant::now(),
            next_window_id: 0,
        }
    }
}

impl<'window> ApplicationHandler for App<'window> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attrs = Window::default_attributes().with_title(
                "Niri Layout - Arrows: Scroll | W: Add Window | [/]: Resize | Q: Remove",
            );
            let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

            let size = window.inner_size();
            self.layout
                .resize_viewport(size.width as f32, size.height as f32);

            let state = pollster::block_on(WgpuState::new(window.clone()));

            self.window = Some(window);
            self.state = Some(state);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(state) = &mut self.state {
                    state.resize(physical_size);
                    self.layout
                        .resize_viewport(physical_size.width as f32, physical_size.height as f32);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let delta_time = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                self.layout.update_animation(delta_time);

                if let Some(state) = &mut self.state {
                    let rects = self.layout.get_visible_rects();
                    state.update_layout(&rects, self.layout.viewport);

                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => error!("Render error: {:?}", e),
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                match key {
                    // Scroll horizontally
                    KeyCode::ArrowLeft => self.layout.scroll_horizontal(-200.0),
                    KeyCode::ArrowRight => self.layout.scroll_horizontal(200.0),

                    // Scroll vertically (between workspaces)
                    KeyCode::ArrowUp => self.layout.scroll_vertical(-self.layout.workspace_height),
                    KeyCode::ArrowDown => {
                        self.layout.scroll_vertical(self.layout.workspace_height);
                        // Auto-create workspace if we're at the end
                        if self.layout.current_workspace >= self.layout.workspaces.len() - 1 {
                            self.layout.add_workspace();
                        }
                    }

                    // Add window in current workspace
                    KeyCode::KeyW => {
                        self.layout.add_window(self.next_window_id);
                        self.next_window_id += 1;
                        info!(
                            "Added window {} to workspace {}",
                            self.next_window_id - 1,
                            self.layout.current_workspace
                        );
                    }

                    // Resize focused window
                    KeyCode::BracketLeft => {
                        self.layout.resize_focused_window(-50.0);
                    }
                    KeyCode::BracketRight => {
                        self.layout.resize_focused_window(50.0);
                    }

                    // Close focused window
                    KeyCode::KeyQ => {
                        self.layout.remove_focused_window();
                        info!("Removed focused window");
                    }

                    _ => {}
                }
            }
            _ => {}
        }
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
