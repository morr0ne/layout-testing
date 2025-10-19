use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod layout;
mod renderer;

use layout::Layout;
use renderer::Renderer;

const NUM_TEXTURES: u32 = 4;

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    layout: Option<Layout>,
    last_frame: Instant,
    next_window_id: u32,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            layout: None,
            last_frame: Instant::now(),
            next_window_id: 0,
        }
    }
}

impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let window =
            Arc::new(
                event_loop.create_window(Window::default_attributes().with_title(
                    "Niri Layout - Arrows: Navigate | W: Add | [/]: Resize | Q: Remove",
                ))?,
            );

        let size = window.inner_size();
        let mut renderer = pollster::block_on(Renderer::new(window.clone()))?;
        let layout = Layout::new(size.width as f32, size.height as f32);

        // Load textures
        for i in 0..NUM_TEXTURES {
            renderer.load_texture(&format!("windows/{}.jpg", i), i)?;
        }

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.layout = Some(layout);

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            if let Err(e) = self.init(event_loop) {
                eprintln!("Init failed: {}", e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(new_size);
                }
                if let Some(layout) = &mut self.layout {
                    layout.resize_viewport(new_size.width as f32, new_size.height as f32);
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                if let (Some(layout), Some(renderer)) = (&mut self.layout, &mut self.renderer) {
                    layout.update(dt);
                    let windows = layout.get_visible_windows();

                    let renderer_windows: Vec<renderer::LayoutWindow> = windows
                        .iter()
                        .map(|w| renderer::LayoutWindow {
                            x: w.x,
                            y: w.y,
                            width: w.width,
                            height: w.height,
                            texture_id: w.id % NUM_TEXTURES,
                            is_focused: w.is_focused,
                        })
                        .collect();

                    match renderer.render(&renderer_windows) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => eprintln!("Render error: {:?}", e),
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(keycode),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(layout) = &mut self.layout {
                    match keycode {
                        KeyCode::KeyW => {
                            layout.add_window(self.next_window_id);
                            self.next_window_id += 1;
                        }
                        KeyCode::KeyQ => layout.remove_focused_window(),
                        _ => layout.handle_key(keycode),
                    }
                }
            }

            _ => {}
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;

    Ok(())
}
