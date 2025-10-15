use anyhow::{Context, Result};
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

mod layouts;
mod renderer;

use layouts::{LayoutEngine, ScrollableLayout};
use renderer::WgpuState;

/*

This is basically the compositor but in this case it's a winit app. Obv this needs to be integrated in verdi as separate bits

*/

struct App<'window> {
    window: Option<Arc<Window>>,
    renderer_state: Option<WgpuState<'window>>,
    layout_engine: Box<dyn LayoutEngine>,
    last_frame_time: Instant,
    next_window_id_counter: u32,
}

impl<'window> Default for App<'window> {
    fn default() -> Self {
        Self {
            window: None,
            renderer_state: None,
            layout_engine: Box::new(ScrollableLayout::new(1920.0, 1080.0)),
            last_frame_time: Instant::now(),
            next_window_id_counter: 0,
        }
    }
}

impl<'window> App<'window> {
    fn init_window_and_renderer(
        &mut self,
        event_loop: &ActiveEventLoop,
    ) -> Result<(Arc<Window>, WgpuState<'window>)> {
        let window_attributes = Window::default_attributes()
            .with_title("Niri Layout - Arrows: Scroll | W: Add Window | [/]: Resize | Q: Remove");
        let window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .context("Failed to create window")?,
        );

        let window_size = window.inner_size();
        self.layout_engine
            .resize_viewport(window_size.width as f32, window_size.height as f32);

        let renderer_state = pollster::block_on(WgpuState::new(window.clone()))?;

        Ok((window, renderer_state))
    }
}

impl<'window> ApplicationHandler for App<'window> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            match self.init_window_and_renderer(event_loop) {
                Ok((window, renderer_state)) => {
                    self.window = Some(window);
                    self.renderer_state = Some(renderer_state);
                }
                Err(error) => {
                    error!("Failed to initialize window and renderer: {}", error);
                    event_loop.exit();
                }
            }
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
            WindowEvent::Resized(new_physical_size) => {
                if let Some(renderer) = &mut self.renderer_state {
                    renderer.resize(new_physical_size);
                    self.layout_engine.resize_viewport(
                        new_physical_size.width as f32,
                        new_physical_size.height as f32,
                    );
                }
            }
            WindowEvent::RedrawRequested => {
                let current_time = Instant::now();
                let frame_delta_time = (current_time - self.last_frame_time).as_secs_f32();
                self.last_frame_time = current_time;

                // Update layout animations
                self.layout_engine.update(frame_delta_time);

                // Render the frame
                if let Some(renderer) = &mut self.renderer_state {
                    let visible_windows = self.layout_engine.get_visible_windows();
                    let viewport = self.layout_engine.viewport();
                    renderer.render_windows(&visible_windows, viewport);

                    match renderer.present() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(render_error) => error!("Render error: {:?}", render_error),
                    }
                }

                // Request the next frame
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
                match keycode {
                    // Add new window to current workspace
                    KeyCode::KeyW => {
                        let new_window_id = self.next_window_id_counter;
                        self.layout_engine.add_window(new_window_id);
                        info!("Added window with ID: {}", new_window_id);
                        self.next_window_id_counter += 1;
                    }

                    // Remove currently focused window
                    KeyCode::KeyQ => {
                        self.layout_engine.remove_focused_window();
                        info!("Removed focused window");
                    }

                    // All other keys are handled by the layout engine
                    _ => self.layout_engine.handle_key(keycode),
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let event_loop = EventLoop::new().context("Failed to create event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .context("Event loop execution failed")?;

    Ok(())
}
