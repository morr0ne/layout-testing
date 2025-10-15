pub mod scrollable;
pub use scrollable::ScrollableLayout;

#[derive(Clone, Debug, Copy)]
pub struct LayoutRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Clone, Debug)]
pub struct LayoutWindow {
    pub rect: LayoutRect,
    pub id: u32,
    pub workspace_id: usize,
    pub is_focused: bool,
}

#[derive(Clone, Debug, Copy)]
pub struct Viewport {
    pub width: f32,
    pub height: f32,
}

// Trait for layout engines - provides window geometry
// This is the only swappable component - different layout algorithms can be plugged in
pub trait LayoutEngine {
    fn get_visible_windows(&self) -> Vec<LayoutWindow>;

    fn viewport(&self) -> Viewport;

    fn add_window(&mut self, window_id: u32);

    fn remove_focused_window(&mut self);

    fn resize_viewport(&mut self, width: f32, height: f32);

    // Per-frame update - allows layouts to animate, update state, etc.
    // Non-animated layouts can just do nothing
    fn update(&mut self, delta_time: f32);

    // Each layout defines its own keybindings
    fn handle_key(&mut self, key: winit::keyboard::KeyCode);
}
