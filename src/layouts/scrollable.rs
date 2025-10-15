use super::{LayoutEngine, LayoutRect, LayoutWindow, Viewport};
use winit::keyboard::KeyCode;

// Layout constants
pub const GAP_SIZE: f32 = 20.0;
const LERP_FACTOR: f32 = 20.0;
const SNAP_THRESHOLD: f32 = 0.5;

#[derive(Clone, Debug)]
pub struct ScrollableLayout {
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

impl LayoutEngine for ScrollableLayout {
    fn get_visible_windows(&self) -> Vec<LayoutWindow> {
        let mut visible_windows = Vec::new();
        let viewport_top_edge = self.scroll_offset_y;
        let viewport_bottom_edge = self.scroll_offset_y + self.viewport.height;

        for (workspace_index, workspace) in self.workspaces.iter().enumerate() {
            let workspace_top_edge = workspace.y_position;
            let workspace_bottom_edge = workspace.y_position + self.workspace_height;

            // Skip workspaces not visible in viewport (viewport culling)
            if workspace_bottom_edge < viewport_top_edge
                || workspace_top_edge > viewport_bottom_edge
            {
                continue;
            }

            let viewport_left_edge = self.scroll_offset_x;
            let viewport_right_edge = self.scroll_offset_x + self.viewport.width;

            for (column_index, column) in workspace.columns.iter().enumerate() {
                let column_left_edge = column.x_position;
                let column_right_edge = column.x_position + column.width;

                // Skip columns not visible in viewport (viewport culling)
                if column_right_edge < viewport_left_edge || column_left_edge > viewport_right_edge
                {
                    continue;
                }

                // Check if this column contains the focused window
                let is_focused = if let Some((focused_workspace_index, focused_column_index)) =
                    self.focused_window
                {
                    focused_workspace_index == workspace_index
                        && focused_column_index == column_index
                } else {
                    false
                };

                let mut vertical_offset = GAP_SIZE; // Start with top gap
                for window_data in &column.windows {
                    visible_windows.push(LayoutWindow {
                        rect: LayoutRect {
                            // Return viewport-relative coordinates
                            x: column.x_position - self.scroll_offset_x,
                            y: workspace.y_position + vertical_offset - self.scroll_offset_y,
                            width: column.width,
                            height: window_data.height,
                        },
                        id: window_data.id,
                        workspace_id: workspace.id,
                        is_focused,
                    });
                    vertical_offset += window_data.height + GAP_SIZE;
                }
            }
        }
        visible_windows
    }

    fn viewport(&self) -> Viewport {
        self.viewport
    }

    fn add_window(&mut self, window_id: u32) {
        let workspace = &mut self.workspaces[self.current_workspace];

        let x_position = if let Some(last_col) = workspace.columns.last() {
            last_col.x_position + last_col.width + GAP_SIZE
        } else {
            GAP_SIZE // Start with gap from left edge
        };

        // Default window width is 1/3 of viewport, accounting for gaps on all sides
        // With 3 windows: 4 gaps (left, between1, between2, right)
        const DEFAULT_WINDOW_COUNT: f32 = 3.0;
        const GAPS_FOR_THREE_WINDOWS: f32 = 4.0;
        let default_width =
            (self.viewport.width - GAPS_FOR_THREE_WINDOWS * GAP_SIZE) / DEFAULT_WINDOW_COUNT;

        workspace.columns.push(Column {
            windows: vec![WindowData {
                id: window_id,
                height: self.viewport.height - 2.0 * GAP_SIZE, // Gap on top and bottom
            }],
            x_position,
            width: default_width,
        });

        // Auto-focus the new window and scroll to it
        let new_column_index = workspace.columns.len() - 1;
        self.set_focus(self.current_workspace, new_column_index);
        self.scroll_to_focused_window();
    }

    fn remove_focused_window(&mut self) {
        if let Some((workspace_index, column_index)) = self.focused_window {
            if workspace_index >= self.workspaces.len() {
                return;
            }

            if column_index >= self.workspaces[workspace_index].columns.len() {
                return;
            }

            // Remove the focused column
            self.workspaces[workspace_index]
                .columns
                .remove(column_index);

            // Reposition all remaining columns to close the gap
            self.reposition_columns(workspace_index);

            // Calculate the total content width of the workspace
            let workspace_content_width =
                if let Some(last_column) = self.workspaces[workspace_index].columns.last() {
                    last_column.x_position + last_column.width + GAP_SIZE
                } else {
                    0.0
                };

            // Adjust scroll position if there's empty space on the right
            if workspace_content_width > self.viewport.width {
                // Content exceeds viewport; ensure we don't scroll past the end
                let max_scroll_x = workspace_content_width - self.viewport.width;
                if self.target_scroll_x > max_scroll_x {
                    self.target_scroll_x = max_scroll_x;
                }
            } else {
                // All content fits in viewport - reset scroll to start
                self.target_scroll_x = 0.0;
            }

            // Update focus after removal
            if self.workspaces[workspace_index].columns.is_empty() {
                // No windows left in workspace
                self.workspaces[workspace_index].focused_column = None;
                self.focused_window = None;
            } else if column_index < self.workspaces[workspace_index].columns.len() {
                // Focus the column that shifted into the removed window's position (from the right)
                self.set_focus(workspace_index, column_index);
            } else if column_index > 0 {
                // We removed the last column, focus the new last column
                self.set_focus(workspace_index, column_index - 1);
            }
        }
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

    fn update(&mut self, delta_time: f32) {
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

    fn handle_key(&mut self, key: KeyCode) {
        match key {
            // Scroll horizontally
            KeyCode::ArrowLeft => self.scroll_horizontal(-200.0),
            KeyCode::ArrowRight => self.scroll_horizontal(200.0),

            // Scroll vertically (between workspaces)
            KeyCode::ArrowUp => self.scroll_vertical(-self.workspace_height),

            KeyCode::ArrowDown => {
                self.scroll_vertical(self.workspace_height);

                // Auto-create workspace if we're at the end
                if self.current_workspace >= self.workspaces.len() - 1 {
                    self.add_workspace();
                }
            }

            // Resize focused window
            KeyCode::BracketLeft => self.resize_focused_window(-50.0),
            KeyCode::BracketRight => self.resize_focused_window(50.0),

            _ => {} // Key not handled by this layout
        }
    }
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

    pub fn new(viewport_width: f32, viewport_height: f32) -> Self {
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

    pub fn add_workspace(&mut self) {
        let new_id = self.workspaces.len();
        let y_position = new_id as f32 * self.workspace_height;
        self.workspaces.push(Workspace {
            id: new_id,
            columns: vec![],
            y_position,
            focused_column: None,
        });
    }

    fn scroll_horizontal(&mut self, horizontal_movement_delta: f32) {
        let current_workspace = &mut self.workspaces[self.current_workspace];

        if current_workspace.columns.is_empty() {
            return;
        }

        let is_moving_left = horizontal_movement_delta < 0.0;
        let target_column_index = match (is_moving_left, current_workspace.focused_column) {
            // Moving left with focus - go to previous column
            (true, Some(current_column_index)) if current_column_index > 0 => {
                Some(current_column_index - 1)
            }
            // Moving right with focus - go to next column
            (false, Some(current_column_index))
                if current_column_index + 1 < current_workspace.columns.len() =>
            {
                Some(current_column_index + 1)
            }
            // No focus - default to first column
            (_, None) => Some(0),
            // Can't move in desired direction
            _ => return,
        };

        if let Some(column_index) = target_column_index {
            self.set_focus(self.current_workspace, column_index);
            self.scroll_to_focused_window();
        }
    }

    fn scroll_vertical(&mut self, vertical_movement_delta: f32) {
        self.target_scroll_y += vertical_movement_delta;

        // Calculate which workspace we should snap to
        let target_workspace_index = (self.target_scroll_y / self.workspace_height).round() as i32;
        let clamped_workspace_index =
            target_workspace_index.clamp(0, self.workspaces.len() as i32 - 1);

        // Snap scroll position to workspace boundary
        self.target_scroll_y = clamped_workspace_index as f32 * self.workspace_height;
        self.current_workspace = clamped_workspace_index as usize;

        // Restore the workspace's previously focused column
        let current_workspace = &self.workspaces[self.current_workspace];
        if let Some(remembered_column_index) = current_workspace.focused_column {
            if remembered_column_index < current_workspace.columns.len() {
                self.set_focus(self.current_workspace, remembered_column_index);
                self.scroll_to_focused_window();
            } else {
                self.focused_window = None;
            }
        } else if !current_workspace.columns.is_empty() {
            // No remembered focus, focus first column
            self.set_focus(self.current_workspace, 0);
            self.scroll_to_focused_window();
        } else {
            self.focused_window = None;
        }
    }

    /// Reposition all columns in a workspace sequentially from left to right
    /// Each column is placed with GAP_SIZE spacing between columns
    fn reposition_columns(&mut self, workspace_index: usize) {
        if workspace_index >= self.workspaces.len() {
            return;
        }

        let workspace = &mut self.workspaces[workspace_index];
        let mut horizontal_position = GAP_SIZE; // Start with left gap
        for column in &mut workspace.columns {
            column.x_position = horizontal_position;
            horizontal_position += column.width + GAP_SIZE;
        }
    }

    fn resize_focused_window(&mut self, width_delta: f32) {
        if let Some((workspace_index, column_index)) = self.focused_window {
            if workspace_index < self.workspaces.len() {
                let workspace = &mut self.workspaces[workspace_index];
                if column_index < workspace.columns.len() {
                    let focused_column = &mut workspace.columns[column_index];

                    // Minimum is 1/6 of viewport (allows 6 windows + 7 gaps to fit)
                    const MIN_WINDOW_FRACTION: f32 = 6.0;
                    const GAP_COUNT_FOR_MIN_WIDTH: f32 = 7.0;
                    let min_width = (self.viewport.width - GAP_COUNT_FOR_MIN_WIDTH * GAP_SIZE)
                        / MIN_WINDOW_FRACTION;

                    // Maximum is full viewport minus left and right gaps
                    let max_width = self.viewport.width - 2.0 * GAP_SIZE;

                    focused_column.width =
                        (focused_column.width + width_delta).clamp(min_width, max_width);

                    // Reposition all columns to account for the width change
                    self.reposition_columns(workspace_index);
                }
            }
        }
    }

    fn scroll_to_focused_window(&mut self) {
        if let Some((workspace_idx, column_index)) = self.focused_window {
            if workspace_idx < self.workspaces.len() {
                // Scroll vertically to the workspace
                self.current_workspace = workspace_idx;
                self.target_scroll_y = workspace_idx as f32 * self.workspace_height;

                // Scroll horizontally to the window (accounting for left gap)
                let workspace = &self.workspaces[workspace_idx];
                if column_index < workspace.columns.len() {
                    let column = &workspace.columns[column_index];
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
