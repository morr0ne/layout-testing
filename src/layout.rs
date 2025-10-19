use winit::keyboard::KeyCode;

const GAP_SIZE: f32 = 20.0;
const LERP_FACTOR: f32 = 20.0;
const SNAP_THRESHOLD: f32 = 0.5;

pub struct Layout {
    workspaces: Vec<Workspace>,
    current_workspace: usize,
    viewport_width: f32,
    viewport_height: f32,
    scroll_x: f32,
    scroll_y: f32,
    target_scroll_x: f32,
    target_scroll_y: f32,
    workspace_height: f32,
    focused_window: Option<(usize, usize)>, // (workspace_idx, column_idx)
}

struct Workspace {
    columns: Vec<Column>,
    y_position: f32,
    focused_column: Option<usize>,
}

struct Column {
    windows: Vec<WindowData>,
    x_position: f32,
    width: f32,
}

struct WindowData {
    id: u32,
    height: f32,
}

pub struct Window {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub id: u32,
    pub is_focused: bool,
}

impl Layout {
    pub fn new(viewport_width: f32, viewport_height: f32) -> Self {
        Self {
            workspaces: vec![Workspace {
                columns: vec![],
                y_position: 0.0,
                focused_column: None,
            }],
            current_workspace: 0,
            viewport_width,
            viewport_height,
            scroll_x: 0.0,
            scroll_y: 0.0,
            target_scroll_x: 0.0,
            target_scroll_y: 0.0,
            workspace_height: viewport_height,
            focused_window: None,
        }
    }

    pub fn get_visible_windows(&self) -> Vec<Window> {
        let mut result = Vec::new();

        for (ws_idx, workspace) in self.workspaces.iter().enumerate() {
            // Skip workspaces not in viewport
            let ws_top = workspace.y_position;
            let ws_bottom = workspace.y_position + self.workspace_height;
            if ws_bottom < self.scroll_y || ws_top > self.scroll_y + self.viewport_height {
                continue;
            }

            for (col_idx, column) in workspace.columns.iter().enumerate() {
                // Skip columns not in viewport
                let col_left = column.x_position;
                let col_right = column.x_position + column.width;
                if col_right < self.scroll_x || col_left > self.scroll_x + self.viewport_width {
                    continue;
                }

                let is_focused = self.focused_window == Some((ws_idx, col_idx));

                let mut y = GAP_SIZE;
                for window in &column.windows {
                    result.push(Window {
                        x: column.x_position - self.scroll_x,
                        y: workspace.y_position + y - self.scroll_y,
                        width: column.width,
                        height: window.height,
                        id: window.id,
                        is_focused,
                    });
                    y += window.height + GAP_SIZE;
                }
            }
        }

        result
    }

    pub fn add_window(&mut self, window_id: u32) {
        let workspace = &mut self.workspaces[self.current_workspace];

        let x_position = workspace
            .columns
            .last()
            .map(|col| col.x_position + col.width + GAP_SIZE)
            .unwrap_or(GAP_SIZE);

        let default_width = (self.viewport_width - 4.0 * GAP_SIZE) / 3.0;
        let window_height = (self.viewport_height - 2.0 * GAP_SIZE).max(10.0);

        workspace.columns.push(Column {
            windows: vec![WindowData {
                id: window_id,
                height: window_height,
            }],
            x_position,
            width: default_width,
        });

        let new_col_idx = workspace.columns.len() - 1;
        self.set_focus(self.current_workspace, new_col_idx);
        self.scroll_to_focused();
    }

    pub fn remove_focused_window(&mut self) {
        if let Some((ws_idx, col_idx)) = self.focused_window {
            if col_idx >= self.workspaces[ws_idx].columns.len() {
                return;
            }

            self.workspaces[ws_idx].columns.remove(col_idx);
            self.reposition_columns(ws_idx);

            // Adjust scroll
            let content_width = self.workspaces[ws_idx]
                .columns
                .last()
                .map(|col| col.x_position + col.width + GAP_SIZE)
                .unwrap_or(0.0);

            if content_width > self.viewport_width {
                let max_scroll = content_width - self.viewport_width;
                self.target_scroll_x = self.target_scroll_x.min(max_scroll);
            } else {
                self.target_scroll_x = 0.0;
            }

            // Update focus
            let workspace = &mut self.workspaces[ws_idx];
            if workspace.columns.is_empty() {
                workspace.focused_column = None;
                self.focused_window = None;
            } else if col_idx < workspace.columns.len() {
                self.set_focus(ws_idx, col_idx);
            } else {
                self.set_focus(ws_idx, col_idx - 1);
            }
        }
    }

    pub fn resize_viewport(&mut self, width: f32, height: f32) {
        let scale = width / self.viewport_width;
        self.viewport_width = width;
        self.viewport_height = height;
        self.workspace_height = height;

        // Ensure window height doesn't go negative for tiny viewports
        let window_height = (height - 2.0 * GAP_SIZE).max(10.0);

        for (idx, workspace) in self.workspaces.iter_mut().enumerate() {
            workspace.y_position = idx as f32 * height;
            for column in &mut workspace.columns {
                column.width *= scale;
                column.x_position *= scale;
                for window in &mut column.windows {
                    window.height = window_height;
                }
            }
        }

        self.scroll_x *= scale;
        self.target_scroll_x *= scale;
        self.scroll_y = self.current_workspace as f32 * height;
        self.target_scroll_y = self.scroll_y;
    }

    pub fn update(&mut self, dt: f32) {
        let lerp = LERP_FACTOR * dt;
        self.scroll_x += (self.target_scroll_x - self.scroll_x) * lerp;
        self.scroll_y += (self.target_scroll_y - self.scroll_y) * lerp;

        if (self.target_scroll_x - self.scroll_x).abs() < SNAP_THRESHOLD {
            self.scroll_x = self.target_scroll_x;
        }
        if (self.target_scroll_y - self.scroll_y).abs() < SNAP_THRESHOLD {
            self.scroll_y = self.target_scroll_y;
        }
    }

    pub fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::ArrowLeft => self.move_focus(-1),
            KeyCode::ArrowRight => self.move_focus(1),
            KeyCode::ArrowUp => self.move_workspace(-1),
            KeyCode::ArrowDown => self.move_workspace(1),
            KeyCode::BracketLeft => self.resize_focused(-50.0),
            KeyCode::BracketRight => self.resize_focused(50.0),
            _ => {}
        }
    }

    fn set_focus(&mut self, ws_idx: usize, col_idx: usize) {
        if ws_idx < self.workspaces.len() {
            let workspace = &mut self.workspaces[ws_idx];
            if col_idx < workspace.columns.len() {
                workspace.focused_column = Some(col_idx);
                self.focused_window = Some((ws_idx, col_idx));
            }
        }
    }

    fn move_focus(&mut self, delta: i32) {
        let workspace = &self.workspaces[self.current_workspace];
        if workspace.columns.is_empty() {
            return;
        }

        let new_col = match workspace.focused_column {
            Some(idx) => {
                let new_idx = (idx as i32 + delta).clamp(0, workspace.columns.len() as i32 - 1);
                new_idx as usize
            }
            None => 0,
        };

        self.set_focus(self.current_workspace, new_col);
        self.scroll_to_focused();
    }

    fn move_workspace(&mut self, delta: i32) {
        let new_ws = (self.current_workspace as i32 + delta)
            .clamp(0, self.workspaces.len() as i32 - 1) as usize;

        // Auto-create workspace if moving down from last
        if delta > 0 && new_ws == self.workspaces.len() - 1 && new_ws == self.current_workspace {
            self.workspaces.push(Workspace {
                columns: vec![],
                y_position: self.workspaces.len() as f32 * self.workspace_height,
                focused_column: None,
            });
        }

        self.current_workspace = new_ws;
        self.target_scroll_y = new_ws as f32 * self.workspace_height;

        let workspace = &self.workspaces[self.current_workspace];
        if let Some(col_idx) = workspace.focused_column {
            if col_idx < workspace.columns.len() {
                self.set_focus(self.current_workspace, col_idx);
                self.scroll_to_focused();
            }
        } else if !workspace.columns.is_empty() {
            self.set_focus(self.current_workspace, 0);
            self.scroll_to_focused();
        } else {
            self.focused_window = None;
        }
    }

    fn reposition_columns(&mut self, ws_idx: usize) {
        let mut x = GAP_SIZE;
        for column in &mut self.workspaces[ws_idx].columns {
            column.x_position = x;
            x += column.width + GAP_SIZE;
        }
    }

    fn resize_focused(&mut self, delta: f32) {
        if let Some((ws_idx, col_idx)) = self.focused_window {
            let workspace = &mut self.workspaces[ws_idx];
            if col_idx < workspace.columns.len() {
                let column = &mut workspace.columns[col_idx];
                let min_width = (self.viewport_width - 7.0 * GAP_SIZE) / 6.0;
                let max_width = self.viewport_width - 2.0 * GAP_SIZE;
                column.width = (column.width + delta).clamp(min_width, max_width);

                self.reposition_columns(ws_idx);
                self.scroll_to_focused();
            }
        }
    }

    fn scroll_to_focused(&mut self) {
        if let Some((ws_idx, col_idx)) = self.focused_window {
            self.current_workspace = ws_idx;
            self.target_scroll_y = ws_idx as f32 * self.workspace_height;

            let workspace = &self.workspaces[ws_idx];
            if col_idx < workspace.columns.len() {
                let column = &workspace.columns[col_idx];
                self.target_scroll_x = (column.x_position - GAP_SIZE).max(0.0);

                // Clamp to valid range
                if let Some(last_col) = workspace.columns.last() {
                    let max_scroll = (last_col.x_position + last_col.width + GAP_SIZE
                        - self.viewport_width)
                        .max(0.0);
                    self.target_scroll_x = self.target_scroll_x.min(max_scroll);
                }
            }
        }
    }
}
