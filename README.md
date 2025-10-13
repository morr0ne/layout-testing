# Layout Testing

Layout prototypes for upstreaming into the [Verdi](https://github.com/verdiwm/verdi) Wayland compositor.

## Current Implementation

**Niri-style scrollable layout**: Infinite horizontal scrolling within workspaces with vertical workspace navigation. Windows are organized in columns with smooth lerp-based animations.

## Build & Run

```bash
cargo run
```

Controls: Arrow keys to navigate, W to add windows, [ ] to resize, Q to close.

## Planned Features

- i3-style tiling layout
- Window rules system
- Floating/dynamic window support
