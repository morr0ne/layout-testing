// ============================================================================
// TEXTURED WINDOW PIPELINE
// Used for rendering window contents with texture sampling
// ============================================================================

struct TexturedVertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct TexturedVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@group(0) @binding(0)
var t_texture: texture_2d<f32>;
@group(0) @binding(1)
var t_sampler: sampler;

@vertex
fn vs_textured(model: TexturedVertexInput) -> TexturedVertexOutput {
    var out: TexturedVertexOutput;
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    out.tex_coords = model.tex_coords;
    return out;
}

@fragment
fn fs_textured(in: TexturedVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_texture, t_sampler, in.tex_coords);
}

// ============================================================================
// SOLID COLOR PIPELINE
// Used for rendering borders and other solid-colored geometry
// ============================================================================

struct ColoredVertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
}

struct ColoredVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_colored(model: ColoredVertexInput) -> ColoredVertexOutput {
    var out: ColoredVertexOutput;
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    out.color = model.color;
    return out;
}

@fragment
fn fs_colored(in: ColoredVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
