#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4  u_projMatrix;
uniform float u_scale;
out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;
void main() {
    vec3 p = a_pos * u_scale;
    v_worldPos = p;
    v_normal = a_normal;
    v_uv = a_uv;
    gl_Position = u_projMatrix * vec4(p, 1.0);
}