#version 330 core
in vec3 v_worldPos;
in vec3 v_normal;
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 fragColor;
void main() {
    vec3 n = normalize(v_normal); // smooth shading
    // vec3 n = normalize(cross(dFdx(v_worldPos), dFdy(v_worldPos))); // flat shading
    float diff = abs(dot(n, normalize(vec3(0.4, 1.0, 0.6))));
    vec3 base = texture(u_tex, v_uv).rgb;
    // fragColor = vec4(1.0, 0.5, 0.1, 1.0);
    fragColor = vec4(base * (0.35 + 0.65 * diff), 1.0);  // texture + a little shading
}