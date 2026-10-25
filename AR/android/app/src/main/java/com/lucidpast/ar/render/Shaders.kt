package com.lucidpast.ar.render

object Shaders {

    val DEPTH_COUNT = """
#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer SplatBuf { uint splats[]; };
layout(std430, binding = 1) buffer CountBuf { uint counts[]; };
uniform mat4 uModel;
uniform mat4 uView;
uniform vec2 uDepthRange;
uniform uint uCount;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uCount) { return; }
    uint base = i * 8u;
    vec3 p = vec3(uintBitsToFloat(splats[base]), uintBitsToFloat(splats[base + 1u]), uintBitsToFloat(splats[base + 2u]));
    vec3 w = (uModel * vec4(p, 1.0)).xyz;
    vec4 t = uView * vec4(w, 1.0);
    float d = -t.z;
    float key = (uDepthRange.y - d) / (uDepthRange.y - uDepthRange.x);
    int k = int(clamp(key, 0.0, 1.0) * 65535.0);
    atomicAdd(counts[k], 1u);
}
""".trim()

    val SCAN_BLOCKS = """
#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 256) in;
layout(std430, binding = 1) readonly buffer CountBuf { uint counts[]; };
layout(std430, binding = 3) writeonly buffer ScannedBuf { uint scanned[]; };
layout(std430, binding = 4) buffer BlockSumsBuf { uint blockSums[]; };
shared uint temp[256];
void main() {
    uint lid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x * 256u + lid;
    temp[lid] = counts[gid];
    barrier();
    for (uint off = 1u; off < 256u; off *= 2u) {
        uint add = (lid >= off) ? temp[lid - off] : 0u;
        barrier();
        temp[lid] += add;
        barrier();
    }
    scanned[gid] = temp[lid];
    if (lid == 255u) { blockSums[gl_WorkGroupID.x] = temp[lid]; }
}
""".trim()

    val SCAN_SUMS = """
#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 256) in;
layout(std430, binding = 4) buffer BlockSumsBuf { uint blockSums[]; };
shared uint temp[256];
void main() {
    uint lid = gl_LocalInvocationID.x;
    temp[lid] = blockSums[lid];
    barrier();
    for (uint off = 1u; off < 256u; off *= 2u) {
        uint add = (lid >= off) ? temp[lid - off] : 0u;
        barrier();
        temp[lid] += add;
        barrier();
    }
    blockSums[lid] = temp[lid];
}
""".trim()

    val SCAN_OFFSETS = """
#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 256) in;
layout(std430, binding = 3) readonly buffer ScannedBuf { uint scanned[]; };
layout(std430, binding = 4) readonly buffer BlockSumsBuf { uint blockSums[]; };
layout(std430, binding = 1) writeonly buffer PrefixBuf { uint prefix[]; };
void main() {
    uint lid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x * 256u + lid;
    uint incl = scanned[gid];
    uint own = (lid > 0u) ? (incl - scanned[gid - 1u]) : incl;
    uint exclInBlock = incl - own;
    uint blockOff = (gl_WorkGroupID.x > 0u) ? blockSums[gl_WorkGroupID.x - 1u] : 0u;
    prefix[gid] = exclInBlock + blockOff;
}
""".trim()

    val SCATTER = """
#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer SplatBuf { uint splats[]; };
layout(std430, binding = 1) readonly buffer PrefixBuf { uint prefix[]; };
layout(std430, binding = 2) buffer CursorBuf { uint cursor[]; };
layout(std430, binding = 5) writeonly buffer OrderBuf { uint order[]; };
uniform mat4 uModel;
uniform mat4 uView;
uniform vec2 uDepthRange;
uniform uint uCount;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uCount) { return; }
    uint base = i * 8u;
    vec3 p = vec3(uintBitsToFloat(splats[base]), uintBitsToFloat(splats[base + 1u]), uintBitsToFloat(splats[base + 2u]));
    vec3 w = (uModel * vec4(p, 1.0)).xyz;
    vec4 t = uView * vec4(w, 1.0);
    float d = -t.z;
    float key = (uDepthRange.y - d) / (uDepthRange.y - uDepthRange.x);
    int k = int(clamp(key, 0.0, 1.0) * 65535.0);
    uint slot = prefix[k] + atomicAdd(cursor[k], 1u);
    if (slot < uCount) { order[slot] = i; }
}
""".trim()

    val PREPARE = """
#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer SplatBuf { uint splats[]; };
layout(std430, binding = 5) readonly buffer OrderBuf { uint order[]; };
layout(std430, binding = 6) writeonly buffer PreparedBuf { vec4 data[]; };
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform vec2 uViewport;
uniform vec2 uFocal;
uniform uint uCount;

mat3 quatToMat3(vec4 r) {
    float w = r.x;
    float x = r.y;
    float y = r.z;
    float z = r.w;
    return mat3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
        2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
        2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)
    );
}

void writeZeros(uint o) {
    data[o] = vec4(0.0);
    data[o + 1u] = vec4(0.0);
    data[o + 2u] = vec4(0.0);
    data[o + 3u] = vec4(0.0);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uCount) { return; }
    uint idx = order[i];
    uint o = 4u * i;
    if (idx >= uCount) {
        writeZeros(o);
        return;
    }
    uint base = idx * 8u;
    vec3 p = vec3(uintBitsToFloat(splats[base]), uintBitsToFloat(splats[base + 1u]), uintBitsToFloat(splats[base + 2u]));
    vec3 sc = vec3(uintBitsToFloat(splats[base + 3u]), uintBitsToFloat(splats[base + 4u]), uintBitsToFloat(splats[base + 5u]));
    uint rgbaW = splats[base + 6u];
    float alpha = float(rgbaW >> 24u) / 255.0;
    vec3 rgb = vec3(
        float(rgbaW & 255u), float((rgbaW >> 8u) & 255u), float((rgbaW >> 16u) & 255u)
    ) / 255.0;
    uint quatW = splats[base + 7u];
    vec4 qn = (vec4(
        float(quatW & 255u), float((quatW >> 8u) & 255u),
        float((quatW >> 16u) & 255u), float(quatW >> 24u)
    ) - 128.0) / 128.0;

    if (alpha < 0.01) {
        writeZeros(o);
        return;
    }

    vec3 wpos = (uModel * vec4(p, 1.0)).xyz;
    vec4 t = uView * vec4(wpos, 1.0);
    vec4 pos2d = uProj * t;
    if (pos2d.w < 0.02) {
        writeZeros(o);
        return;
    }

    mat3 M3 = mat3(uModel);
    mat3 R = quatToMat3(qn);
    mat3 S = mat3(sc.x, 0.0, 0.0, 0.0, sc.y, 0.0, 0.0, 0.0, sc.z);
    mat3 M = M3 * R * S;
    mat3 cov3 = M * transpose(M);

    mat3 W = transpose(mat3(uView));
    mat3 J = mat3(
        uFocal.x / t.z, 0.0, -(uFocal.x * t.x) / (t.z * t.z),
        0.0, uFocal.y / t.z, -(uFocal.y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0
    );
    mat3 T = W * J;
    mat3 cov2 = transpose(T) * cov3 * T;
    cov2[0][0] += 0.3;
    cov2[1][1] += 0.3;

    float mid = 0.5 * (cov2[0][0] + cov2[1][1]);
    float rad = length(vec2(0.5 * (cov2[0][0] - cov2[1][1]), cov2[0][1]));
    float l1 = mid + rad;
    float l2 = mid - rad;
    if (l1 <= 0.0 || l2 < 0.05) {
        writeZeros(o);
        return;
    }

    vec2 e1;
    if (abs(cov2[0][1]) < 1e-6) {
        e1 = (cov2[0][0] >= cov2[1][1]) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    } else {
        e1 = normalize(vec2(cov2[0][1], l1 - cov2[0][0]));
    }
    vec2 e2 = vec2(-e1.y, e1.x);
    float h1 = min(3.0 * sqrt(l1), 256.0);
    float h2 = min(3.0 * sqrt(l2), 256.0);

    vec2 centerPx = (vec2(pos2d.x / pos2d.w, pos2d.y / pos2d.w) * 0.5 + 0.5) * uViewport;
    float det = cov2[0][0] * cov2[1][1] - cov2[0][1] * cov2[0][1];
    vec3 conic = vec3(cov2[1][1], -cov2[0][1], cov2[0][0]) / det;

    data[o] = vec4(centerPx, 0.0, 0.0);
    data[o + 1u] = vec4(e1 * h1, e2 * h2);
    data[o + 2u] = vec4(conic, alpha);
    data[o + 3u] = vec4(rgb, alpha);
}
""".trim()

    val DRAW_VERT = """
#version 310 es
precision highp float;
precision highp int;
layout(std430, binding = 6) readonly buffer PreparedBuf { vec4 data[]; };
uniform vec2 uViewport;
out vec4 vColor;
out vec2 vOffset;
out vec3 vConic;
void main() {
    uint o = 4u * uint(gl_InstanceID);
    vec4 d0 = data[o];
    vec4 d1 = data[o + 1u];
    vec4 d2 = data[o + 2u];
    vec4 d3 = data[o + 3u];
    int c = gl_VertexID & 3;
    float cx = (c == 0 || c == 2) ? -1.0 : 1.0;
    float cy = (c == 0 || c == 1) ? -1.0 : 1.0;
    vec2 offset = d1.xy * cx + d1.zw * cy;
    vec2 ndc = (d0.xy + offset) / uViewport * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vColor = vec4(d3.rgb, d2.w);
    vOffset = offset;
    vConic = d2.xyz;
}
""".trim()

    val DRAW_FRAG = """
#version 310 es
precision highp float;
in vec4 vColor;
in vec2 vOffset;
in vec3 vConic;
out vec4 outColor;
void main() {
    float power = -0.5 * (
        vConic.x * vOffset.x * vOffset.x
        + 2.0 * vConic.y * vOffset.x * vOffset.y
        + vConic.z * vOffset.y * vOffset.y
    );
    float alpha = vColor.a * exp(power);
    if (alpha < 0.004) { discard; }
    outColor = vec4(vColor.rgb * alpha, alpha);
}
""".trim()
}
