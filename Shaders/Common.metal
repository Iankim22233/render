#include <metal_stdlib>
using namespace metal;

// Must match `FrameUniforms` in Renderer.swift exactly.
struct FrameUniforms {
    float4x4 view;
    float4x4 proj;
    float4x4 viewProj;
    float4x4 invViewProj;
    float4x4 invProj;
    float4x4 invView;
    float4x4 viewProjNoJitter;
    float4x4 invViewProjNoJitter;
    float4x4 prevViewProjNoJitter;
    float4x4 cascadeViewProj[4];
    float4 cascadeInfo[4];   // x: texel world size, y: depth range, z: radius, w: unused
    float4 cameraPos;        // xyz, w: time
    float4 sunDir;           // xyz towards sun, w: angular radius (rad)
    float4 sunColor;         // rgb irradiance, w: sun disk visibility (clouds)
    float4 screenSize;       // w, h, 1/w, 1/h
    float4 jitter;           // xy jitter, z frame index, w near plane
    float4 fog;              // x density, y height falloff, z base height, w anisotropy
    float4 fogColor;         // rgb scattering albedo, w ambient strength
    float4 wind;             // xy direction, z strength, w time
    float4 terrain;          // x size, y 1/size, z heightmap res, w canopy height scale
    float4 shadowParams;     // x PCF samples, y PCSS, z softness, w shadow distance
    float4 qualityParams;    // x contact-shadow steps, y volumetric steps, z volumetric distance, w AO enabled
    float4 grassParams;      // x radius, y base spacing, z height scale, w dryness
    float4 sceneParams;      // x autumn, y cloud coverage, z turbidity, w moss
    float4 postParams;       // x exposure EV, y bloom, z tonemapper, w vignette
    float4 postParams2;      // x saturation, y contrast, z grain, w sharpen
    float4 colorGrass;       // rgb grass tint, w fog noise
    float4 misc;             // x LOD bias, y clouds on, z debug view, w cloud shadow strength
    float4 water;            // x level, y enabled, z planar reflections, w unused
    float4 waterColor;       // rgb absorption per meter
    float4 weather;          // x rain, y wetness, z overcast, w lightning flash
    float4 flags;            // x reflection pass
    float4 snow;             // x snowfall, y snow cover on surfaces, z stars, w night (0 day .. 1 full night)
    float4 moon;             // xyz direction to the moon, w: 1 when the moon is the key light (night)
    float4 moonParams;       // x elongation from the sun (rad; pi = full), y disk brightness, z moonlight irradiance
    float4 skySun;           // xyz real sun direction (for the sky model; sunDir is the key light)
};

struct TreeVertex {
    float3 position;
    float3 normal;
    float3 tangent;
    float2 uv;
    float4 extra;   // x flex, y phase, z layer, w ao
};

struct TreeInstance {
    float4 posScale;
    float4 params;  // x rotation, y tint seed, z wind phase, w lod fade
};

struct GBufferOut {
    float4 albedo   [[color(0)]];  // rgb albedo, a: ao
    float4 normal   [[color(1)]];  // xyz world normal, w: roughness
    float4 material [[color(2)]];  // x translucency, y material id / 3, z specular, w unused
};

struct FSQOut {
    float4 position [[position]];
    float2 uv;
};

constant float PI = 3.14159265359;

constexpr sampler sLinearClamp(filter::linear, mip_filter::linear, address::clamp_to_edge);
constexpr sampler sLinearRepeat(filter::linear, mip_filter::linear, address::repeat);
constexpr sampler sPointClamp(filter::nearest, address::clamp_to_edge);
constexpr sampler sSky(filter::linear, s_address::repeat, t_address::clamp_to_edge);
constexpr sampler sShadowCmp(filter::linear, compare_func::less_equal, address::clamp_to_edge);

vertex FSQOut fullscreenVS(uint vid [[vertex_id]]) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    FSQOut o;
    o.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    o.uv = uv;
    return o;
}

// ---------------------------------------------------------------- hashing / noise

inline uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline float hashU(uint v) { return float(pcgHash(v)) * (1.0 / 4294967296.0); }

inline float hash12(float2 p) {
    float3 p3 = fract(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
inline float2 hash22(float2 p) {
    float3 p3 = fract(float3(p.xyx) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}
inline float hash13(float3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
inline float3 hash33(float3 p3) {
    p3 = fract(p3 * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

inline float vnoise2(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = hash12(i), b = hash12(i + float2(1, 0)), c = hash12(i + float2(0, 1)), d = hash12(i + float2(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

inline float vnoise3(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i), n100 = hash13(i + float3(1, 0, 0)), n010 = hash13(i + float3(0, 1, 0)), n110 = hash13(i + float3(1, 1, 0));
    float n001 = hash13(i + float3(0, 0, 1)), n101 = hash13(i + float3(1, 0, 1)), n011 = hash13(i + float3(0, 1, 1)), n111 = hash13(i + float3(1, 1, 1));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
               mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

inline float fbm2(float2 p, int oct) {
    float s = 0.0, a = 0.5, n = 0.0;
    for (int i = 0; i < oct; i++) {
        s += a * vnoise2(p);
        n += a;
        p = float2(p.x * 1.6 - p.y * 1.2, p.x * 1.2 + p.y * 1.6) + 17.13;
        a *= 0.5;
    }
    return s / n;
}

inline float fbm3(float3 p, int oct) {
    float s = 0.0, a = 0.5, n = 0.0;
    for (int i = 0; i < oct; i++) {
        s += a * vnoise3(p);
        n += a;
        p = p * 2.03 + 11.7;
        a *= 0.5;
    }
    return s / n;
}

// Interleaved gradient noise, decorrelated per frame.
inline float ign(float2 px, float frame) {
    px += 5.588238 * fmod(frame, 64.0);
    return fract(52.9829189 * fract(dot(px, float2(0.06711056, 0.00583715))));
}

// ---------------------------------------------------------------- common math

inline float3 reconstructWorld(float2 uv, float depth, float4x4 invVP) {
    float4 ndc = float4(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    float4 w = invVP * ndc;
    return w.xyz / w.w;
}

inline float3 reconstructView(float2 uv, float depth, float4x4 invP) {
    float4 ndc = float4(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    float4 v = invP * ndc;
    return v.xyz / v.w;
}

inline float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

inline float2 heightUV(float2 xz, constant FrameUniforms& u) {
    float n = u.terrain.z;
    float2 t = xz * u.terrain.y + 0.5;
    return (t * (n - 1.0) + 0.5) / n;
}

inline float3 rotY(float3 v, float c, float s) { return float3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z); }

// ---------------------------------------------------------------- sky

inline float2 skyUV(float3 dir) {
    float az = atan2(dir.z, dir.x);
    float el = asin(clamp(dir.y, -1.0, 1.0));
    float v = 0.5 - 0.5 * sign(el) * sqrt(abs(el) / (PI * 0.5));
    return float2(az / (2.0 * PI) + 0.5, v);
}

inline float3 sampleSky(texture2d<float> lut, float3 dir) {
    return lut.sample(sSky, skyUV(dir), level(0)).rgb;
}

// L1 spherical-harmonics irradiance (coefficients already include the radiance projection).
inline float3 shIrradiance(float3 n, constant float4* sh) {
    float3 e = PI * 0.282095 * sh[0].rgb
             + (2.0 * PI / 3.0) * 0.488603 * (sh[1].rgb * n.y + sh[2].rgb * n.z + sh[3].rgb * n.x);
    return max(e, 0.0);
}

inline float phaseHG(float cosT, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cosT, 1e-4), 1.5));
}

// Canopy sky-occlusion: forest floors receive much less skylight than clearings.
inline float canopyOcclusion(float3 P, texture2d<float> canopy, texture2d<float> heightTex, constant FrameUniforms& u) {
    float4 c = canopy.sample(sLinearClamp, P.xz * u.terrain.y + 0.5, level(0));
    float ground = heightTex.sample(sLinearClamp, heightUV(P.xz, u), level(0)).r;
    float topH = c.g * u.terrain.w;
    float above = P.y - ground;
    float under = 1.0 - smoothstep(topH * 0.55, topH * 1.05 + 0.01, above);
    return 1.0 - c.r * under * 0.72;
}

// ---------------------------------------------------------------- shadows

// Returns cascade-local coords in [0,1] (xy) and depth (z). Far cascades occupy a sub-rectangle of their slice (cascadeInfo.w).
inline bool cascadeCoord(float3 P, int c, constant FrameUniforms& u, thread float3& sc) {
    float4 lp = u.cascadeViewProj[c] * float4(P, 1.0);
    sc = lp.xyz / lp.w;
    sc.xy = sc.xy * float2(0.5, -0.5) + 0.5;
    return all(sc.xy > 0.015) && all(sc.xy < 0.985) && sc.z > 0.0 && sc.z < 1.0;
}

inline float2 cascadeTexCoord(float2 local, int c, constant FrameUniforms& u) {
    return clamp(local, 0.0, 1.0) * u.cascadeInfo[c].w;
}

// Beyond the shadow cascades, estimate sun visibility from the canopy map so distant forest and fog stay consistent.
inline float shadowFallback(float3 P, texture2d<float> canopy, texture2d<float> heightTex, constant FrameUniforms& u) {
    float4 c = canopy.sample(sLinearClamp, P.xz * u.terrain.y + 0.5, level(0));
    float ground = heightTex.sample(sLinearClamp, heightUV(P.xz, u), level(0)).r;
    float topH = c.g * u.terrain.w;
    float rel = saturate((P.y - ground) / max(topH, 1.0));
    float slant = saturate(1.0 - u.sunDir.y) * 0.25 + 0.75;
    return 1.0 - saturate(c.r * 1.1) * (1.0 - smoothstep(0.55, 1.0, rel)) * 0.88 * slant;
}

inline float cascadeEdgeFade(float3 sc) {
    float e = max(abs(sc.x - 0.5), abs(sc.y - 0.5)) * 2.0;
    return smoothstep(0.86, 0.98, e);
}

// Cheap single-tap (hardware 2x2 PCF) shadow, used by volumetrics.
inline float shadowSimple(float3 P, constant FrameUniforms& u, depth2d_array<float> sm, texture2d<float> canopy, texture2d<float> heightTex) {
    for (int c = 0; c < 4; c++) {
        float3 sc;
        if (cascadeCoord(P, c, u, sc)) {
            float bias = (u.cascadeInfo[c].x * 1.5 + 0.03) / u.cascadeInfo[c].y;
            float s = sm.sample_compare(sShadowCmp, cascadeTexCoord(sc.xy, c, u), c, sc.z - bias);
            if (c == 3) s = mix(s, shadowFallback(P, canopy, heightTex, u), cascadeEdgeFade(sc));
            return s;
        }
    }
    return shadowFallback(P, canopy, heightTex, u);
}

// ---------------------------------------------------------------- water & rain helpers

inline float lakeMaskAt(float2 xz, texture2d<float> lakeTex, constant FrameUniforms& u) {
    return lakeTex.sample(sLinearClamp, xz * u.terrain.y + 0.5, level(0)).r;
}

// Ray / lake-surface intersection. Returns distance or -1.
inline float waterHit(float3 ro, float3 rd, constant FrameUniforms& u, texture2d<float> lakeTex, thread float3& hitP) {
    if (u.water.y < 0.5 || abs(rd.y) < 1e-5) return -1.0;
    float t = (u.water.x - ro.y) / rd.y;
    if (t <= 0.0) return -1.0;
    hitP = ro + rd * t;
    if (lakeMaskAt(hitP.xz, lakeTex, u) < 0.5) return -1.0;
    return t;
}

// Expanding rain-drop rings: returns a normal perturbation (xz).
inline float2 rainRipples(float2 p, float t, float intensity) {
    float2 n = 0.0;
    if (intensity <= 0.001) return n;
    for (int layer = 0; layer < 3; layer++) {
        float sc = 2.6 + float(layer) * 1.37;
        float2 q = p * sc + float2(float(layer) * 13.7, float(layer) * 7.3);
        float2 cell = floor(q);
        float2 f = fract(q);
        float2 h = hash22(cell + float(layer) * 31.0);
        if (h.x > intensity * 0.9 + 0.08) continue;
        float2 c = 0.25 + 0.5 * h;
        float age = fract(t * (0.9 + h.y * 0.6) + h.x * 7.0);
        float2 dv = f - c;
        float d = length(dv);
        float r = age * 0.42;
        float x = d - r;
        float ring = sin(x * 70.0) * exp(-x * x * 900.0) * (1.0 - age) * (1.0 - age);
        n += (dv / max(d, 1e-3)) * ring;
    }
    return n * 0.55;
}

// ---------------------------------------------------------------- snow helpers

constant float3 kSnowAlbedo = float3(0.86, 0.9, 0.96);

// How much snow sits on a surface: upward-facing surfaces fill first, steeper ones as cover grows.
inline float snowCoverMask(float upY, float3 wp, float cover, float bias) {
    if (cover <= 0.001) return 0.0;
    float n = fbm2(wp.xz * 0.35 + wp.y * 0.2, 3);
    float t = mix(0.95, 0.02, cover) + bias;
    return smoothstep(t, t + 0.25, upY + (n - 0.5) * 0.4) * smoothstep(0.0, 0.2, cover);
}
