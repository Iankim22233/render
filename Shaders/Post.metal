// ================================================================ TEMPORAL ANTI-ALIASING

inline float3 rgbToYCoCg(float3 c) {
    return float3(dot(c, float3(0.25, 0.5, 0.25)), dot(c, float3(0.5, 0.0, -0.5)), dot(c, float3(-0.25, 0.5, -0.25)));
}
inline float3 yCoCgToRgb(float3 c) {
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}
inline float3 tmWeight(float3 c) { return c / (1.0 + luminance(c)); }
inline float3 tmInverse(float3 c) { return c / max(1.0 - luminance(c), 1e-3); }

inline float3 clipAABB(float3 mn, float3 mx, float3 p) {
    float3 center = 0.5 * (mx + mn);
    float3 ext = 0.5 * (mx - mn) + 1e-5;
    float3 v = p - center;
    float3 a = abs(v / ext);
    float m = max(a.x, max(a.y, a.z));
    return m > 1.0 ? center + v / m : p;
}

inline float3 sampleCatmullRom(texture2d<float> tex, float2 uv, float2 texSize) {
    float2 samplePos = uv * texSize;
    float2 tp1 = floor(samplePos - 0.5) + 0.5;
    float2 f = samplePos - tp1;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 off12 = w2 / w12;
    float2 tp0 = (tp1 - 1.0) / texSize;
    float2 tp3 = (tp1 + 2.0) / texSize;
    float2 tp12 = (tp1 + off12) / texSize;
    float3 r = 0.0;
    r += tex.sample(sLinearClamp, float2(tp0.x, tp0.y), level(0)).rgb * w0.x * w0.y;
    r += tex.sample(sLinearClamp, float2(tp12.x, tp0.y), level(0)).rgb * w12.x * w0.y;
    r += tex.sample(sLinearClamp, float2(tp3.x, tp0.y), level(0)).rgb * w3.x * w0.y;
    r += tex.sample(sLinearClamp, float2(tp0.x, tp12.y), level(0)).rgb * w0.x * w12.y;
    r += tex.sample(sLinearClamp, float2(tp12.x, tp12.y), level(0)).rgb * w12.x * w12.y;
    r += tex.sample(sLinearClamp, float2(tp3.x, tp12.y), level(0)).rgb * w3.x * w12.y;
    r += tex.sample(sLinearClamp, float2(tp0.x, tp3.y), level(0)).rgb * w0.x * w3.y;
    r += tex.sample(sLinearClamp, float2(tp12.x, tp3.y), level(0)).rgb * w12.x * w3.y;
    r += tex.sample(sLinearClamp, float2(tp3.x, tp3.y), level(0)).rgb * w3.x * w3.y;
    return max(r, 0.0);
}

fragment float4 taaFS(FSQOut in [[stage_in]],
                      constant FrameUniforms& u [[buffer(2)]],
                      constant float4& tp [[buffer(3)]],
                      texture2d<float> cur [[texture(0)]],
                      texture2d<float> hist [[texture(1)]],
                      depth2d<float> depthTex [[texture(2)]],
                      texture2d<float> lakeTex [[texture(3)]]) {
    int2 px = int2(in.position.xy);
    int2 size = int2(cur.get_width(), cur.get_height());
    float3 m1 = 0.0, m2 = 0.0, center = 0.0;
    float closest = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            int2 p = clamp(px + int2(x, y), int2(0), size - 1);
            float3 s = tmWeight(cur.read(uint2(p)).rgb);
            float3 yc = rgbToYCoCg(s);
            m1 += yc;
            m2 += yc * yc;
            if (x == 0 && y == 0) center = s;
            closest = max(closest, depthTex.read(uint2(p)));
        }
    }
    float3 mean = m1 / 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
    float3 wp = reconstructWorld(in.uv, max(closest, 1e-6), u.invViewProjNoJitter);
    // reproject the lake surface (not the lake bed) so reflections don't smear
    float3 camP = u.cameraPos.xyz;
    float3 wh;
    float tw = waterHit(camP, normalize(wp - camP), u, lakeTex, wh);
    if (tw > 0.0 && tw < length(wp - camP)) wp = wh;
    float4 pc = u.prevViewProjNoJitter * float4(wp, 1.0);
    float2 prevUV = (pc.xy / pc.w) * float2(0.5, -0.5) + 0.5;
    bool valid = tp.y < 0.5 && pc.w > 0.0 && all(prevUV > 0.0) && all(prevUV < 1.0);
    float3 result;
    if (valid) {
        float3 h = tmWeight(sampleCatmullRom(hist, prevUV, float2(size)));
        float3 hy = rgbToYCoCg(h);
        float gamma = 1.1;
        hy = clipAABB(mean - gamma * sigma, mean + gamma * sigma, hy);
        float velPx = length((in.uv - prevUV) * float2(size));
        float blend = mix(tp.x, 0.78, saturate(velPx / 30.0));
        result = yCoCgToRgb(mix(rgbToYCoCg(center), hy, blend));
    } else {
        result = center;
    }
    result = tmInverse(max(result, 0.0));
    if (any(isnan(result)) || any(isinf(result))) result = 0.0;
    return float4(result, 1.0);
}

// ================================================================ BLOOM

fragment float4 bloomDownFS(FSQOut in [[stage_in]],
                            texture2d<float> src [[texture(0)]],
                            constant float4& bp [[buffer(0)]]) {
    float2 t = bp.xy;
    float2 uv = in.uv;
    #define S(dx, dy) src.sample(sLinearClamp, uv + t * float2(dx, dy), level(0)).rgb
    float3 a = S(-2, -2), b = S(0, -2), c = S(2, -2);
    float3 d = S(-2, 0), e = S(0, 0), f = S(2, 0);
    float3 g = S(-2, 2), h = S(0, 2), i = S(2, 2);
    float3 j = S(-1, -1), k = S(1, -1), l = S(-1, 1), m = S(1, 1);
    #undef S
    float3 r;
    if (bp.z > 0.5) {
        float3 g0 = (a + b + d + e) * 0.25, g1 = (b + c + e + f) * 0.25, g2 = (d + e + g + h) * 0.25, g3 = (e + f + h + i) * 0.25, g4 = (j + k + l + m) * 0.25;
        float w0 = 0.125 / (1.0 + luminance(g0)), w1 = 0.125 / (1.0 + luminance(g1)), w2 = 0.125 / (1.0 + luminance(g2));
        float w3 = 0.125 / (1.0 + luminance(g3)), w4 = 0.5 / (1.0 + luminance(g4));
        r = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
    } else {
        r = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    }
    if (any(isnan(r))) r = 0.0;
    return float4(max(r, 0.0), 1.0);
}

fragment float4 bloomUpFS(FSQOut in [[stage_in]],
                          texture2d<float> src [[texture(0)]],
                          constant float4& bp [[buffer(0)]]) {
    float2 t = bp.xy * bp.z;
    float2 uv = in.uv;
    float3 s = src.sample(sLinearClamp, uv, level(0)).rgb * 4.0;
    s += (src.sample(sLinearClamp, uv + float2(-t.x, 0), level(0)).rgb + src.sample(sLinearClamp, uv + float2(t.x, 0), level(0)).rgb
        + src.sample(sLinearClamp, uv + float2(0, -t.y), level(0)).rgb + src.sample(sLinearClamp, uv + float2(0, t.y), level(0)).rgb) * 2.0;
    s += src.sample(sLinearClamp, uv + float2(-t.x, -t.y), level(0)).rgb + src.sample(sLinearClamp, uv + float2(t.x, -t.y), level(0)).rgb
       + src.sample(sLinearClamp, uv + float2(-t.x, t.y), level(0)).rgb + src.sample(sLinearClamp, uv + float2(t.x, t.y), level(0)).rgb;
    return float4(s / 16.0, 1.0);
}

// ================================================================ AUTO EXPOSURE

kernel void exposureKernel(texture2d<float> src [[texture(0)]],
                           device float* expo [[buffer(0)]],
                           constant float4& ep [[buffer(1)]],
                           constant float4& np [[buffer(2)]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sumL[256];
    threadgroup float sumW[256];
    uint w = src.get_width(), h = src.get_height();
    uint n = w * h;
    float sl = 0.0, sw = 0.0;
    for (uint i = tid; i < n; i += 256u) {
        uint2 p = uint2(i % w, i / w);
        float l = luminance(src.read(p).rgb);
        float2 uv = (float2(p) + 0.5) / float2(w, h);
        float cw = 1.0 - 0.65 * saturate(length(uv - 0.5) * 1.6);
        sl += log2(max(l, 1e-5)) * cw;
        sw += cw;
    }
    sumL[tid] = sl;
    sumW[tid] = sw;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { sumL[tid] += sumL[tid + s]; sumW[tid] += sumW[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float avg = exp2(sumL[0] / max(sumW[0], 1e-4));
        // darker target at night so night still looks like night (but stars and the sky glow are visible)
        // at night the eye doesn't fully adapt: brightness follows the moonlight setting (np.x night, np.y night key factor)
        float key = 0.13 * mix(1.0, np.y, np.x);
        float target = clamp(key / max(avg, 1e-8), 0.5, 12000.0) * exp2(ep.z);
        if (ep.w < 0.5) target = 8.0 * exp2(ep.z);
        float c = expo[0];
        if (!(c > 0.0) || !(c < 1e5) || ep.y > 100.0) c = target;
        else c = exp2(mix(log2(c), log2(target), 1.0 - exp(-ep.x * ep.y)));
        expo[0] = c;
    }
}

// ================================================================ FINAL: sharpen, bloom, exposure, tonemap, grade

inline float3 agxContrast(float3 x) {
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

inline float3 agxTonemap(float3 c) {
    const float3x3 m = float3x3(float3(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
                                float3(0.0784335999999992, 0.878468636469772, 0.0784336),
                                float3(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
    const float3x3 mi = float3x3(float3(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
                                 float3(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
                                 float3(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));
    const float minEv = -12.47393, maxEv = 4.026069;
    c = m * max(c, float3(1e-10));
    c = clamp(log2(c), minEv, maxEv);
    c = (c - minEv) / (maxEv - minEv);
    c = agxContrast(c);
    // gentle "punchy" look
    float l = dot(c, float3(0.2126, 0.7152, 0.0722));
    c = pow(max(c, 0.0), float3(1.12));
    c = l + 1.15 * (c - l);
    c = mi * c;
    return pow(max(c, 0.0), float3(2.2));
}

inline float3 acesFitted(float3 v) {
    float3 a = float3(dot(float3(0.59719, 0.35458, 0.04823), v), dot(float3(0.07600, 0.90834, 0.01566), v), dot(float3(0.02840, 0.13383, 0.83777), v));
    float3 r = (a * (a + 0.0245786) - 0.000090537) / (a * (0.983729 * a + 0.4329510) + 0.238081);
    float3 o = float3(dot(float3(1.60475, -0.53108, -0.07367), r), dot(float3(-0.10208, 1.10813, -0.00605), r), dot(float3(-0.00327, -0.07276, 1.07602), r));
    return saturate(o);
}

fragment float4 finalFS(FSQOut in [[stage_in]],
                        constant FrameUniforms& u [[buffer(2)]],
                        texture2d<float> color [[texture(0)]],
                        texture2d<float> bloom [[texture(1)]],
                        texture2d<float> rainTex [[texture(2)]],
                        device const float* expo [[buffer(0)]]) {
    float2 uv = in.uv;
    float2 ts = 1.0 / float2(color.get_width(), color.get_height());
    float3 c = color.sample(sLinearClamp, uv, level(0)).rgb;
    int debug = int(u.misc.z + 0.5);
    if (debug >= 1 && debug <= 4) return float4(c, 1.0);
    float sharp = u.postParams2.w;
    if (sharp > 0.0) {
        float3 n = color.sample(sLinearClamp, uv + float2(0, -ts.y), level(0)).rgb;
        float3 s = color.sample(sLinearClamp, uv + float2(0, ts.y), level(0)).rgb;
        float3 e = color.sample(sLinearClamp, uv + float2(ts.x, 0), level(0)).rgb;
        float3 w = color.sample(sLinearClamp, uv + float2(-ts.x, 0), level(0)).rgb;
        float3 mn = min(c, min(min(n, s), min(e, w)));
        float3 mx = max(c, max(max(n, s), max(e, w)));
        float3 blur = (n + s + e + w) * 0.25;
        c = clamp(c + (c - blur) * sharp * 1.6, mn, mx);
    }
    float3 b = bloom.sample(sLinearClamp, uv, level(0)).rgb * (1.0 / 6.0);
    c = mix(c, b, u.postParams.y);
    if (u.weather.x + u.snow.x > 0.001) {
        float4 rn = rainTex.sample(sLinearClamp, uv, level(0));
        c = c * (1.0 - rn.a) + rn.rgb;
    }
    // low-light (scotopic) vision: very dark areas lose colour and shift blue
    float la = luminance(c);
    c = mix(c, la * float3(0.62, 0.86, 1.35), smoothstep(2e-4, 1e-5, la) * 0.75);
    c *= expo[0];
    float3 t = u.postParams.z < 0.5 ? agxTonemap(c) : acesFitted(c);
    float3 dsp = pow(max(t, 0.0), float3(1.0 / 2.2));
    float l = luminance(dsp);
    dsp = l + (dsp - l) * u.postParams2.x;
    dsp = (dsp - 0.5) * u.postParams2.y + 0.5;
    float aspect = u.screenSize.x * u.screenSize.w;
    float2 q = (uv - 0.5) * float2(aspect, 1.0);
    dsp *= 1.0 - u.postParams.w * smoothstep(0.15, 1.1, dot(q, q) * 1.2);
    float grain = hash12(in.position.xy + fract(u.cameraPos.w * 13.7) * 1000.0) - 0.5;
    dsp += grain * u.postParams2.z * 0.035 * (1.0 - l * 0.5);
    return float4(pow(saturate(dsp), float3(2.2)), 1.0);
}
