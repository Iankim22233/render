// Function constant: the planar-reflection pass clips everything below the water surface.
constant bool kReflectFC [[function_constant(0)]];
constant bool kReflect = is_function_constant_defined(kReflectFC) && kReflectFC;

// ================================================================ TERRAIN

struct TerrainVOut {
    float4 position [[position]];
    float3 worldPos;
};

constant uint2 kQuadCorner[6] = { uint2(0, 0), uint2(1, 0), uint2(0, 1), uint2(1, 0), uint2(1, 1), uint2(0, 1) };

vertex TerrainVOut terrainVS(uint vid [[vertex_id]], uint iid [[instance_id]],
                             device const float4* chunks [[buffer(0)]],
                             constant FrameUniforms& u [[buffer(2)]],
                             constant float4x4& vp [[buffer(3)]],
                             texture2d<float> heightTex [[texture(0)]]) {
    float4 ch = chunks[iid];
    uint res = uint(ch.w);
    uint quad = vid / 6u;
    uint2 o = kQuadCorner[vid % 6u];
    float2 local;
    float skirt = 0.0;
    uint mainCount = res * res;
    if (quad < mainCount) {
        uint qx = quad % res, qz = quad / res;
        local = float2(float(qx + o.x), float(qz + o.y)) / float(res);
    } else {
        uint s = quad - mainCount;
        uint side = s / res, i = s % res;
        float a = float(i + o.x) / float(res);
        skirt = float(o.y);
        if (side == 0u) local = float2(a, 0.0);
        else if (side == 1u) local = float2(a, 1.0);
        else if (side == 2u) local = float2(0.0, a);
        else local = float2(1.0, a);
    }
    float2 xz = ch.xy + local * ch.z;
    float h = heightTex.sample(sLinearClamp, heightUV(xz, u), level(0)).r;
    h -= skirt * (0.5 + ch.z * 0.03);
    TerrainVOut out;
    out.worldPos = float3(xz.x, h, xz.y);
    out.position = vp * float4(out.worldPos, 1.0);
    return out;
}

// --- Hex-tiling (Mikkelsen 2022) to hide texture repetition
inline void hexGrid(float2 st, thread float3& W, thread int2& v1, thread int2& v2, thread int2& v3) {
    float2 s = st * 3.4641016;
    float2 skewed = float2(s.x - 0.57735027 * s.y, 1.15470054 * s.y);
    int2 baseId = int2(floor(skewed));
    float3 temp = float3(fract(skewed), 0.0);
    temp.z = 1.0 - temp.x - temp.y;
    float sg = step(0.0, -temp.z);
    float s2 = 2.0 * sg - 1.0;
    W = float3(-temp.z * s2, sg - temp.y * s2, sg - temp.x * s2);
    int isg = int(sg);
    v1 = baseId + int2(isg, isg);
    v2 = baseId + int2(isg, 1 - isg);
    v3 = baseId + int2(1 - isg, isg);
}

struct MatSample { float4 alb; float4 ext; };

inline MatSample sampleHex(texture2d_array<float> A, texture2d_array<float> E, sampler s, float2 uv, uint layer, float2 dx, float2 dy) {
    float3 W;
    int2 v0, v1, v2;
    hexGrid(uv, W, v0, v1, v2);
    int2 vs[3] = { v0, v1, v2 };
    float4 a[3], e[3];
    for (int i = 0; i < 3; i++) {
        float2 fv = float2(vs[i]);
        float ang = hash12(fv * 1.37 + 17.3) * 2.0 * PI;
        float cs = cos(ang), sn = sin(ang);
        float2 c = float2(fv.x + 0.5 * fv.y, 0.8660254 * fv.y) / 3.4641016;
        float2 d = uv - c;
        float2 r = float2(cs * d.x - sn * d.y, sn * d.x + cs * d.y) + c + hash22(fv + 0.31);
        float2 rdx = float2(cs * dx.x - sn * dx.y, sn * dx.x + cs * dx.y);
        float2 rdy = float2(cs * dy.x - sn * dy.y, sn * dy.x + cs * dy.y);
        a[i] = A.sample(s, r, layer, gradient2d(rdx, rdy));
        e[i] = E.sample(s, r, layer, gradient2d(rdx, rdy));
        float2 tn = e[i].xy * 2.0 - 1.0;
        tn = float2(cs * tn.x + sn * tn.y, -sn * tn.x + cs * tn.y);
        e[i].xy = tn * 0.5 + 0.5;
    }
    float3 w = pow(max(W, 0.0), float3(5.0)) * (float3(a[0].a, a[1].a, a[2].a) + 0.35);
    w /= max(w.x + w.y + w.z, 1e-5);
    MatSample o;
    o.alb = a[0] * w.x + a[1] * w.y + a[2] * w.z;
    o.ext = e[0] * w.x + e[1] * w.y + e[2] * w.z;
    return o;
}

fragment GBufferOut terrainFS(TerrainVOut in [[stage_in]],
                              constant FrameUniforms& u [[buffer(2)]],
                              texture2d_array<float> gA [[texture(0)]],
                              texture2d_array<float> gE [[texture(1)]],
                              texture2d<float> heightTex [[texture(2)]],
                              texture2d<float> canopyTex [[texture(3)]],
                              texture2d<float> lakeTex [[texture(4)]],
                              sampler ms [[sampler(0)]]) {
    float3 wp = in.worldPos;
    if (kReflect && wp.y < u.water.x - 0.05) discard_fragment();
    float texel = u.terrain.x / (u.terrain.z - 1.0);
    float hL = heightTex.sample(sLinearClamp, heightUV(wp.xz - float2(texel, 0), u), level(0)).r;
    float hR = heightTex.sample(sLinearClamp, heightUV(wp.xz + float2(texel, 0), u), level(0)).r;
    float hD = heightTex.sample(sLinearClamp, heightUV(wp.xz - float2(0, texel), u), level(0)).r;
    float hU = heightTex.sample(sLinearClamp, heightUV(wp.xz + float2(0, texel), u), level(0)).r;
    float3 Nm = normalize(float3(hL - hR, 2.0 * texel, hD - hU));

    float4 can = canopyTex.sample(sLinearClamp, wp.xz * u.terrain.y + 0.5, level(0));
    float slope = 1.0 - Nm.y;
    float m1 = fbm2(wp.xz * 0.035, 4);
    float m2 = fbm2(wp.xz * 0.11 + 31.0, 3);
    float m3 = fbm2(wp.xz * 0.5 + 7.0, 2);

    float w[6];
    w[0] = can.b * (0.45 + 0.9 * m1);
    float litter = (0.3 + can.r) * (1.0 - can.b * 0.75) * (0.8 + 0.4 * m3);
    w[1] = litter * (1.0 - can.a);
    w[2] = litter * can.a;
    w[3] = u.sceneParams.w * smoothstep(0.42, 0.72, m2) * (0.3 + can.r) * 0.95;
    w[4] = smoothstep(0.3, 0.5, slope + (m2 - 0.5) * 0.2);
    w[5] = 0.3 * smoothstep(0.55, 0.8, 1.0 - m1) * (1.0 - can.b);
    float wsum = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i != 4) w[i] *= (1.0 - w[4]);
        wsum += w[i];
    }
    for (int i = 0; i < 6; i++) w[i] /= max(wsum, 1e-4);

    float2 uv = wp.xz / 2.2;
    float2 dx = dfdx(uv), dy = dfdy(uv);
    float hgt[6];
    MatSample smp[6];
    float maxH = -10.0;
    for (int i = 0; i < 6; i++) {
        hgt[i] = -10.0;
        if (w[i] > 0.015) {
            float sc = (i == 4) ? 0.55 : 1.0;
            smp[i] = sampleHex(gA, gE, ms, uv * sc + float2(float(i) * 0.37, float(i) * 0.71), uint(i), dx * sc, dy * sc);
            hgt[i] = smp[i].alb.a * 0.6 + w[i];
            maxH = max(maxH, hgt[i]);
        }
    }
    float4 alb = 0.0, ext = 0.0;
    float bsum = 0.0;
    for (int i = 0; i < 6; i++) {
        if (hgt[i] > -5.0) {
            float b = max(hgt[i] - (maxH - 0.22), 0.0);
            alb += smp[i].alb * b;
            ext += smp[i].ext * b;
            bsum += b;
        }
    }
    alb /= max(bsum, 1e-5);
    ext /= max(bsum, 1e-5);

    float3 albedo = alb.rgb;
    albedo *= mix(0.8, 1.15, m1) * mix(0.9, 1.08, m3);
    albedo *= mix(float3(1.0), u.colorGrass.rgb, saturate(w[0] * 1.5));

    float3 T = normalize(float3(1, 0, 0) - Nm * Nm.x);
    float3 B = normalize(float3(0, 0, 1) - Nm * Nm.z);
    float2 tn = (ext.xy * 2.0 - 1.0) * 1.2;
    float3 N = normalize(T * tn.x + B * tn.y + Nm * sqrt(saturate(1.0 - dot(tn, tn))));
    float rough = clamp(ext.z, 0.3, 1.0);
    float spec = 0.5;

    // lake bed & wet shoreline
    float under = 0.0;
    if (u.water.y > 0.5) {
        float lm = lakeMaskAt(wp.xz, lakeTex, u);
        float near = max(max(lakeMaskAt(wp.xz + float2(9, 0), lakeTex, u), lakeMaskAt(wp.xz - float2(9, 0), lakeTex, u)),
                         max(lakeMaskAt(wp.xz + float2(0, 9), lakeTex, u), lakeMaskAt(wp.xz - float2(0, 9), lakeTex, u)));
        near = max(near, lm);
        under = smoothstep(u.water.x + 0.15, u.water.x - 0.35, wp.y) * lm;
        float shoreWet = smoothstep(u.water.x + 1.0, u.water.x + 0.05, wp.y) * near * (1.0 - under);
        float3 bed = mix(float3(0.07, 0.06, 0.04), float3(0.17, 0.15, 0.11), smoothstep(0.3, 0.7, m3)) * mix(0.85, 1.1, m1);
        albedo = mix(albedo, bed, under);
        N = normalize(mix(N, Nm, under * 0.6));
        albedo *= mix(1.0, 0.5, shoreWet);
        rough = mix(rough, 0.22, shoreWet);
    }
    // rain: darkened porous surfaces, lower roughness, puddles with ripples in flat spots
    float wet = u.weather.y;
    if (wet > 0.001 && under < 0.99) {
        float porous = 1.0 - w[4] * 0.5;
        albedo *= mix(1.0, 0.6, wet * porous);
        rough = mix(rough, 0.5, wet * 0.6);
        float flat = smoothstep(0.985, 0.998, Nm.y);
        float pnz = fbm2(wp.xz * 0.11 + 5.0, 3) + (m3 - 0.5) * 0.2;
        float puddle = smoothstep(0.67, 0.71, pnz + wet * 0.06 - 0.06) * flat * wet * (1.0 - can.b * 0.9) * (1.0 - under);
        if (puddle > 0.001) {
            float2 rip = rainRipples(wp.xz, u.wind.w, u.weather.x);
            float3 pn3 = normalize(float3(rip.x, 1.0, rip.y));
            N = normalize(mix(N, pn3, puddle));
            albedo *= mix(1.0, 0.4, puddle);
            rough = mix(rough, 0.03, puddle);
            spec = mix(spec, 0.5, puddle);
        }
    }
    // snow: lies on flatter ground first, thinner under dense canopy (trees intercept it), sparkles
    float cover = u.snow.y * (1.0 - can.r * 0.45) * (1.0 - under);
    if (cover > 0.001) {
        float sm = snowCoverMask(Nm.y, wp, cover, -0.12);
        float3 sn = normalize(Nm + float3(fbm2(wp.xz * 3.1, 2) - 0.5, 0.0, fbm2(wp.xz * 3.1 + 9.7, 2) - 0.5) * 0.3);
        float3 cell = floor(wp * 90.0);
        bool glint = hash13(cell) > 0.9965;
        if (glint) sn = normalize(sn + (hash33(cell + 3.1) - 0.5) * 1.6);
        N = normalize(mix(N, sn, sm));
        albedo = mix(albedo, kSnowAlbedo, sm);
        rough = mix(rough, glint ? 0.06 : 0.62, sm);
        spec = mix(spec, glint ? 1.0 : 0.5, sm);
    }

    GBufferOut o;
    o.albedo = float4(albedo, ext.w);
    o.normal = float4(N, rough);
    o.material = float4(0.0, 0.0, spec, 1.0);
    return o;
}

// ================================================================ TREES / PLANTS / ROCKS

struct TreeVOut {
    float4 position [[position, invariant]];
    float3 worldPos;
    float3 normal;
    float3 tangent;
    float2 uv;
    float4 extra;
    float4 inst;      // x tint seed, y lod fade, z scale
    float3 localPos;
};

inline float3 applyWind(float3 wp, float3 basePos, float flex, float phase, float instPhase, float leaf, float3 nrm, constant FrameUniforms& u) {
    float t = u.wind.w;
    float strength = u.wind.z;
    float2 dir = u.wind.xy;
    float gust = 0.55 + 0.45 * sin(t * 0.37 + dot(basePos.xz, dir) * 0.015) * sin(t * 0.23 + basePos.x * 0.01 + 1.3);
    float sway = (0.65 * gust + 0.35 * sin(t * 0.85 + instPhase)) * strength;
    float3 off = float3(dir.x, 0.0, dir.y) * sway * flex * 1.2;
    float br = sin(t * 1.9 + phase * 6.283 + instPhase) * 0.5 + sin(t * 3.3 + phase * 12.0) * 0.25;
    off += float3(dir.x * 0.4, 0.3, dir.y * 0.4) * br * strength * flex * 0.55;
    off += nrm * leaf * sin(t * 7.0 + phase * 31.0 + dot(wp, float3(1.3))) * 0.03 * strength * min(flex * 5.0, 1.0);
    return wp + off;
}

inline TreeVOut treeTransform(TreeVertex v, TreeInstance inst, constant FrameUniforms& u, float4x4 vp, float leaf) {
    float c = cos(inst.params.x), s = sin(inst.params.x);
    float sc = inst.posScale.w;
    float3 lp = v.position * sc;
    float3 n = rotY(v.normal, c, s);
    float3 wp = rotY(lp, c, s) + inst.posScale.xyz;
    wp = applyWind(wp, inst.posScale.xyz, v.extra.x * sc, v.extra.y, inst.params.z, leaf, n, u);
    TreeVOut o;
    o.worldPos = wp;
    o.position = vp * float4(wp, 1.0);
    o.normal = n;
    o.tangent = rotY(v.tangent, c, s);
    o.uv = v.uv;
    o.extra = v.extra;
    o.inst = float4(inst.params.y, inst.params.w, sc, 0.0);
    o.localPos = lp;
    return o;
}

vertex TreeVOut barkVS(uint vid [[vertex_id]], uint iid [[instance_id]],
                       device const TreeVertex* verts [[buffer(0)]],
                       device const TreeInstance* insts [[buffer(1)]],
                       constant FrameUniforms& u [[buffer(2)]],
                       constant float4x4& vp [[buffer(3)]]) {
    return treeTransform(verts[vid], insts[iid], u, vp, 0.0);
}

vertex TreeVOut leafVS(uint vid [[vertex_id]], uint iid [[instance_id]],
                       device const TreeVertex* verts [[buffer(0)]],
                       device const TreeInstance* insts [[buffer(1)]],
                       constant FrameUniforms& u [[buffer(2)]],
                       constant float4x4& vp [[buffer(3)]]) {
    return treeTransform(verts[vid], insts[iid], u, vp, 1.0);
}

inline void lodFadeDiscard(float fade, float2 px, float frame) {
    if (fade >= 0.999) return;
    float d = ign(px, frame);
    if (fade > 0.0) { if (d >= fade) discard_fragment(); }
    else { if (d < -fade) discard_fragment(); }
}

fragment GBufferOut barkFS(TreeVOut in [[stage_in]],
                           constant FrameUniforms& u [[buffer(2)]],
                           texture2d_array<float> bA [[texture(0)]],
                           texture2d_array<float> bE [[texture(1)]],
                           texture2d_array<float> gA [[texture(2)]],
                           sampler ms [[sampler(0)]]) {
    if (kReflect && in.worldPos.y < u.water.x - 0.05) discard_fragment();
    uint layer = uint(in.extra.z + 0.5);
    float3 gN = normalize(in.normal);
    float3 N = gN;
    float4 alb, ext;
    if (layer == 6u) {
        float3 p = in.worldPos * 0.42;
        float3 bw = pow(abs(gN), float3(4.0));
        bw /= (bw.x + bw.y + bw.z);
        float4 ax = bA.sample(ms, p.zy, 6), ay = bA.sample(ms, p.xz, 6), az = bA.sample(ms, p.xy, 6);
        float4 ex = bE.sample(ms, p.zy, 6), ey = bE.sample(ms, p.xz, 6), ez = bE.sample(ms, p.xy, 6);
        alb = ax * bw.x + ay * bw.y + az * bw.z;
        ext = ex * bw.x + ey * bw.y + ez * bw.z;
        float2 tx = ex.xy * 2.0 - 1.0, ty = ey.xy * 2.0 - 1.0, tz = ez.xy * 2.0 - 1.0;
        N = normalize(gN + (float3(0.0, tx.y, tx.x) * bw.x + float3(ty.x, 0.0, ty.y) * bw.y + float3(tz.x, tz.y, 0.0) * bw.z) * 1.2);
    } else {
        alb = bA.sample(ms, in.uv, layer);
        ext = bE.sample(ms, in.uv, layer);
        float3 T = normalize(in.tangent - gN * dot(in.tangent, gN));
        float3 B = cross(gN, T);
        float2 tn = (ext.xy * 2.0 - 1.0) * 1.3;
        N = normalize(T * tn.x + B * tn.y + gN * sqrt(saturate(1.0 - dot(tn, tn))));
    }
    // moss on upward faces, on the north side and near the ground
    float mossAmt = u.sceneParams.w;
    float mn = fbm2(in.worldPos.xz * 0.9 + in.worldPos.y * 0.7, 3);
    float mossMask;
    if (layer == 6u) {
        mossMask = smoothstep(0.15, 0.7, gN.y + (mn - 0.5) * 0.6);
    } else {
        float low = 1.0 - smoothstep(0.2, 1.4 + mn * 1.5, in.localPos.y);
        float north = saturate(0.4 - gN.z * 0.6);
        mossMask = max(low * north, smoothstep(0.45, 0.85, gN.y) * 0.9);
    }
    mossMask = saturate(mossMask * mossAmt * smoothstep(0.25, 0.6, mn + 0.2));
    float4 moss = gA.sample(ms, in.worldPos.xz * 0.6 + in.worldPos.y * 0.3, 3);
    float tint = mix(0.82, 1.12, in.inst.x);
    float3 albedo = mix(alb.rgb * tint, moss.rgb * 1.1, mossMask);
    float wet = u.weather.y;
    albedo *= mix(1.0, 0.6, wet);
    float snowM = 0.0;
    if (u.snow.y > 0.001) {
        snowM = snowCoverMask(gN.y, in.worldPos, u.snow.y, layer == 6u ? -0.15 : 0.05);
        // snow plastered on the windward side of trunks
        float windward = saturate(dot(gN.xz, -u.wind.xy)) * 0.55 * u.snow.y;
        snowM = max(snowM, windward * smoothstep(0.35, 0.65, fbm2(in.worldPos.xz * 2.0 + in.worldPos.y * 1.3, 2)));
        albedo = mix(albedo, kSnowAlbedo, snowM);
    }
    GBufferOut o;
    o.albedo = float4(albedo, ext.w * in.extra.w);
    o.normal = float4(normalize(mix(N, gN, mossMask * 0.5)), mix(ext.z, 0.95, mossMask) * mix(1.0, 0.45, wet));
    o.normal.w = mix(o.normal.w, 0.62, snowM);
    o.material = float4(mossMask * 0.15 * (1.0 - snowM), 1.0 / 3.0, 0.5, 1.0);
    return o;
}

inline float3 autumnColor(uint layer, float v, float inst) {
    float h = fract(v * 7.31 + inst * 3.7);
    if (layer == 3u) { // birch: yellows
        return mix(float3(0.42, 0.3, 0.025), float3(0.55, 0.42, 0.06), h);
    }
    if (layer == 5u) { // beech: copper / orange
        return mix(float3(0.4, 0.14, 0.02), float3(0.5, 0.28, 0.04), h);
    }
    // oak & others: brown/orange/red mix
    float3 a = float3(0.38, 0.1, 0.02), b = float3(0.45, 0.25, 0.04), c = float3(0.22, 0.12, 0.04);
    return h < 0.5 ? mix(a, b, h * 2.0) : mix(b, c, (h - 0.5) * 2.0);
}

fragment GBufferOut leafFS(TreeVOut in [[stage_in]],
                           constant FrameUniforms& u [[buffer(2)]],
                           texture2d_array<float> lA [[texture(0)]],
                           texture2d_array<float> lE [[texture(1)]],
                           sampler ms [[sampler(0)]]) {
    uint layer = uint(in.extra.z + 0.5);
    float4 a = lA.sample(ms, in.uv, layer);
    float4 e = lE.sample(ms, in.uv, layer);
    float3 N = normalize(in.normal);
    float3 T = normalize(in.tangent - N * dot(in.tangent, N));
    float3 B = cross(N, T);
    float2 tn = e.xy * 2.0 - 1.0;
    N = normalize(N + (T * tn.x + B * tn.y) * 0.8);
    float3 rv = hash33(float3(e.z * 91.7, in.inst.x * 13.1, in.extra.y * 7.0)) * 2.0 - 1.0;
    N = normalize(N + rv * 0.35);

    float3 albedo = a.rgb;
    float tint = mix(0.8, 1.15, in.inst.x);
    albedo *= tint;
    albedo.r *= mix(0.9, 1.15, fract(in.inst.x * 7.7));
    bool conifer = (layer == 0u || layer == 1u || layer == 6u);
    float autumn = u.sceneParams.x * (layer == 4u ? 0.6 : 1.0);
    if (!conifer && autumn > 0.0) {
        float r = fract(e.z * 3.17 + in.inst.x * 1.9);
        float amt = saturate(autumn * 1.35 - r * 0.55 - (1.0 - in.extra.w) * 0.25);
        float3 ac = autumnColor(layer, e.z, in.inst.x) * mix(0.75, 1.1, fract(e.z * 13.3));
        albedo = mix(albedo, ac, amt);
    }
    float wet = u.weather.y;
    albedo *= mix(1.0, 0.85, wet);
    float snowM = 0.0;
    if (u.snow.y > 0.001) {
        float up = normalize(in.normal).y * 0.7 + N.y * 0.3;
        snowM = snowCoverMask(up, in.worldPos, u.snow.y * (conifer ? 1.0 : 0.7), conifer ? -0.08 : 0.12) * (0.55 + 0.45 * fract(e.z * 5.3));
        albedo = mix(albedo, kSnowAlbedo, snowM);
    }
    GBufferOut o;
    o.albedo = float4(albedo, in.extra.w);
    o.normal = float4(N, mix(mix(conifer ? 0.6 : 0.5, 0.22, wet), 0.62, snowM));
    o.material = float4(saturate(e.w * (conifer ? 0.7 : 1.0)) * (1.0 - snowM), 2.0 / 3.0, mix(conifer ? 0.4 : 0.55, 0.85, wet), 1.0);
    return o;
}

// Depth prepass for foliage: alpha test + LOD cross-fade dither, depth only.
fragment void leafDepthFS(TreeVOut in [[stage_in]],
                          constant FrameUniforms& u [[buffer(2)]],
                          texture2d_array<float> lA [[texture(0)]],
                          sampler ms [[sampler(0)]]) {
    if (kReflect && in.worldPos.y < u.water.x - 0.05) discard_fragment();
    uint layer = uint(in.extra.z + 0.5);
    if (lA.sample(ms, in.uv, layer).a < 0.5) discard_fragment();
    lodFadeDiscard(in.inst.y, in.position.xy, u.jitter.z);
}

// Shadow pass: alpha-tested foliage
fragment void leafShadowFS(TreeVOut in [[stage_in]],
                           texture2d_array<float> lA [[texture(0)]]) {
    uint layer = uint(in.extra.z + 0.5);
    if (lA.sample(sLinearRepeat, in.uv, layer, bias(0.5)).a < 0.5) discard_fragment();
}

// ================================================================ GRASS

struct GrassVOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float t;
    float seed;
    float dry;
};

vertex GrassVOut grassVS(uint vid [[vertex_id]], uint iid [[instance_id]],
                         constant FrameUniforms& u [[buffer(2)]],
                         constant float4x4& vp [[buffer(3)]],
                         constant float4* ring [[buffer(4)]],
                         texture2d<float> heightTex [[texture(0)]],
                         texture2d<float> canopyTex [[texture(1)]]) {
    GrassVOut o;
    o.position = float4(2.0, 2.0, 2.0, 1.0);
    o.worldPos = 0.0; o.normal = float3(0, 1, 0); o.t = 0.0; o.seed = 0.0; o.dry = 0.0;

    float innerR = ring[0].x, outerR = ring[0].y, spacing = ring[0].z, widthMul = ring[0].w;
    uint gridN = uint(ring[1].x);
    float band = ring[1].z;
    uint ix = iid % gridN, iz = iid / gridN;
    float3 cam = u.cameraPos.xyz;
    int2 baseCell = int2(floor(cam.xz / spacing)) - int2(int(gridN / 2u));
    int2 cell = baseCell + int2(int(ix), int(iz));
    uint h = pcgHash(uint(cell.x) * 1973u + uint(cell.y) * 9277u + uint(spacing * 1000.0) * 26699u);
    float r0 = hashU(h), r1 = hashU(h ^ 0x9e3779b9u), r2 = hashU(h * 747796405u + 1u), r3 = hashU(h + 0x68bc21ebu);
    float r4 = hashU(h ^ 0x02e5be93u), r5 = hashU(h * 3u + 0x967a889bu), r6 = hashU(h ^ 0x51ed270bu);
    float2 pos = (float2(cell) + float2(r0, r1)) * spacing;
    float dist = length(pos - cam.xz);
    if (dist > outerR) return o;
    float keep = (innerR > 0.0 ? smoothstep(innerR, innerR + band, dist) : 1.0) * (1.0 - smoothstep(outerR - band, outerR, dist));
    float4 can = canopyTex.sample(sLinearClamp, pos * u.terrain.y + 0.5, level(0));
    float density = saturate(can.b * 1.15);
    if (r2 > density * keep) return o;
    float groundH = heightTex.sample(sLinearClamp, heightUV(pos, u), level(0)).r;
    float bladeH = u.grassParams.z * mix(0.4, 1.15, r3) * (0.55 + 0.6 * density);
    float reed = 0.0;
    if (u.water.y > 0.5) {
        reed = smoothstep(u.water.x + 0.7, u.water.x + 0.25, groundH) * step(u.water.x - 1.0, groundH);
        bladeH = mix(bladeH, mix(1.3, 2.3, r3) + max(u.water.x - groundH, 0.0), reed);
    }
    // deep snow buries most grass; a few dry stalks still poke through
    float buried = saturate(u.snow.y * (1.0 - can.r * 0.45) * 1.1) * (1.0 - reed * 0.6);
    if (fract(r6 * 13.7) > saturate(1.04 - buried * 1.15)) return o;
    bladeH *= 1.0 - buried * 0.55;
    float4 cp = vp * float4(pos.x, groundH + bladeH * 0.5, pos.y, 1.0);
    float m = bladeH + 0.5;
    if (cp.w < -m || abs(cp.x) > cp.w + m * 2.0 || abs(cp.y) > cp.w + m * 2.0) return o;

    float width = 0.03 * widthMul * mix(0.75, 1.35, r4) * mix(1.0, 0.75, reed);
    float facing = r5 * 2.0 * PI;
    float2 fdir = float2(cos(facing), sin(facing));
    float2 side = float2(-fdir.y, fdir.x);
    float leanA = r6 * 2.0 * PI;
    float2 leanDir = float2(cos(leanA), sin(leanA));
    float lean = mix(0.08, 0.5, fract(r3 * 7.3)) * mix(1.0, 0.35, reed);
    float t = u.wind.w, strength = u.wind.z;
    float gust = vnoise2(pos * 0.07 - u.wind.xy * t * 1.4);
    float2 windBend = u.wind.xy * (0.1 + gust * 0.9) * strength * 0.9 + u.wind.xy * 0.08 * sin(t * 2.6 + r0 * 30.0) * strength;

    uint seg = vid / 2u;
    float sideSign = (vid % 2u == 0u) ? -1.0 : 1.0;
    float tt = float(seg) / 4.0;
    if (vid >= 8u) { tt = 1.0; sideSign = 0.0; }
    float2 bend = (leanDir * lean + windBend) * tt * tt;
    float droop = dot(bend, bend);
    float3 p = float3(pos.x, groundH - 0.03, pos.y);
    p += float3(bend.x, 0.0, bend.y) * bladeH;
    p.y += bladeH * tt * (1.0 - 0.35 * min(droop, 1.0));
    float wdt = width * (1.0 - pow(tt, 1.3) * 0.9);
    p.xz += side * wdt * 0.5 * sideSign;

    float3 n = normalize(mix(float3(fdir.x, 0.0, fdir.y), float3(0.0, 1.0, 0.0), 0.45) + float3(side.x, 0.0, side.y) * sideSign * 0.35);
    o.position = vp * float4(p, 1.0);
    o.worldPos = p;
    o.normal = n;
    o.t = tt;
    o.seed = r4;
    o.dry = mix(r5, 0.99, reed * 0.6);
    return o;
}

fragment GBufferOut grassFS(GrassVOut in [[stage_in]], bool front [[front_facing]],
                            constant FrameUniforms& u [[buffer(2)]]) {
    float3 n = normalize(in.normal);
    if (!front) n = float3(-n.x, n.y, -n.z);
    float3 root = float3(0.025, 0.045, 0.01);
    float3 tip = float3(0.1, 0.16, 0.03) * mix(0.8, 1.2, in.seed);
    float3 col = mix(root, tip, smoothstep(0.0, 0.9, in.t));
    float dry = step(1.0 - u.grassParams.w * 0.7, fract(in.dry * 5.3));
    float3 dryCol = mix(float3(0.07, 0.06, 0.03), float3(0.3, 0.26, 0.13), smoothstep(0.0, 0.8, in.t));
    col = mix(col, dryCol, dry * 0.85);
    col *= u.colorGrass.rgb;
    float ao = mix(0.2, 1.0, pow(in.t, 0.65));
    float wet = u.weather.y;
    col *= mix(1.0, 0.82, wet);
    col = mix(col, float3(0.32, 0.26, 0.15), saturate(u.snow.y * 1.5) * 0.7);   // winter grass is dry and straw-coloured
    col = mix(col, kSnowAlbedo * 0.9, u.snow.y * smoothstep(0.3, 1.0, in.t) * 0.45);
    GBufferOut o;
    o.albedo = float4(col, ao);
    o.normal = float4(n, mix(0.8, 0.5, wet));
    o.material = float4(mix(0.6, 0.3, dry), 1.0, mix(0.25, 0.4, wet), 1.0);
    return o;
}

// ================================================================ IMPOSTORS (far-distance tree billboards)

struct ImpVOut {
    float4 position [[position]];
    float3 worldPos;
    float2 uv;
    float layer;
    float c;
    float s;
    float tint;
    float fade;
};

// imp[0] = (half width, center y, half height, layer), imp[1] = facing target (xyz; w = 1: xz is a direction, 0: a point)
vertex ImpVOut impostorVS(uint vid [[vertex_id]], uint iid [[instance_id]],
                          device const TreeInstance* insts [[buffer(1)]],
                          constant FrameUniforms& u [[buffer(2)]],
                          constant float4x4& vp [[buffer(3)]],
                          constant float4* imp [[buffer(4)]]) {
    TreeInstance inst = insts[iid];
    float2 corner = float2((vid & 1u) ? 1.0 : -1.0, (vid & 2u) ? 1.0 : -1.0);
    float3 pos = inst.posScale.xyz;
    float sc = inst.posScale.w;
    float2 to = imp[1].w > 0.5 ? imp[1].xz : (imp[1].xz - pos.xz);
    float2 dirn = normalize(to + float2(1e-5, 0.0));
    float s = dirn.x, c = dirn.y;
    float3 local = float3(corner.x * imp[0].x, imp[0].y + corner.y * imp[0].z, 0.0) * sc;
    float3 wp = rotY(local, c, s) + pos;
    float hfac = saturate(local.y / max(imp[0].y * 2.0 * sc, 1.0));
    float sway = (0.6 + 0.4 * sin(u.wind.w * 0.8 + inst.params.z)) * u.wind.z * 0.35;
    wp.xz += u.wind.xy * sway * hfac * hfac;
    ImpVOut o;
    o.position = vp * float4(wp, 1.0);
    o.worldPos = wp;
    o.uv = float2(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
    o.layer = imp[0].w;
    o.c = c;
    o.s = s;
    o.tint = inst.params.y;
    o.fade = inst.params.w;
    return o;
}

fragment GBufferOut impostorFS(ImpVOut in [[stage_in]],
                               constant FrameUniforms& u [[buffer(2)]],
                               texture2d_array<float> iA [[texture(0)]],
                               texture2d_array<float> iN [[texture(1)]],
                               texture2d_array<float> iM [[texture(2)]]) {
    if (kReflect && in.worldPos.y < u.water.x - 0.05) discard_fragment();
    uint layer = uint(in.layer + 0.5);
    float4 m = iM.sample(sLinearClamp, in.uv, layer);
    if (m.w < 0.4) discard_fragment();
    lodFadeDiscard(in.fade, in.position.xy, u.jitter.z);
    float inv = 1.0 / max(m.w, 0.05);
    float4 a = iA.sample(sLinearClamp, in.uv, layer) * inv;
    float4 nr = iN.sample(sLinearClamp, in.uv, layer);
    float3 n = normalize(nr.xyz + float3(0.0, 1e-4, 0.0));
    n = rotY(n, in.c, in.s);
    float wet = u.weather.y;
    GBufferOut o;
    o.albedo = float4(a.rgb * mix(0.85, 1.12, in.tint) * mix(1.0, 0.82, wet), saturate(a.a));
    o.normal = float4(n, saturate(nr.w * inv) * mix(1.0, 0.5, wet));
    o.material = float4(saturate(m.x * inv), 2.0 / 3.0, saturate(m.z * inv), 1.0);
    if (u.snow.y > 0.001) {
        float snowM = snowCoverMask(n.y, in.worldPos, u.snow.y * 0.9, -0.05);
        o.albedo.rgb = mix(o.albedo.rgb, kSnowAlbedo, snowM);
        o.material.x *= 1.0 - snowM;
    }
    return o;
}

fragment void impostorShadowFS(ImpVOut in [[stage_in]], texture2d_array<float> iM [[texture(2)]]) {
    if (iM.sample(sLinearClamp, in.uv, uint(in.layer + 0.5)).w < 0.4) discard_fragment();
}
