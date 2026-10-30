// ================================================================ ATMOSPHERE (single scattering + multi-scatter approx)

constant float kEarthR = 6360000.0;
constant float kAtmoR = 6460000.0;
constant float3 kRayleigh = float3(5.802e-6, 13.558e-6, 33.1e-6);
constant float kMieS = 3.996e-6;
constant float kMieE = 4.40e-6;
constant float3 kOzone = float3(0.650e-6, 1.881e-6, 0.085e-6);

inline float2 raySphere(float3 ro, float3 rd, float r) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float d = b * b - c;
    if (d < 0.0) return float2(-1.0, -1.0);
    float s = sqrt(d);
    return float2(-b - s, -b + s);
}

inline void atmoMedium(float h, float turb, thread float3& sR, thread float& sM, thread float3& ext) {
    float rD = exp(-h / 8000.0);
    float mD = exp(-h / 1200.0);
    float oD = max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);
    sR = kRayleigh * rD;
    sM = kMieS * turb * mD;
    ext = sR + kMieE * turb * mD + kOzone * oD;
}

inline float3 atmoSunTransmittance(float3 p, float3 L, float turb) {
    float2 g = raySphere(p, L, kEarthR);
    if (g.x > 0.0) return float3(0.0);
    float2 a = raySphere(p, L, kAtmoR);
    float tMax = a.y;
    const int N = 12;
    float3 od = 0.0;
    for (int i = 0; i < N; i++) {
        float f0 = float(i) / float(N), f1 = float(i + 1) / float(N);
        float t = tMax * (f0 * f0 + f1 * f1) * 0.5;
        float dt = tMax * (f1 * f1 - f0 * f0);
        float3 pp = p + L * t;
        float3 sR, ext; float sM;
        atmoMedium(length(pp) - kEarthR, turb, sR, sM, ext);
        od += ext * dt;
    }
    return exp(-od);
}

// Single scattering (+ crude multi-scattering) along a view ray for one light source of unit irradiance.
inline float3 skyScatter(float3 ro, float3 rd, float3 L, float turb, float tMax, bool hitGround, int N) {
    float cosT = dot(rd, L);
    float pR = 3.0 / (16.0 * PI) * (1.0 + cosT * cosT);
    const float g = 0.8;
    float pM = 3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + cosT * cosT)) / ((2.0 + g * g) * pow(max(1.0 + g * g - 2.0 * g * cosT, 1e-4), 1.5));
    float3 zenithSunT = atmoSunTransmittance(ro, normalize(float3(0.0, 1.0, 0.0) + L), turb);
    float3 Lsum = 0.0, T = 1.0;
    for (int i = 0; i < N; i++) {
        float f0 = float(i) / float(N), f1 = float(i + 1) / float(N);
        float t = tMax * (f0 * f0 + f1 * f1) * 0.5;
        float dt = tMax * (f1 * f1 - f0 * f0);
        float3 p = ro + rd * t;
        float3 sR, ext; float sM;
        atmoMedium(length(p) - kEarthR, turb, sR, sM, ext);
        float3 sunT = atmoSunTransmittance(p, L, turb);
        float3 S = (sR * pR + sM * pM) * sunT;
        // crude multiple-scattering term: isotropic, driven by how much light reaches the upper sky
        S += (sR + sM) * (0.06 / PI) * zenithSunT * saturate(L.y * 4.0 + 0.3);
        float3 Tr = exp(-ext * dt);
        Lsum += T * S * (1.0 - Tr) / max(ext, float3(1e-9));
        T *= Tr;
    }
    if (hitGround) {
        float3 p = ro + rd * tMax;
        float3 n = normalize(p);
        Lsum += T * atmoSunTransmittance(p + n * 20.0, L, turb) * max(dot(n, L), 0.0) * (0.1 / PI);
    }
    return Lsum;
}

kernel void skyLUTKernel(texture2d<float, access::write> outTex [[texture(0)]],
                         constant FrameUniforms& u [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint W = outTex.get_width(), H = outTex.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float2 uv = (float2(gid) + 0.5) / float2(W, H);
    float az = (uv.x - 0.5) * 2.0 * PI;
    float s = (0.5 - uv.y) * 2.0;
    float el = sign(s) * s * s * PI * 0.5;
    float3 rd = float3(cos(el) * cos(az), sin(el), cos(el) * sin(az));
    float turb = u.sceneParams.z;
    float3 ro = float3(0.0, kEarthR + 300.0, 0.0);
    float2 atm = raySphere(ro, rd, kAtmoR);
    float2 gnd = raySphere(ro, rd, kEarthR);
    bool hitGround = gnd.x > 0.0;
    float tMax = hitGround ? gnd.x : atm.y;
    // sunlight (incl. twilight) from the real sun, plus the dim blue moonlit sky
    float3 Lsum = skyScatter(ro, rd, u.skySun.xyz, turb, tMax, hitGround, 32);
    if (u.moonParams.z > 0.0 && u.moon.y > -0.1) {
        Lsum += skyScatter(ro, rd, u.moon.xyz, turb, tMax, hitGround, 16) * u.moonParams.z;
    }
    // overcast: grey, flatter, dimmer sky
    float ov = u.weather.z;
    float lum = dot(Lsum, float3(0.2126, 0.7152, 0.0722));
    Lsum = mix(Lsum, float3(lum) * float3(0.94, 0.98, 1.05) * 1.2, ov * 0.85) * mix(1.0, 0.75, ov);
    // night: faint airglow / starlight so moonless nights aren't pure black (brighter towards the horizon)
    Lsum += float3(0.22, 0.3, 0.55) * 1.0e-4 * u.snow.w * (0.5 + 0.5 * u.snow.z) * (1.0 + 1.2 * (1.0 - saturate(rd.y)));
    outTex.write(float4(Lsum, 1.0), gid);
}

kernel void skySHKernel(texture2d<float> lut [[texture(0)]],
                        device float4* shOut [[buffer(0)]],
                        constant FrameUniforms& u [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const int NT = 24, NP = 48;
    float dw = 4.0 * PI / float(NT * NP);
    float3 zen = sampleSky(lut, float3(0.0, 1.0, 0.0));
    float3 groundRad = float3(0.07, 0.08, 0.045) / PI * (u.sunColor.rgb * max(u.sunDir.y, 0.0) * 0.35 + zen * PI * 0.7);
    float3 c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0, avg = 0.0;
    float navg = 0.0;
    for (int i = 0; i < NT; i++) {
        float cosT = 1.0 - 2.0 * (float(i) + 0.5) / float(NT);
        float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
        for (int j = 0; j < NP; j++) {
            float phi = 2.0 * PI * (float(j) + 0.5) / float(NP);
            float3 d = float3(sinT * cos(phi), cosT, sinT * sin(phi));
            float3 L = d.y > 0.0 ? sampleSky(lut, d) : groundRad;
            c0 += L * 0.282095 * dw;
            c1 += L * 0.488603 * d.y * dw;
            c2 += L * 0.488603 * d.z * dw;
            c3 += L * 0.488603 * d.x * dw;
            if (d.y > 0.0) { avg += L; navg += 1.0; }
        }
    }
    shOut[0] = float4(c0, 0.0);
    shOut[1] = float4(c1, 0.0);
    shOut[2] = float4(c2, 0.0);
    shOut[3] = float4(c3, 0.0);
    shOut[4] = float4(avg / max(navg, 1.0), 0.0);
}

// ================================================================ CLOUDS (2D layer, lit with forward scattering)

inline float cloudDensityAt(float2 q, float cov) {
    float n = fbm2(q, 6);
    return smoothstep(1.0 - cov - 0.08, 1.0 - cov + 0.3, n);
}

inline float4 cloudLayer(float3 dir, constant FrameUniforms& u, texture2d<float> lut) {
    float cov = u.sceneParams.y;
    if (u.misc.y < 0.5 || dir.y <= 0.012 || cov <= 0.001) return float4(0.0);
    float t = (1800.0 - u.cameraPos.y) / dir.y;
    float2 p = u.cameraPos.xz + dir.xz * t;
    float2 q = p * 0.00045 + u.wind.xy * u.wind.w * 0.003;
    float dens = cloudDensityAt(q, cov);
    if (dens <= 0.001) return float4(0.0);
    float2 sdir = normalize(u.sunDir.xz + float2(1e-4));
    float sd = cloudDensityAt(q + sdir * 0.035, cov) * 0.6 + cloudDensityAt(q + sdir * 0.08, cov) * 0.4;
    float cosT = dot(dir, u.sunDir.xyz);
    float phase = (mix(phaseHG(cosT, 0.65), phaseHG(cosT, -0.15), 0.5)) * 4.0 * PI;
    float ov = u.weather.z;
    float3 sunLit = u.sunColor.rgb * exp(-sd * 2.8) * (0.25 + 0.2 * phase) * mix(1.0, 0.6, dens) * (1.0 - ov * 0.7);
    float3 amb = sampleSky(lut, float3(0.0, 1.0, 0.0)) * mix(2.8, 1.6, ov) * mix(1.0, 0.75, dens) * (1.0 - ov * 0.35 * dens);
    float alpha = dens * smoothstep(0.012, 0.15, dir.y);
    return float4((sunLit + amb) * alpha, alpha);
}

inline float cloudShadow(float3 P, constant FrameUniforms& u) {
    float cov = u.sceneParams.y;
    if (u.misc.y < 0.5 || cov <= 0.001 || u.sunDir.y <= 0.02) return 1.0;
    float t = (1800.0 - P.y) / u.sunDir.y;
    float2 p = P.xz + u.sunDir.xz * t;
    float2 q = p * 0.00045 + u.wind.xy * u.wind.w * 0.003;
    return 1.0 - cloudDensityAt(q, cov) * u.misc.w;
}

// Procedural star field (two layers on a cube-face grid) + Milky Way, sized to ~1 pixel.
inline float3 starField(float3 dir, constant FrameUniforms& u) {
    float amt = u.snow.z * u.snow.w;
    if (amt <= 0.001 || dir.y < -0.02) return float3(0.0);
    float pix = 2.0 / (u.proj[1][1] * u.screenSize.y);
    float3 a = abs(dir);
    float2 fuv;
    float face;
    if (a.x >= a.y && a.x >= a.z) { fuv = dir.yz / a.x; face = dir.x > 0.0 ? 0.0 : 1.0; }
    else if (a.y >= a.z) { fuv = dir.xz / a.y; face = dir.y > 0.0 ? 2.0 : 3.0; }
    else { fuv = dir.xy / a.z; face = dir.z > 0.0 ? 4.0 : 5.0; }
    float3 col = 0.0;
    for (int layer = 0; layer < 3; layer++) {
        float N = layer == 0 ? 70.0 : (layer == 1 ? 180.0 : 420.0);
        float2 g = (fuv * 0.5 + 0.5) * N;
        float2 cell = floor(g);
        float2 key = cell + face * 1013.0 + float(layer) * 517.0;
        float exist = hash12(key + 0.37);
        float prob = (layer == 0 ? 0.3 : (layer == 1 ? 0.1 : 0.035)) * saturate(u.snow.z * 1.3);
        if (exist > prob) continue;
        float2 h = hash22(key);
        float2 sp = cell + 0.15 + 0.7 * h;
        float d = length(g - sp) / N * 2.0;
        float sigma = max(pix * 0.7, 1e-5);
        float mag = pow(hash12(key + 3.3), 7.0) * (layer == 0 ? 1.0 : (layer == 1 ? 0.3 : 0.08)) + (layer == 0 ? 0.02 : 0.004);
        float twinkle = 0.7 + 0.3 * sin(u.wind.w * (2.0 + h.x * 6.0) + h.y * 40.0);
        float temp = hash12(key + 9.1);
        float3 tint = temp < 0.5 ? mix(float3(0.65, 0.78, 1.0), float3(1.0), temp * 2.0) : mix(float3(1.0), float3(1.0, 0.78, 0.55), (temp - 0.5) * 2.0);
        col += tint * mag * twinkle * exp(-d * d / (sigma * sigma)) * 0.05;
    }
    // Milky Way: a soft, mottled band along a great circle
    float3 galN = normalize(float3(0.35, 0.3, -0.88));
    float b = dot(dir, galN);
    float band = exp(-b * b * 14.0);
    float mott = fbm3(dir * 6.0 + 3.0, 4);
    float dust = smoothstep(0.35, 0.65, fbm3(dir * 11.0 + 9.0, 3));
    col += float3(0.55, 0.6, 0.78) * band * (0.4 + 0.9 * mott) * (1.0 - 0.6 * dust * band) * 3.5e-4 * u.snow.z;
    // extinction / airmass near the horizon
    col *= smoothstep(-0.02, 0.22, dir.y);
    // the reflection pass renders at lower resolution (bigger star spots): keep reflected stars as bright as real ones
    if (u.flags.x > 0.5) col *= 0.3;
    return col * amt;
}

// The moon: lit sphere with a phase terminator, maria, limb darkening and faint earthshine.
inline float3 moonDisk(float3 dir, constant FrameUniforms& u) {
    float3 md = u.moon.xyz;
    if (md.y < -0.03) return float3(0.0);
    const float R = 0.0085;
    float cosA = dot(dir, md);
    if (cosA < cos(R * 1.4)) return float3(0.0);
    float3 right = normalize(cross(md, abs(md.y) > 0.99 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0)));
    float3 up = cross(right, md);
    float3 off = dir - md * cosA;
    float2 p = float2(dot(off, right), dot(off, up)) / R;
    float r = length(p);
    float pix = 2.0 / (u.proj[1][1] * u.screenSize.y) / R;
    float edge = 1.0 - smoothstep(1.0 - pix, 1.0 + pix, r);
    if (edge <= 0.0) return float3(0.0);
    float z = sqrt(max(0.0, 1.0 - r * r));
    float3 n = right * p.x + up * p.y - md * z;
    float e = u.moonParams.x;
    float3 Ls = cos(e) * md + sin(e) * right;
    float lit = smoothstep(-0.04, 0.1, dot(n, Ls));
    float2 q = p * 2.1;
    float mare = smoothstep(0.47, 0.62, fbm2(q + 3.7, 4));
    float crater = fbm2(q * 4.0 + 11.0, 3);
    float alb = (0.17 - 0.08 * mare) * (0.82 + 0.36 * crater);
    float limb = mix(0.72, 1.0, z);
    float3 col = float3(1.0, 0.97, 0.92) * alb * limb * (lit + 0.0012);   // + earthshine on the dark side
    float extinction = smoothstep(-0.03, 0.15, md.y);
    return col * edge * u.moonParams.y * mix(float3(1.0, 0.75, 0.5), float3(1.0), extinction);
}

inline float3 skyRadiance(float3 dir, constant FrameUniforms& u, texture2d<float> lut) {
    float3 sky = sampleSky(lut, dir);
    float cosA = dot(dir, u.sunDir.xyz);
    float sunR = 0.0095;
    float edge = smoothstep(cos(sunR * 1.25), cos(sunR), cosA);
    float r = saturate(acos(min(cosA, 1.0)) / sunR);
    float limb = 1.0 - 0.6 * (1.0 - sqrt(max(1.0 - r * r, 0.0)));
    float3 sun = u.moon.w > 0.5 ? float3(0.0) : u.sunColor.rgb * 40.0 * edge * limb;
    sun += moonDisk(dir, u);
    float4 cl = cloudLayer(dir, u, lut);
    float3 flash = u.weather.w * float3(0.35, 0.38, 0.5) * (0.4 + 0.6 * saturate(dir.y * 2.0));
    return (sky + sun + starField(dir, u)) * (1.0 - cl.a) + cl.rgb + flash;
}

// ================================================================ SHADOWS

inline float2 vogel(int i, int n, float phi) {
    float r = sqrt((float(i) + 0.5) / float(n));
    float th = float(i) * 2.39996323 + phi;
    return r * float2(cos(th), sin(th));
}

inline float shadowPCSS(float3 P, float3 N, constant FrameUniforms& u, depth2d_array<float> sm, float noise, float noise2,
                        texture2d<float> canopy, texture2d<float> heightTex) {
    int casc = -1;
    float3 sc = 0.0;
    for (int c = 0; c < 4; c++) {
        float3 Pn = P + N * u.cascadeInfo[c].x * 2.0;
        if (cascadeCoord(Pn, c, u, sc)) { casc = c; break; }
    }
    if (casc < 0) return shadowFallback(P, canopy, heightTex, u);
    if (casc == 3 && noise2 < cascadeEdgeFade(sc)) return shadowFallback(P, canopy, heightTex, u);
    float edge = max(abs(sc.x - 0.5), abs(sc.y - 0.5)) * 2.0;
    if (casc < 3 && edge > 0.8 && noise2 < (edge - 0.8) / 0.17) {
        float3 sc2;
        if (cascadeCoord(P + N * u.cascadeInfo[casc + 1].x * 2.0, casc + 1, u, sc2)) { casc += 1; sc = sc2; }
    }
    float texelW = u.cascadeInfo[casc].x;
    float range = u.cascadeInfo[casc].y;
    float frac = u.cascadeInfo[casc].w;
    float res = float(sm.get_width()) * frac;   // effective resolution of this cascade
    float bias = (texelW * 0.8 + 0.015) / range;
    float z = sc.z - bias;
    int samples = int(u.shadowParams.x);
    float phi = noise * 2.0 * PI;
    float lightSize = tan(u.sunDir.w) * u.shadowParams.z;   // penumbra width per meter of blocker distance (radius)
    float filterTexels;
    if (u.shadowParams.y > 0.5) {
        float searchTexels = clamp(lightSize * 2.0 * 35.0 / texelW, 2.0, 56.0);
        int bs = clamp(samples / 2, 8, 24);
        float sum = 0.0;
        float cnt = 0.0;
        for (int i = 0; i < bs; i++) {
            float2 off = vogel(i, bs, phi + 1.3) * searchTexels / res;
            float d = sm.sample(sPointClamp, cascadeTexCoord(sc.xy + off, casc, u), casc);
            if (d < z) { sum += d; cnt += 1.0; }
        }
        if (cnt < 0.5) return 1.0;
        float avgB = sum / cnt;
        float distWorld = (z - avgB) * range;
        float penumbraWorld = distWorld * lightSize * 2.0;
        filterTexels = clamp(penumbraWorld / texelW, 1.0, 72.0);
    } else {
        filterTexels = 1.8;
    }
    float lit = 0.0;
    for (int i = 0; i < samples; i++) {
        float2 off = vogel(i, samples, phi) * filterTexels / res;
        lit += sm.sample_compare(sShadowCmp, cascadeTexCoord(sc.xy + off, casc, u), casc, z);
    }
    return lit / float(samples);
}

inline float contactShadow(float3 P, float3 L, constant FrameUniforms& u, depth2d<float> depthTex, float noise, int steps, float len) {
    float near = u.jitter.w;
    for (int i = 0; i < steps; i++) {
        float t = (float(i) + noise) / float(steps);
        float3 Q = P + L * (len * t * t + 0.015);
        float4 c = u.viewProj * float4(Q, 1.0);
        if (c.w <= 0.0) break;
        float2 suv = c.xy / c.w * float2(0.5, -0.5) + 0.5;
        if (any(suv < 0.0) || any(suv > 1.0)) break;
        float d = depthTex.sample(sPointClamp, suv);
        float sceneLin = near / max(d, 1e-7);
        float diff = c.w - sceneLin;
        if (diff > 0.02 && diff < 0.35) return smoothstep(1.0, 0.6, t);
    }
    return 1.0;
}

// ================================================================ GTAO

fragment float gtaoFS(FSQOut in [[stage_in]],
                      constant FrameUniforms& u [[buffer(2)]],
                      constant float4& p [[buffer(3)]],
                      depth2d<float> depthTex [[texture(0)]],
                      texture2d<float> gNrm [[texture(1)]]) {
    float2 uv = in.uv;
    float d = depthTex.sample(sPointClamp, uv);
    if (d <= 0.0) return 1.0;
    float3 vpos = reconstructView(uv, d, u.invProj);
    float3 nW = gNrm.sample(sPointClamp, uv).xyz;
    float3 vN = normalize((u.view * float4(nW, 0.0)).xyz);
    float3 viewV = normalize(-vpos);
    float radius = p.x;
    int slices = int(p.y), steps = int(p.z);
    float zv = max(-vpos.z, 0.05);
    float2 rUV = float2(u.proj[0][0], u.proj[1][1]) * 0.5 * radius / zv;
    float minUV = 1.5 * u.screenSize.w;
    if (rUV.y < minUV) return 1.0;
    rUV = min(rUV, float2(0.25));
    float noiseA = ign(in.position.xy, u.jitter.z);
    float noiseB = ign(in.position.xy + float2(17.0, 59.0), u.jitter.z * 1.7 + 3.0);
    float falloffRange = 0.615 * radius;
    float falloffMul = -1.0 / falloffRange;
    float falloffAdd = (radius - falloffRange) / falloffRange + 1.0;
    float vis = 0.0;
    for (int s = 0; s < slices; s++) {
        float phi = (float(s) + noiseA) / float(slices) * PI;
        float2 omega = float2(cos(phi), -sin(phi));
        float3 dirV = float3(cos(phi), sin(phi), 0.0);
        float3 orthoDir = dirV - dot(dirV, viewV) * viewV;
        float3 axis = normalize(cross(orthoDir, viewV));
        float3 projN = vN - axis * dot(vN, axis);
        float projLen = length(projN);
        float sgnN = sign(dot(orthoDir, projN));
        float cosN = saturate(dot(projN, viewV) / max(projLen, 1e-4));
        float n = sgnN * acos(cosN);
        float low0 = cos(n + PI * 0.5), low1 = cos(n - PI * 0.5);
        float hc0 = low0, hc1 = low1;
        for (int k = 0; k < steps; k++) {
            float st = (float(k) + noiseB) / float(steps);
            st = st * st + 0.02;
            float2 off = omega * st * rUV;
            float2 uv0 = uv + off, uv1 = uv - off;
            float d0 = depthTex.sample(sPointClamp, uv0);
            float d1 = depthTex.sample(sPointClamp, uv1);
            float3 s0 = reconstructView(uv0, max(d0, 1e-7), u.invProj) - vpos;
            float3 s1 = reconstructView(uv1, max(d1, 1e-7), u.invProj) - vpos;
            float l0 = length(s0), l1 = length(s1);
            float w0 = saturate(l0 * falloffMul + falloffAdd);
            float w1 = saturate(l1 * falloffMul + falloffAdd);
            float shc0 = mix(low0, dot(s0 / max(l0, 1e-4), viewV), w0);
            float shc1 = mix(low1, dot(s1 / max(l1, 1e-4), viewV), w1);
            hc0 = max(hc0, shc0);
            hc1 = max(hc1, shc1);
        }
        float h0 = -acos(clamp(hc1, -1.0, 1.0));
        float h1 = acos(clamp(hc0, -1.0, 1.0));
        h0 = n + clamp(h0 - n, -PI * 0.5, PI * 0.5);
        h1 = n + clamp(h1 - n, -PI * 0.5, PI * 0.5);
        float iarc0 = (cosN + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) * 0.25;
        float iarc1 = (cosN + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) * 0.25;
        vis += projLen * (iarc0 + iarc1);
    }
    vis /= float(slices);
    return pow(saturate(vis), p.w);
}

fragment float aoBlurFS(FSQOut in [[stage_in]],
                        constant FrameUniforms& u [[buffer(2)]],
                        texture2d<float> ao [[texture(0)]],
                        depth2d<float> depthTex [[texture(1)]]) {
    float2 ts = 1.0 / float2(ao.get_width(), ao.get_height());
    float near = u.jitter.w;
    float d0 = near / max(depthTex.sample(sPointClamp, in.uv), 1e-7);
    float sum = 0.0, wsum = 0.0;
    for (int y = -2; y <= 1; y++) {
        for (int x = -2; x <= 1; x++) {
            float2 suv = in.uv + (float2(x, y) + 0.5) * ts;
            float d = near / max(depthTex.sample(sPointClamp, suv), 1e-7);
            float w = exp(-abs(d - d0) / (d0 * 0.04 + 0.02));
            sum += ao.sample(sLinearClamp, suv).r * w;
            wsum += w;
        }
    }
    return sum / max(wsum, 1e-4);
}

// ================================================================ VOLUMETRIC LIGHT (shadowed in-scattering → god rays)

inline float fogDensityAt(float3 p, constant FrameUniforms& u) {
    float h = p.y - u.fog.z;
    float d = u.fog.x * exp(-max(h, 0.0) * u.fog.y);
    if (u.colorGrass.w > 0.5) {
        float3 q = p * 0.03 + float3(u.wind.x, 0.05, u.wind.y) * u.wind.w * 0.06;
        float n = vnoise3(q) * 0.65 + vnoise3(q * 2.9 + 4.1) * 0.35;
        d *= mix(0.25, 1.75, n);
    }
    return d;
}

struct VolOut {
    float4 scatter [[color(0)]];
    float depth [[color(1)]];
};

fragment VolOut volumetricFS(FSQOut in [[stage_in]],
                             constant FrameUniforms& u [[buffer(2)]],
                             constant float4* sh [[buffer(1)]],
                             depth2d<float> depthTex [[texture(0)]],
                             depth2d_array<float> sm [[texture(1)]],
                             texture2d<float> canopyTex [[texture(2)]],
                             texture2d<float> heightTex [[texture(3)]],
                             texture2d<float> lakeTex [[texture(4)]]) {
    VolOut o;
    float2 uv = in.uv;
    uint2 dpx = min(uint2(uv * u.screenSize.xy), uint2(u.screenSize.xy) - 1u);
    float d = depthTex.read(dpx);
    float3 cam = u.cameraPos.xyz;
    float3 farP = reconstructWorld(uv, max(d, 1e-6), u.invViewProj);
    float3 rd = normalize(farP - cam);
    float sceneDist = d > 0.0 ? length(farP - cam) : 1e5;
    float3 wHit;
    float tW = waterHit(cam, rd, u, lakeTex, wHit);
    if (tW > 0.0) sceneDist = min(sceneDist, tW);
    float march = min(sceneDist, u.qualityParams.z);
    int N = int(u.qualityParams.y);
    float jit = ign(in.position.xy, u.jitter.z);
    float3 L = u.sunDir.xyz;
    float cosT = dot(rd, L);
    float phase = mix(phaseHG(cosT, u.fog.w), phaseHG(cosT, -0.2), 0.18);
    float3 avgSky = sh[4].rgb + u.weather.w * float3(0.3, 0.33, 0.42);
    float3 albedo = u.fogColor.rgb;
    float ambK = u.fogColor.w;
    float3 inscat = 0.0;
    float T = 1.0;
    for (int i = 0; i < N; i++) {
        float f0 = float(i) / float(N), f1 = float(i + 1) / float(N);
        float t0 = march * f0 * f0, t1 = march * f1 * f1;
        float t = mix(t0, t1, jit);
        float dt = t1 - t0;
        float3 p = cam + rd * t;
        float dens = fogDensityAt(p, u);
        if (dens < 1e-7) continue;
        float shadow = shadowSimple(p, u, sm, canopyTex, heightTex);
        float occ = canopyOcclusion(p, canopyTex, heightTex, u);
        float3 Ls = u.sunColor.rgb * shadow * phase + avgSky * ambK * occ;
        float Tr = exp(-dens * dt);
        inscat += T * Ls * albedo * (1.0 - Tr);
        T *= Tr;
        if (T < 0.002) break;
    }
    if (sceneDist > march && T > 0.002) {
        float3 p0 = cam + rd * march;
        float k = u.fog.y;
        float d0 = u.fog.x * exp(-max(p0.y - u.fog.z, 0.0) * k);
        float rem = min(sceneDist, 40000.0) - march;
        float ky = k * rd.y;
        float od = abs(ky) > 1e-5 ? d0 * (1.0 - exp(-ky * rem)) / ky : d0 * rem;
        od = clamp(od, 0.0, 50.0);
        float Tr = exp(-od);
        float3 Ls = u.sunColor.rgb * phase * 0.7 + avgSky * ambK;
        inscat += T * Ls * albedo * (1.0 - Tr);
        T *= Tr;
    }
    o.scatter = float4(inscat, T);
    o.depth = sceneDist;
    return o;
}

inline float4 upsampleVolumetric(texture2d<float> vol, texture2d<float> vdep, float2 uv, float dist) {
    float2 sz = float2(vol.get_width(), vol.get_height());
    float2 p = uv * sz - 0.5;
    float2 f = fract(p);
    int2 i0 = int2(floor(p));
    float4 sum = 0.0;
    float wsum = 0.0;
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            int2 c = clamp(i0 + int2(i, j), int2(0), int2(sz) - 1);
            float bw = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
            float dd = vdep.read(uint2(c)).r;
            float rel = abs(dd - dist) / max(min(dd, dist), 0.5);
            float w = bw * (exp(-rel * 6.0) + 1e-4);
            sum += vol.read(uint2(c)) * w;
            wsum += w;
        }
    }
    return sum / max(wsum, 1e-6);
}

// ================================================================ DEFERRED LIGHTING

inline float3 multiBounceAO(float ao, float3 albedo) {
    float3 a = 2.0404 * albedo - 0.3324;
    float3 b = -4.7951 * albedo + 0.6417;
    float3 c = 2.7552 * albedo + 0.6903;
    return max(float3(ao), ((ao * a + b) * ao + c) * ao);
}

// Animated underwater light caustics (sharp bright web).
inline float causticPattern(float2 p, float t) {
    float a = vnoise2(p * 1.7 + float2(t * 0.35, t * 0.21));
    float b = vnoise2(p * 2.3 - float2(t * 0.27, -t * 0.31) + 7.3);
    float c = 1.0 - abs(a - b);
    return pow(c, 8.0) * 2.4;
}

inline float3 lightningIrradiance(constant FrameUniforms& u) { return u.weather.w * float3(0.9, 1.0, 1.3); }

struct ShadeResult {
    float3 color;
    float dist;
    float shadow;
    float ao;
    float3 albedo;
    float3 normal;
    float3 P;
};

#define LIGHT_ARGS constant FrameUniforms& u, constant float4* sh, texture2d<float> gAlb, texture2d<float> gNrm, \
    texture2d<float> gMat, depth2d<float> depthTex, depth2d_array<float> sm, texture2d<float> aoTex, \
    texture2d<float> skyLUT, texture2d<float> canopyTex, texture2d<float> heightTex, texture2d<float> lakeTex
#define LIGHT_PASS u, sh, gAlb, gNrm, gMat, depthTex, sm, aoTex, skyLUT, canopyTex, heightTex, lakeTex

inline ShadeResult shadeOpaque(uint2 px, float d, float noise, float noise2, LIGHT_ARGS) {
    ShadeResult r;
    float2 uv = (float2(px) + 0.5) * u.screenSize.zw;
    float3 cam = u.cameraPos.xyz;
    float3 L = u.sunDir.xyz;
    float4 alb = gAlb.read(px);
    float4 nr = gNrm.read(px);
    float4 mt = gMat.read(px);
    float3 albedo = alb.rgb;
    float matAO = alb.a;
    float3 N = normalize(nr.xyz);
    float rough = clamp(nr.w, 0.03, 1.0);
    float trans = mt.x;
    int matId = int(mt.y * 3.0 + 0.5);
    float specL = mt.z;
    bool foliage = matId >= 2;
    float3 P = reconstructWorld(uv, d, u.invViewProj);
    float3 V = normalize(cam - P);
    float dist = length(P - cam);
    float NdotV = max(dot(N, V), 1e-4);

    float shadow = 1.0;
    if (L.y > -0.05) {
        shadow = shadowPCSS(P, N, u, sm, noise, noise2, canopyTex, heightTex);
        if (u.qualityParams.x > 0.5 && shadow > 0.01 && dist < 120.0) {
            shadow *= contactShadow(P, L, u, depthTex, noise2, int(u.qualityParams.x), foliage ? 0.3 : 0.7);
        }
        shadow *= cloudShadow(P, u);
    } else {
        shadow = 0.0;
    }
    float3 sunE = u.sunColor.rgb * shadow;
    float3 ambScale = float3(1.0);
    // underwater: light is absorbed on its way down and focused into caustics
    if (u.water.y > 0.5 && P.y < u.water.x && lakeMaskAt(P.xz, lakeTex, u) > 0.5) {
        float dw = u.water.x - P.y;
        float sl = max(L.y, 0.25);
        sunE *= exp(-u.waterColor.rgb * dw / sl) * mix(1.0, causticPattern(P.xz + L.xz * dw / sl, u.wind.w), saturate(dw * 1.5));
        ambScale = exp(-u.waterColor.rgb * dw * 0.8);
    }
    float NdotL = dot(N, L);
    float diffTerm = foliage ? saturate((NdotL + 0.4) / 1.4) : saturate(NdotL);
    float3 H = normalize(L + V);
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));
    float a = rough * rough;
    float a2 = a * a;
    float dd = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * dd * dd);
    float nl = saturate(NdotL);
    float vis = 0.5 / max(nl * sqrt(NdotV * NdotV * (1.0 - a2) + a2) + NdotV * sqrt(nl * nl * (1.0 - a2) + a2), 1e-5);
    float3 F0 = float3(0.04 * specL * 2.0);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
    float3 spec = min(D * vis * F * nl, float3(40.0));
    float3 diffuse = albedo / PI * diffTerm * (1.0 - F);
    float3 direct = sunE * (diffuse + spec);
    if (trans > 0.0) {
        float back = saturate(-NdotL);
        float fwd = pow(saturate(dot(-V, L)), 8.0);
        float3 tcol = saturate(albedo * albedo * float3(9.0, 8.5, 4.0) + albedo * float3(0.35, 0.5, 0.1));
        direct += sunE * tcol / PI * trans * (back * 0.8 + fwd * 0.9 + 0.1);
    }

    float ao = 1.0;
    if (u.qualityParams.w > 0.5) ao = aoTex.sample(sLinearClamp, uv).r;
    if (matId == 3) ao = mix(1.0, ao, 0.55);
    float totalAO = ao * matAO;
    float3 aoMB = multiBounceAO(totalAO, albedo);
    float occ = canopyOcclusion(P, canopyTex, heightTex, u);
    float3 irr = shIrradiance(N, sh) + lightningIrradiance(u) * saturate(N.y * 0.5 + 0.6);
    if (foliage) irr = mix(irr, shIrradiance(-N, sh), trans * 0.45);
    float3 ambient = albedo / PI * irr * aoMB * occ;
    float3 R = reflect(-V, N);
    float3 skyR = sampleSky(skyLUT, normalize(float3(R.x, max(R.y, 0.03), R.z)));
    skyR = mix(skyR, irr / PI, rough);
    float3 Fa = F0 + (max(float3(1.0 - rough), F0) - F0) * pow(1.0 - NdotV, 5.0);
    float specOcc = saturate(pow(NdotV + totalAO, exp2(-16.0 * rough - 1.0)) - 1.0 + totalAO);
    float3 specAmb = skyR * Fa * specOcc * occ * (matId == 3 ? 0.2 : (matId == 2 ? 0.5 : 1.0));
    r.color = direct + (ambient + specAmb) * ambScale;
    r.dist = dist;
    r.shadow = shadow;
    r.ao = totalAO;
    r.albedo = albedo;
    r.normal = N;
    r.P = P;
    return r;
}

// ---------------------------------------------------------------- water surface

inline float waterH(float2 p, float t, float2 w) {
    float h = vnoise2(p * 0.33 + w * t * 0.22) * 0.5;
    h += vnoise2(p * 0.85 - w.yx * t * 0.33 + 3.1) * 0.25;
    h += vnoise2(p * 2.1 + w * t * 0.6 + 7.7) * 0.12;
    h += vnoise2(p * 4.9 - w * t * 0.95 + 1.3) * 0.06;
    return h;
}

inline float3 waterNormal(float3 wp, float dist, constant FrameUniforms& u) {
    float t = u.wind.w;
    float2 w = u.wind.xy;
    float e = 0.04 + dist * 0.002;
    float h0 = waterH(wp.xz, t, w);
    float hx = waterH(wp.xz + float2(e, 0.0), t, w);
    float hz = waterH(wp.xz + float2(0.0, e), t, w);
    float amp = (0.05 + 0.5 * u.wind.z + 0.15 * u.weather.x) / (1.0 + dist * 0.008);
    float3 n = normalize(float3(-(hx - h0) / e * amp, 1.0, -(hz - h0) / e * amp));
    if (u.weather.x > 0.001) {
        float2 rip = rainRipples(wp.xz, t, u.weather.x) * (1.0 - smoothstep(25.0, 60.0, dist));
        n = normalize(n + float3(rip.x, 0.0, rip.y));
    }
    return n;
}

inline float3 shadeWater(float3 wp, float3 V, float tW, float3 bottomColor, float3 bottomP, bool bottomValid, float3 Nw,
                         float2 uv, texture2d<float> reflTex, LIGHT_ARGS) {
    float3 L = u.sunDir.xyz;
    float NdotV = saturate(dot(Nw, V));
    float F = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);
    // refraction with wavelength-dependent absorption + in-water scattering
    float path = bottomValid ? clamp(length(bottomP - wp), 0.0, 60.0) : 60.0;
    float3 absorb = exp(-u.waterColor.rgb * path);
    float sunVis = shadowSimple(wp, u, sm, canopyTex, heightTex) * cloudShadow(wp, u);
    float3 irr = shIrradiance(float3(0.0, 1.0, 0.0), sh) + lightningIrradiance(u);
    float3 inScatter = float3(0.007, 0.013, 0.011) * (irr / PI + u.sunColor.rgb * sunVis * 0.3);
    float3 refr = bottomColor * absorb + inScatter * (1.0 - dot(absorb, float3(1.0 / 3.0))) * 4.0;
    // reflection: planar reflection pass (with ripple distortion) or sky
    float3 refl;
    if (u.water.z > 0.5) {
        float2 ruv = clamp(uv + Nw.xz * 0.05, float2(0.001), float2(0.999));
        refl = reflTex.sample(sLinearClamp, ruv, level(0)).rgb;
    } else {
        float3 R = reflect(-V, Nw);
        refl = skyRadiance(normalize(float3(R.x, max(R.y, 0.01), R.z)), u, skyLUT);
    }
    // sun glint
    float3 H = normalize(L + V);
    float rough = 0.07 + u.weather.x * 0.06;
    float a2 = rough * rough * rough * rough;
    float NdotH = saturate(dot(Nw, H));
    float dd = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * dd * dd);
    float nl = saturate(dot(Nw, L));
    float vis = 0.5 / max(nl * sqrt(NdotV * NdotV * (1.0 - a2) + a2) + NdotV * sqrt(nl * nl * (1.0 - a2) + a2), 1e-5);
    float Fs = 0.02 + 0.98 * pow(1.0 - saturate(dot(V, H)), 5.0);
    float3 glint = min(u.sunColor.rgb * sunVis * D * vis * Fs * nl, float3(80.0));
    return mix(refr, refl, F) + glint;
}

inline float4 analyticFog(float3 cam, float3 rd, float dist, constant FrameUniforms& u, constant float4* sh) {
    float k = u.fog.y;
    float d0 = u.fog.x * exp(-max(cam.y - u.fog.z, 0.0) * k);
    float ky = k * rd.y;
    float Ld = min(dist, 20000.0);
    float od = abs(ky) > 1e-5 ? d0 * (1.0 - exp(-ky * Ld)) / ky : d0 * Ld;
    float T = exp(-clamp(od, 0.0, 50.0));
    float phase = mix(phaseHG(dot(rd, u.sunDir.xyz), u.fog.w), phaseHG(dot(rd, u.sunDir.xyz), -0.2), 0.18);
    float3 Ls = u.sunColor.rgb * phase * 0.6 + sh[4].rgb * u.fogColor.w;
    return float4(Ls * u.fogColor.rgb * (1.0 - T), T);
}

fragment float4 lightingFS(FSQOut in [[stage_in]],
                           constant FrameUniforms& u [[buffer(2)]],
                           constant float4* sh [[buffer(1)]],
                           texture2d<float> gAlb [[texture(0)]],
                           texture2d<float> gNrm [[texture(1)]],
                           texture2d<float> gMat [[texture(2)]],
                           depth2d<float> depthTex [[texture(3)]],
                           depth2d_array<float> sm [[texture(4)]],
                           texture2d<float> aoTex [[texture(5)]],
                           texture2d<float> skyLUT [[texture(6)]],
                           texture2d<float> volTex [[texture(7)]],
                           texture2d<float> volDepth [[texture(8)]],
                           texture2d<float> canopyTex [[texture(9)]],
                           texture2d<float> heightTex [[texture(10)]],
                           texture2d<float> lakeTex [[texture(11)]],
                           texture2d<float> reflTex [[texture(12)]]) {
    uint2 px = uint2(in.position.xy);
    float2 uv = in.uv;
    float3 cam = u.cameraPos.xyz;
    float noise = ign(in.position.xy, u.jitter.z);
    float noise2 = ign(in.position.xy + float2(41.0, 13.0), u.jitter.z + 19.0);
    int debug = int(u.misc.z + 0.5);
    float3 rd = normalize(reconstructWorld(uv, 1e-6, u.invViewProj) - cam);
    float d = depthTex.read(px);
    float opaqueDist = d > 0.0 ? length(reconstructWorld(uv, d, u.invViewProj) - cam) : 1e9;

    // lake surface in front of the opaque scene?
    float3 wp = 0.0;
    float tW = waterHit(cam, rd, u, lakeTex, wp);
    bool water = tW > 0.0 && tW < opaqueDist && u.flags.x < 0.5;
    uint2 spx = px;
    float3 Nw = float3(0.0, 1.0, 0.0);
    if (water) {
        Nw = waterNormal(wp, tW, u);
        // refraction: shade a displaced underwater pixel instead of the one directly behind the surface
        float2 offPx = Nw.xz * clamp(opaqueDist - tW, 0.0, 4.0) * 16.0 * (u.screenSize.y / 1080.0) / (1.0 + tW * 0.04);
        int2 cand = clamp(int2(px) + int2(offPx), int2(0), int2(u.screenSize.xy) - 1);
        float dc = depthTex.read(uint2(cand));
        if (dc > 0.0) {
            float3 Pc = reconstructWorld((float2(cand) + 0.5) * u.screenSize.zw, dc, u.invViewProj);
            if (Pc.y < u.water.x - 0.02) spx = uint2(cand);
        }
    }
    float ds = depthTex.read(spx);
    float3 color;
    float dist;
    ShadeResult sr;
    sr.shadow = 1.0; sr.ao = 1.0; sr.albedo = 0.0; sr.normal = 0.0; sr.P = 0.0;
    if (ds <= 0.0) {
        dist = 1e5;
        color = skyRadiance(rd, u, skyLUT);
    } else {
        sr = shadeOpaque(spx, ds, noise, noise2, LIGHT_PASS);
        color = sr.color;
        dist = sr.dist;
    }
    if (water) {
        color = shadeWater(wp, -rd, tW, color, sr.P, ds > 0.0, Nw, uv, reflTex, LIGHT_PASS);
        dist = tW;
    }
    float4 fog = u.flags.x > 0.5 ? analyticFog(cam, rd, dist, u, sh) : upsampleVolumetric(volTex, volDepth, uv, dist);
    color = color * fog.a + fog.rgb;
    if (debug == 1) return float4(sr.albedo, 1.0);
    if (debug == 2) return float4(sr.normal * 0.5 + 0.5, 1.0);
    if (debug == 3) return float4(float3(sr.ao), 1.0);
    if (debug == 4) return float4(float3(sr.shadow), 1.0);
    if (any(isnan(color)) || any(isinf(color))) color = 0.0;
    return float4(color, 1.0);
}
