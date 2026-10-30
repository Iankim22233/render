// ================================================================ RAIN
// Camera-centred, world-stable box of falling drops rendered as motion-blurred streaks.

struct RainVOut {
    float4 position [[position]];
    float alpha;
    float2 q;
};

vertex RainVOut rainVS(uint vid [[vertex_id]], uint iid [[instance_id]],
                       constant FrameUniforms& u [[buffer(2)]],
                       constant float4x4& vp [[buffer(3)]],
                       texture2d<float> canopyTex [[texture(0)]],
                       texture2d<float> heightTex [[texture(1)]]) {
    RainVOut o;
    o.position = float4(2.0, 2.0, 2.0, 1.0);
    o.alpha = 0.0;
    o.q = 0.0;
    const float3 B = float3(34.0, 24.0, 34.0);
    float3 cam = u.cameraPos.xyz;
    uint h = pcgHash(iid * 3u + 17u);
    float3 r = float3(hashU(h), hashU(h ^ 0x9e3779b9u), hashU(h * 747796405u + 7u));
    float speed = mix(7.0, 10.0, hashU(h + 0x68bc21ebu));
    float t = u.wind.w;
    float3 vel = float3(u.wind.x * u.wind.z * 4.0, -speed, u.wind.y * u.wind.z * 4.0);
    float3 origin = cam - float3(B.x * 0.5, B.y * 0.4, B.z * 0.5);
    float3 p = r * B + vel * t - origin;
    p = origin + (p - B * floor(p / B));
    float3 toCam = cam - p;
    float dist = length(toCam);
    float ground = heightTex.sample(sLinearClamp, heightUV(p.xz, u), level(0)).r;
    if (u.water.y > 0.5) ground = max(ground, u.water.x);
    if (p.y < ground) return o;
    float4 can = canopyTex.sample(sLinearClamp, p.xz * u.terrain.y + 0.5, level(0));
    float topH = can.g * u.terrain.w;
    float underCanopy = 1.0 - smoothstep(topH * 0.6, topH + 0.01, p.y - ground);
    float a = (1.0 - can.r * underCanopy * 0.7);
    a *= smoothstep(0.3, 1.2, dist) * (1.0 - smoothstep(B.x * 0.3, B.x * 0.5, dist));
    if (a <= 0.001) return o;
    float3 dir = normalize(vel);
    float len = speed * 0.04;
    float3 side = normalize(cross(dir, toCam / max(dist, 1e-3)));
    float w = 0.001 + dist * 0.0007;
    float sx = (vid & 1u) ? 1.0 : -1.0;
    float sy = (vid & 2u) ? 1.0 : 0.0;
    float3 wp = p - dir * len * sy + side * w * sx;
    o.position = vp * float4(wp, 1.0);
    o.alpha = a;
    o.q = float2(sx, sy);
    return o;
}

fragment float4 rainFS(RainVOut in [[stage_in]],
                       constant FrameUniforms& u [[buffer(2)]],
                       constant float4* sh [[buffer(1)]]) {
    float edge = 1.0 - abs(in.q.x);
    float along = sin(saturate(in.q.y) * PI);
    float a = in.alpha * edge * along * 0.3;
    // drops refract/reflect the bright surrounding sky; add a little sun and lightning
    float3 env = sh[4].rgb * 1.9 + u.sunColor.rgb * 0.04 + u.weather.w * float3(0.6, 0.65, 0.8);
    return float4(env * a, a * 0.2);
}

// ================================================================ SNOW
// Slowly tumbling flakes with per-flake flutter, drifting with the wind.

vertex RainVOut snowVS(uint vid [[vertex_id]], uint iid [[instance_id]],
                       constant FrameUniforms& u [[buffer(2)]],
                       constant float4x4& vp [[buffer(3)]],
                       texture2d<float> canopyTex [[texture(0)]],
                       texture2d<float> heightTex [[texture(1)]]) {
    RainVOut o;
    o.position = float4(2.0, 2.0, 2.0, 1.0);
    o.alpha = 0.0;
    o.q = 0.0;
    const float3 B = float3(30.0, 22.0, 30.0);
    float3 cam = u.cameraPos.xyz;
    uint h = pcgHash(iid * 7u + 101u);
    float3 r = float3(hashU(h), hashU(h ^ 0x9e3779b9u), hashU(h * 747796405u + 7u));
    float speed = mix(0.7, 1.5, hashU(h + 0x68bc21ebu));
    float t = u.wind.w;
    float3 vel = float3(u.wind.x * u.wind.z * 2.2, -speed, u.wind.y * u.wind.z * 2.2);
    float3 origin = cam - float3(B.x * 0.5, B.y * 0.4, B.z * 0.5);
    float3 p = r * B + vel * t - origin;
    p = origin + (p - B * floor(p / B));
    float ph = hashU(h + 11u) * 6.283, fr = mix(0.8, 1.9, hashU(h + 13u));
    p.x += sin(t * fr + ph) * 0.35;
    p.z += cos(t * fr * 0.83 + ph * 1.7) * 0.35;
    float3 toCam = cam - p;
    float dist = length(toCam);
    float ground = heightTex.sample(sLinearClamp, heightUV(p.xz, u), level(0)).r;
    if (u.water.y > 0.5) ground = max(ground, u.water.x);
    if (p.y < ground) return o;
    float4 can = canopyTex.sample(sLinearClamp, p.xz * u.terrain.y + 0.5, level(0));
    float topH = can.g * u.terrain.w;
    float underCanopy = 1.0 - smoothstep(topH * 0.6, topH + 0.01, p.y - ground);
    float a = (1.0 - can.r * underCanopy * 0.6);
    a *= smoothstep(0.7, 2.0, dist) * (1.0 - smoothstep(B.x * 0.3, B.x * 0.5, dist));
    if (a <= 0.001) return o;
    float size = mix(0.005, 0.014, pow(hashU(h + 17u), 2.0));
    float w = max(size, dist * 0.0006);
    float3 vdir = toCam / max(dist, 1e-3);
    float3 right = normalize(cross(float3(0.0, 1.0, 0.0), vdir));
    float3 upv = cross(vdir, right);
    float sx = (vid & 1u) ? 1.0 : -1.0;
    float sy = (vid & 2u) ? 1.0 : -1.0;
    float3 wp = p + right * w * sx + upv * w * sy * 1.25;
    o.position = vp * float4(wp, 1.0);
    o.alpha = a * saturate(size / w + 0.25);
    o.q = float2(sx, sy);
    return o;
}

fragment float4 snowFS(RainVOut in [[stage_in]],
                       constant FrameUniforms& u [[buffer(2)]],
                       constant float4* sh [[buffer(1)]]) {
    float d = length(in.q);
    float a = in.alpha * smoothstep(1.0, 0.2, d) * 0.7;
    float3 env = sh[4].rgb * 1.1 + u.sunColor.rgb * 0.12 + u.weather.w * float3(0.5, 0.55, 0.7);
    return float4(env * a, a);
}
