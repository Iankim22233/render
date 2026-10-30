import Foundation
import simd

typealias float2 = SIMD2<Float>
typealias float3 = SIMD3<Float>
typealias float4 = SIMD4<Float>
typealias float4x4 = simd_float4x4

extension float4x4 {
    static var identity: float4x4 { matrix_identity_float4x4 }

    init(translation t: float3) {
        self = matrix_identity_float4x4
        columns.3 = float4(t.x, t.y, t.z, 1)
    }

    init(rotationY a: Float) {
        let c = cos(a), s = sin(a)
        self = float4x4(columns: (float4(c, 0, -s, 0), float4(0, 1, 0, 0), float4(s, 0, c, 0), float4(0, 0, 0, 1)))
    }

    /// Right-handed view space (camera looks down -Z), reversed-Z, infinite far plane. Depth = near / viewDistance.
    static func perspectiveReversedZInfinite(fovY: Float, aspect: Float, near: Float) -> float4x4 {
        let f = 1 / tan(fovY * 0.5)
        return float4x4(columns: (float4(f / aspect, 0, 0, 0),
                                  float4(0, f, 0, 0),
                                  float4(0, 0, 0, -1),
                                  float4(0, 0, near, 0)))
    }

    /// Right-handed orthographic projection with z mapped to [0, 1] (near -> 0).
    static func orthographic(left l: Float, right r: Float, bottom b: Float, top t: Float, near n: Float, far f: Float) -> float4x4 {
        return float4x4(columns: (float4(2 / (r - l), 0, 0, 0),
                                  float4(0, 2 / (t - b), 0, 0),
                                  float4(0, 0, -1 / (f - n), 0),
                                  float4(-(r + l) / (r - l), -(t + b) / (t - b), -n / (f - n), 1)))
    }

    static func lookAt(eye: float3, center: float3, up: float3) -> float4x4 {
        let f = simd_normalize(center - eye)
        let s = simd_normalize(simd_cross(f, up))
        let u = simd_cross(s, f)
        return float4x4(columns: (float4(s.x, u.x, -f.x, 0),
                                  float4(s.y, u.y, -f.y, 0),
                                  float4(s.z, u.z, -f.z, 0),
                                  float4(-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1)))
    }
}

@inline(__always) func mix(_ a: Float, _ b: Float, _ t: Float) -> Float { a + (b - a) * t }
@inline(__always) func mix(_ a: float3, _ b: float3, _ t: Float) -> float3 { a + (b - a) * t }
@inline(__always) func saturate(_ x: Float) -> Float { min(max(x, 0), 1) }
@inline(__always) func smoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
    let t = saturate((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)
}
@inline(__always) func clampf(_ x: Float, _ a: Float, _ b: Float) -> Float { min(max(x, a), b) }

/// Extracts the 6 frustum planes (xyz normal pointing inward, w distance) from a Metal-style clip matrix (z in [0,1]).
func frustumPlanes(_ m: float4x4) -> [float4] {
    let r0 = float4(m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x)
    let r1 = float4(m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y)
    let r2 = float4(m.columns.0.z, m.columns.1.z, m.columns.2.z, m.columns.3.z)
    let r3 = float4(m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w)
    var planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2]
    for i in 0..<planes.count {
        let len = simd_length(float3(planes[i].x, planes[i].y, planes[i].z))
        if len > 1e-6 { planes[i] /= len } else { planes[i] = float4(0, 0, 0, 1) }
    }
    return planes
}

@inline(__always) func sphereVisible(_ planes: UnsafeBufferPointer<float4>, _ c: float3, _ r: Float) -> Bool {
    for p in planes {
        if p.x * c.x + p.y * c.y + p.z * c.z + p.w < -r { return false }
    }
    return true
}

@inline(__always) func aabbVisible(_ planes: UnsafeBufferPointer<float4>, _ mn: float3, _ mx: float3) -> Bool {
    for p in planes {
        let v = float3(p.x >= 0 ? mx.x : mn.x, p.y >= 0 ? mx.y : mn.y, p.z >= 0 ? mx.z : mn.z)
        if p.x * v.x + p.y * v.y + p.z * v.z + p.w < 0 { return false }
    }
    return true
}

func halton(_ index: Int, _ base: Int) -> Float {
    var f: Float = 1, r: Float = 0, i = index
    while i > 0 {
        f /= Float(base)
        r += f * Float(i % base)
        i /= base
    }
    return r
}

/// Arbitrary unit vector perpendicular to v.
func anyPerpendicular(_ v: float3) -> float3 {
    let a: float3 = abs(v.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0)
    return simd_normalize(simd_cross(v, a))
}

func rotate(_ v: float3, axis: float3, angle: Float) -> float3 {
    let q = simd_quatf(angle: angle, axis: simd_normalize(axis))
    return q.act(v)
}
