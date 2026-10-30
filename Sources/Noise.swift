import Foundation
import simd

struct RNG: RandomNumberGenerator {
    var s: UInt64
    init(_ seed: UInt64) { s = seed &* 0x9E3779B97F4A7C15 &+ 0x632BE59BD9B4E019 }
    mutating func next() -> UInt64 {
        s &+= 0x9E3779B97F4A7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func float() -> Float { Float(next() >> 40) / Float(1 << 24) }
    mutating func range(_ a: Float, _ b: Float) -> Float { a + (b - a) * float() }
    mutating func int(_ a: Int, _ b: Int) -> Int { a + Int(next() % UInt64(b - a + 1)) }
    mutating func chance(_ p: Float) -> Bool { float() < p }
    mutating func gauss() -> Float {
        let u1 = max(float(), 1e-7), u2 = float()
        return sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
    }
    mutating func unitVector() -> float3 {
        let z = range(-1, 1), a = range(0, 2 * .pi)
        let r = sqrt(max(0, 1 - z * z))
        return float3(r * cos(a), z, r * sin(a))
    }
}

@inline(__always) func hashU32(_ x: UInt32) -> UInt32 {
    var x = x
    x ^= x >> 16; x &*= 0x7feb352d
    x ^= x >> 15; x &*= 0x846ca68b
    x ^= x >> 16
    return x
}

@inline(__always) func hash2i(_ x: Int32, _ y: Int32, _ seed: UInt32) -> UInt32 {
    hashU32((UInt32(bitPattern: x) &* 0x8da6b343) ^ (UInt32(bitPattern: y) &* 0xd8163841) ^ (seed &* 0xcb1ab31f) ^ 0x9e3779b9)
}

@inline(__always) func hashFloat(_ h: UInt32) -> Float { Float(h >> 8) * (1.0 / 16777216.0) }

private let gradTable: [float2] = (0..<16).map { i in
    let a = Float(i) / 16 * 2 * .pi + 0.2
    return float2(cos(a), sin(a))
}

@inline(__always) private func wrapI(_ v: Int32, _ p: Int32) -> Int32 {
    if p <= 0 { return v }
    let m = v % p
    return m < 0 ? m + p : m
}

/// Gradient noise in roughly [-1, 1]. Tileable when `period` > 0 (period in lattice cells).
func gradNoise(_ x: Float, _ y: Float, period: Int32 = 0, seed: UInt32 = 0) -> Float {
    gradNoise(x, y, px: period, py: period, seed: seed)
}

func gradNoise(_ x: Float, _ y: Float, px period: Int32, py periodY: Int32, seed: UInt32 = 0) -> Float {
    let fx0 = floorf(x), fy0 = floorf(y)
    let fx = x - fx0, fy = y - fy0
    let ix = Int32(fx0), iy = Int32(fy0)
    let x0 = wrapI(ix, period), x1 = wrapI(ix + 1, period)
    let y0 = wrapI(iy, periodY), y1 = wrapI(iy + 1, periodY)
    let g00 = gradTable[Int(hash2i(x0, y0, seed) & 15)]
    let g10 = gradTable[Int(hash2i(x1, y0, seed) & 15)]
    let g01 = gradTable[Int(hash2i(x0, y1, seed) & 15)]
    let g11 = gradTable[Int(hash2i(x1, y1, seed) & 15)]
    let n00 = g00.x * fx + g00.y * fy
    let n10 = g10.x * (fx - 1) + g10.y * fy
    let n01 = g01.x * fx + g01.y * (fy - 1)
    let n11 = g11.x * (fx - 1) + g11.y * (fy - 1)
    let ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    let uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    let a = n00 + (n10 - n00) * ux
    let b = n01 + (n11 - n01) * ux
    return (a + (b - a) * uy) * 1.414
}

func fbm(_ x: Float, _ y: Float, octaves: Int, period: Int32 = 0, seed: UInt32 = 0, gain: Float = 0.5) -> Float {
    fbm(x, y, octaves: octaves, px: period, py: period, seed: seed, gain: gain)
}

func fbm(_ x: Float, _ y: Float, octaves: Int, px: Int32, py: Int32, seed: UInt32 = 0, gain: Float = 0.5) -> Float {
    var sum: Float = 0, amp: Float = 0.5, fx = x, fy = y, ppx = px, ppy = py, norm: Float = 0
    for o in 0..<octaves {
        sum += amp * gradNoise(fx, fy, px: ppx, py: ppy, seed: seed &+ UInt32(o) &* 1013)
        norm += amp
        amp *= gain
        fx *= 2; fy *= 2
        ppx *= 2; ppy *= 2
    }
    return sum / norm
}

func ridged(_ x: Float, _ y: Float, octaves: Int, period: Int32 = 0, seed: UInt32 = 0) -> Float {
    ridged(x, y, octaves: octaves, px: period, py: period, seed: seed)
}

func ridged(_ x: Float, _ y: Float, octaves: Int, px: Int32, py: Int32, seed: UInt32 = 0) -> Float {
    var sum: Float = 0, amp: Float = 0.5, fx = x, fy = y, p = px, q = py, norm: Float = 0, prev: Float = 1
    for o in 0..<octaves {
        var n = 1 - abs(gradNoise(fx, fy, px: p, py: q, seed: seed &+ UInt32(o) &* 7919))
        n = n * n
        sum += n * amp * prev
        prev = n
        norm += amp
        amp *= 0.5
        fx *= 2; fy *= 2
        p *= 2; q *= 2
    }
    return sum / norm
}

/// Tileable Worley noise. Returns (F1, F2, cell id hash in [0,1)). Coordinates in cell units.
func voronoi(_ x: Float, _ y: Float, period: Int32, seed: UInt32, jitter: Float = 1) -> (Float, Float, Float) {
    voronoi(x, y, px: period, py: period, seed: seed, jitter: jitter)
}

func voronoi(_ x: Float, _ y: Float, px period: Int32, py periodY: Int32, seed: UInt32, jitter: Float = 1) -> (Float, Float, Float) {
    let ix = Int32(floorf(x)), iy = Int32(floorf(y))
    let fx = x - floorf(x), fy = y - floorf(y)
    var f1: Float = 9, f2: Float = 9, id: Float = 0
    for dy: Int32 in -1...1 {
        for dx: Int32 in -1...1 {
            let cx = wrapI(ix + dx, period), cy = wrapI(iy + dy, periodY)
            let h = hash2i(cx, cy, seed)
            let px = Float(dx) + 0.5 + (hashFloat(h) - 0.5) * jitter
            let py = Float(dy) + 0.5 + (hashFloat(hashU32(h)) - 0.5) * jitter
            let ddx = px - fx, ddy = py - fy
            let d = sqrt(ddx * ddx + ddy * ddy)
            if d < f1 { f2 = f1; f1 = d; id = hashFloat(hashU32(h ^ 0x5bd1e995)) } else if d < f2 { f2 = d }
        }
    }
    return (f1, f2, id)
}
