import Foundation
import Metal
import simd

// MARK: - Canvas: a tiny CPU rasterizer used to paint leaves, needles, bark and forest-floor textures.

final class Canvas {
    let w: Int, h: Int, wrap: Bool
    var col: [float3]
    var alpha: [Float]
    var height: [Float]
    var varn: [Float]
    var thick: [Float]

    init(w: Int, h: Int, wrap: Bool, base: float3, alpha a: Float = 1) {
        self.w = w; self.h = h; self.wrap = wrap
        col = Array(repeating: base, count: w * h)
        alpha = Array(repeating: a, count: w * h)
        height = Array(repeating: 0, count: w * h)
        varn = Array(repeating: 0, count: w * h)
        thick = Array(repeating: 0, count: w * h)
    }

    @inline(__always) func index(_ x: Int, _ y: Int) -> Int {
        if wrap {
            let xx = ((x % w) + w) % w, yy = ((y % h) + h) % h
            return yy * w + xx
        }
        if x < 0 || y < 0 || x >= w || y >= h { return -1 }
        return y * w + x
    }

    /// Elongated shape (leaf, needle, blade, pinna). `profile(t)` returns the half-width fraction along the shape.
    func shape(base: float2, angle: Float, length: Float, width: Float, profile: (Float) -> Float,
               color: float3, h0: Float, dome: Float, variation: Float = 0, translucency: Float = 0,
               vein: Float = 0, opacity: Float = 1, edgeDark: Float = 0.12) {
        let dir = float2(cos(angle), sin(angle))
        let perp = float2(-dir.y, dir.x)
        let hw = width * 0.5
        let tipP = base + dir * length
        let cs = [base + perp * hw, base - perp * hw, tipP + perp * hw, tipP - perp * hw]
        let minX = Int(floor(cs.map { $0.x }.min()!)) - 1, maxX = Int(ceil(cs.map { $0.x }.max()!)) + 1
        let minY = Int(floor(cs.map { $0.y }.min()!)) - 1, maxY = Int(ceil(cs.map { $0.y }.max()!)) + 1
        if !wrap && (maxX < 0 || maxY < 0 || minX >= w || minY >= h) { return }
        let invLen = 1 / max(length, 0.001)
        for y in minY...maxY {
            for x in minX...maxX {
                let p = float2(Float(x) + 0.5, Float(y) + 0.5) - base
                let along = simd_dot(p, dir)
                if along < -1 || along > length + 1 { continue }
                let t = along * invLen
                let tc = min(max(t, 0), 1)
                let half = profile(tc) * hw
                let across = simd_dot(p, perp)
                var cov = min(max(half - abs(across) + 0.5, 0), 1)
                if along < 0 { cov *= max(0, 1 + along) }
                if along > length { cov *= max(0, 1 - (along - length)) }
                cov *= opacity
                if cov <= 0.002 { continue }
                let idx = index(x, y)
                if idx < 0 { continue }
                let s = half > 0.05 ? min(abs(across) / half, 1) : 1
                var c = color
                if vein > 0 {
                    let mr = max(0, 1 - abs(across) / max(1.0, hw * 0.07))
                    c = c * (1 + mr * 0.35 * vein)
                    let vv = 0.5 + 0.5 * cos((tc * 11 - s * 1.7) * 2 * .pi)
                    c *= 1 - vein * 0.14 * pow(vv, 10)
                }
                c *= (1 - edgeDark) + edgeDark * (1 - s * s)
                let hh = h0 + dome * sqrt(max(0, 1 - s * s)) * max(profile(tc), 0.2)
                col[idx] = col[idx] + (c - col[idx]) * cov
                alpha[idx] = alpha[idx] + (1 - alpha[idx]) * cov
                height[idx] = height[idx] + (hh - height[idx]) * cov
                if cov > 0.5 {
                    varn[idx] = variation
                    thick[idx] = translucency * (0.7 + 0.3 * s)
                }
            }
        }
    }

    /// Tapered capsule (twigs, stems, needles).
    func line(_ p0: float2, _ p1: float2, r0: Float, r1: Float, color: float3, h0: Float, dome: Float,
              variation: Float = 0, translucency: Float = 0) {
        let d = p1 - p0
        let len2 = max(simd_length_squared(d), 1e-6)
        let rMax = max(r0, r1) + 1
        let minX = Int(floor(min(p0.x, p1.x) - rMax)), maxX = Int(ceil(max(p0.x, p1.x) + rMax))
        let minY = Int(floor(min(p0.y, p1.y) - rMax)), maxY = Int(ceil(max(p0.y, p1.y) + rMax))
        if !wrap && (maxX < 0 || maxY < 0 || minX >= w || minY >= h) { return }
        for y in minY...maxY {
            for x in minX...maxX {
                let p = float2(Float(x) + 0.5, Float(y) + 0.5)
                let t = min(max(simd_dot(p - p0, d) / len2, 0), 1)
                let r = r0 + (r1 - r0) * t
                let dist = simd_length(p - (p0 + d * t))
                let cov = min(max(r - dist + 0.5, 0), 1)
                if cov <= 0.002 { continue }
                let idx = index(x, y)
                if idx < 0 { continue }
                let s = min(dist / max(r, 0.3), 1)
                let c = color * (0.8 + 0.2 * (1 - s))
                let hh = h0 + dome * sqrt(max(0, 1 - s * s)) * r
                col[idx] = col[idx] + (c - col[idx]) * cov
                alpha[idx] = alpha[idx] + (1 - alpha[idx]) * cov
                height[idx] = height[idx] + (hh - height[idx]) * cov
                if cov > 0.5 { varn[idx] = variation; thick[idx] = translucency }
            }
        }
    }

    func polyline(_ pts: [float2], r0: Float, r1: Float, color: float3, h0: Float, dome: Float) {
        guard pts.count > 1 else { return }
        for i in 0..<(pts.count - 1) {
            let a = Float(i) / Float(pts.count - 1), b = Float(i + 1) / Float(pts.count - 1)
            line(pts[i], pts[i + 1], r0: mix(r0, r1, a), r1: mix(r0, r1, b), color: color, h0: h0, dome: dome)
        }
    }

    func fill(_ f: (Float, Float) -> (float3, Float)) {
        for y in 0..<h {
            let v = (Float(y) + 0.5) / Float(h)
            for x in 0..<w {
                let u = (Float(x) + 0.5) / Float(w)
                let (c, hh) = f(u, v)
                col[y * w + x] = c
                height[y * w + x] = hh
            }
        }
    }
}

// MARK: - Encoding helpers

@inline(__always) func linearToSRGB(_ x: Float) -> Float {
    let c = min(max(x, 0), 1)
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1 / 2.4) - 0.055
}
@inline(__always) func u8(_ x: Float) -> UInt8 { UInt8(min(max(x * 255 + 0.5, 0), 255)) }

func boxBlurWrap(_ src: [Float], w: Int, h: Int, radius r: Int) -> [Float] {
    var tmp = [Float](repeating: 0, count: w * h)
    var out = [Float](repeating: 0, count: w * h)
    let inv = 1 / Float(2 * r + 1)
    for y in 0..<h {
        var acc: Float = 0
        for k in -r...r { acc += src[y * w + ((k % w) + w) % w] }
        for x in 0..<w {
            tmp[y * w + x] = acc * inv
            acc += src[y * w + (x + r + 1) % w] - src[y * w + ((x - r) % w + w) % w]
        }
    }
    for x in 0..<w {
        var acc: Float = 0
        for k in -r...r { acc += tmp[(((k % h) + h) % h) * w + x] }
        for y in 0..<h {
            out[y * w + x] = acc * inv
            acc += tmp[((y + r + 1) % h) * w + x] - tmp[(((y - r) % h + h) % h) * w + x]
        }
    }
    return out
}

struct LayerData {
    var w: Int, h: Int
    var albedo: [UInt8]   // RGBA8 (sRGB rgb + alpha/height)
    var extra: [UInt8]    // RGBA8 (normal.xy, roughness/variation, ao/translucency)
    var mips: [(albedo: [UInt8], extra: [UInt8])] = []
}

/// Surface layer (ground / bark): albedo.a = normalized height, extra = (nx, ny, roughness, ao).
func encodeSurface(_ c: Canvas, normalStrength: Float, roughness: Float, roughVar: Float = 0.1, aoStrength: Float = 0.08) -> LayerData {
    let w = c.w, h = c.h
    let blurred = boxBlurWrap(c.height, w: w, h: h, radius: 10)
    var hMin = Float.greatestFiniteMagnitude, hMax = -Float.greatestFiniteMagnitude
    for v in c.height { hMin = min(hMin, v); hMax = max(hMax, v) }
    let hScale = 1 / max(hMax - hMin, 1e-4)
    var alb = [UInt8](repeating: 0, count: w * h * 4)
    var ext = [UInt8](repeating: 0, count: w * h * 4)
    for y in 0..<h {
        for x in 0..<w {
            let i = y * w + x
            let hl = c.height[y * w + (x + w - 1) % w], hr = c.height[y * w + (x + 1) % w]
            let hu = c.height[((y + h - 1) % h) * w + x], hd = c.height[((y + 1) % h) * w + x]
            let n = simd_normalize(float3((hl - hr) * 0.5 * normalStrength, (hu - hd) * 0.5 * normalStrength, 1))
            let cavity = blurred[i] - c.height[i]
            let ao = clampf(1 - cavity * aoStrength, 0.25, 1)
            let hn = (c.height[i] - hMin) * hScale
            let col = c.col[i]
            alb[i * 4 + 0] = u8(linearToSRGB(col.x))
            alb[i * 4 + 1] = u8(linearToSRGB(col.y))
            alb[i * 4 + 2] = u8(linearToSRGB(col.z))
            alb[i * 4 + 3] = u8(hn)
            ext[i * 4 + 0] = u8(n.x * 0.5 + 0.5)
            ext[i * 4 + 1] = u8(n.y * 0.5 + 0.5)
            ext[i * 4 + 2] = u8(clampf(roughness + (0.5 - hn) * roughVar, 0.05, 1))
            ext[i * 4 + 3] = u8(ao)
        }
    }
    return LayerData(w: w, h: h, albedo: alb, extra: ext)
}

/// Foliage card layer with CPU mip chain that preserves alpha-test coverage (so trees don't thin out in the distance).
func encodeFoliage(_ c: Canvas, normalStrength: Float) -> LayerData {
    let w = c.w, h = c.h
    // average leaf color for bleeding into transparent texels
    var avg = float3(0, 0, 0); var cnt: Float = 0
    for i in 0..<(w * h) where c.alpha[i] > 0.5 { avg += c.col[i]; cnt += 1 }
    avg /= max(cnt, 1)

    var colA = [float4](repeating: .zero, count: w * h)
    var extA = [float4](repeating: .zero, count: w * h)
    for y in 0..<h {
        for x in 0..<w {
            let i = y * w + x
            let a = c.alpha[i]
            let rgb = a > 0.02 ? c.col[i] : avg
            colA[i] = float4(rgb.x, rgb.y, rgb.z, a)
            let hl = c.height[y * w + max(x - 1, 0)], hr = c.height[y * w + min(x + 1, w - 1)]
            let hu = c.height[max(y - 1, 0) * w + x], hd = c.height[min(y + 1, h - 1) * w + x]
            var n = simd_normalize(float3((hl - hr) * 0.5 * normalStrength, (hu - hd) * 0.5 * normalStrength, 1))
            if a < 0.5 { n = float3(0, 0, 1) }
            extA[i] = float4(n.x * 0.5 + 0.5, n.y * 0.5 + 0.5, c.varn[i], c.thick[i])
        }
    }
    func coverage(_ arr: [float4], scale: Float) -> Float {
        var k = 0
        for v in arr where v.w * scale > 0.5 { k += 1 }
        return Float(k) / Float(max(arr.count, 1))
    }
    let baseCov = coverage(colA, scale: 1)

    func pack(_ cArr: [float4], _ eArr: [float4]) -> ([UInt8], [UInt8]) {
        var a = [UInt8](repeating: 0, count: cArr.count * 4)
        var e = [UInt8](repeating: 0, count: cArr.count * 4)
        for i in 0..<cArr.count {
            a[i * 4] = u8(linearToSRGB(cArr[i].x)); a[i * 4 + 1] = u8(linearToSRGB(cArr[i].y))
            a[i * 4 + 2] = u8(linearToSRGB(cArr[i].z)); a[i * 4 + 3] = u8(cArr[i].w)
            e[i * 4] = u8(eArr[i].x); e[i * 4 + 1] = u8(eArr[i].y); e[i * 4 + 2] = u8(eArr[i].z); e[i * 4 + 3] = u8(eArr[i].w)
        }
        return (a, e)
    }
    let (a0, e0) = pack(colA, extA)
    var layer = LayerData(w: w, h: h, albedo: a0, extra: e0)

    var cw = w, ch = h, curC = colA, curE = extA
    while cw > 1 || ch > 1 {
        let nw = max(cw / 2, 1), nh = max(ch / 2, 1)
        var nc = [float4](repeating: .zero, count: nw * nh)
        var ne = [float4](repeating: .zero, count: nw * nh)
        for y in 0..<nh {
            for x in 0..<nw {
                var sumC = float3(0, 0, 0), sumA: Float = 0, sumE = float4(0, 0, 0, 0), sumEW: Float = 0
                var plainC = float3(0, 0, 0)
                for dy in 0..<2 {
                    for dx in 0..<2 {
                        let sx = min(x * 2 + dx, cw - 1), sy = min(y * 2 + dy, ch - 1)
                        let v = curC[sy * cw + sx]
                        sumC += float3(v.x, v.y, v.z) * v.w; sumA += v.w
                        plainC += float3(v.x, v.y, v.z)
                        let wgt = v.w + 0.01
                        sumE += curE[sy * cw + sx] * wgt; sumEW += wgt
                    }
                }
                let rgb = sumA > 1e-4 ? sumC / sumA : plainC * 0.25
                nc[y * nw + x] = float4(rgb.x, rgb.y, rgb.z, sumA * 0.25)
                ne[y * nw + x] = sumE / sumEW
            }
        }
        // alpha-coverage preservation
        var lo: Float = 0.5, hi: Float = 6
        for _ in 0..<14 {
            let mid = (lo + hi) * 0.5
            if coverage(nc, scale: mid) < baseCov { lo = mid } else { hi = mid }
        }
        let s = (lo + hi) * 0.5
        for i in 0..<nc.count { nc[i].w = min(nc[i].w * s, 1) }
        let (pa, pe) = pack(nc, ne)
        layer.mips.append((pa, pe))
        curC = nc; curE = ne; cw = nw; ch = nh
    }
    return layer
}

// MARK: - Ground layers (tileable). Index: 0 grass soil, 1 leaf litter, 2 needle litter, 3 moss, 4 rock, 5 dirt

enum LeafKind { case oak, birch, beech }

func leafProfile(_ kind: LeafKind) -> (Float) -> Float {
    switch kind {
    case .oak:
        return { t in
            let b = pow(sin(Float.pi * pow(max(t, 0.001), 1.25)), 0.75)
            return b * (0.6 + 0.4 * abs(sin(t * .pi * 4.5 + 0.35)))
        }
    case .birch:
        return { t in
            let b = pow(sin(Float.pi * pow(max(t, 0.001), 0.62)), 1.1)
            let f = t * 22
            return b * (1 + 0.08 * (f - floor(f) - 0.5))
        }
    case .beech:
        return { t in pow(sin(Float.pi * pow(max(t, 0.001), 0.85)), 0.8) * (1 + 0.05 * sin(t * .pi * 16)) }
    }
}

let needleProfile: (Float) -> Float = { t in min(1, (1 - t) * 3) * min(1, t * 6 + 0.35) }
let bladeProfile: (Float) -> Float = { t in pow(max(1 - t, 0), 0.7) * min(1, t * 8 + 0.2) }

func genGroundGrass(_ N: Int, seed: UInt64) -> Canvas {
    let c = Canvas(w: N, h: N, wrap: true, base: .zero)
    let s = UInt32(truncatingIfNeeded: seed)
    c.fill { u, v in
        let n = fbm(u * 8, v * 8, octaves: 5, period: 8, seed: s)
        let n2 = fbm(u * 48, v * 48, octaves: 3, period: 48, seed: s &+ 7)
        let col = mix(float3(0.05, 0.04, 0.026), float3(0.1, 0.075, 0.045), n * 0.5 + 0.5) * (0.85 + 0.3 * n2)
        return (col, 3 * n + n2)
    }
    var rng = RNG(seed)
    let count = N * N / 36
    for i in 0..<count {
        let p = float2(rng.range(0, Float(N)), rng.range(0, Float(N)))
        let dry = rng.chance(0.18)
        let g = rng.float()
        var col = dry ? mix(float3(0.16, 0.13, 0.06), float3(0.26, 0.22, 0.11), g)
                      : mix(float3(0.035, 0.075, 0.015), float3(0.1, 0.15, 0.035), g)
        col *= rng.range(0.8, 1.15)
        c.shape(base: p, angle: rng.range(0, 2 * .pi), length: rng.range(14, 50), width: rng.range(2.2, 4.5),
                profile: bladeProfile, color: col, h0: 3 + Float(i) / Float(count) * 6, dome: 1.2, edgeDark: 0.25)
    }
    return c
}

func genLeafLitter(_ N: Int, seed: UInt64) -> Canvas {
    let c = Canvas(w: N, h: N, wrap: true, base: .zero)
    let s = UInt32(truncatingIfNeeded: seed)
    c.fill { u, v in
        let n = fbm(u * 6, v * 6, octaves: 5, period: 6, seed: s)
        return (mix(float3(0.03, 0.02, 0.013), float3(0.07, 0.045, 0.028), n * 0.5 + 0.5), n * 3)
    }
    var rng = RNG(seed)
    let palette: [float3] = [float3(0.22, 0.12, 0.05), float3(0.32, 0.18, 0.07), float3(0.42, 0.29, 0.14),
                             float3(0.12, 0.07, 0.035), float3(0.45, 0.22, 0.06), float3(0.5, 0.38, 0.17),
                             float3(0.28, 0.2, 0.11)]
    let count = N * N / 260
    for i in 0..<count {
        let age = Float(i) / Float(count)
        let p = float2(rng.range(0, Float(N)), rng.range(0, Float(N)))
        let k = rng.float()
        let kind: LeafKind = k < 0.3 ? .oak : (k < 0.75 ? .beech : .birch)
        let len = (kind == .birch ? rng.range(28, 45) : rng.range(42, 85))
        var col = palette[rng.int(0, palette.count - 1)] * rng.range(0.75, 1.15)
        col *= mix(0.45, 1.0, age)  // older, deeper leaves are darker / decomposed
        let curl = rng.chance(0.25)
        c.shape(base: p, angle: rng.range(0, 2 * .pi), length: len, width: len * rng.range(0.42, 0.65),
                profile: leafProfile(kind), color: col, h0: 2 + age * 10, dome: curl ? 6 : 2.2,
                vein: 0.8, edgeDark: curl ? 0.45 : 0.15)
    }
    for _ in 0..<(N * N / 9000) {
        let p = float2(rng.range(0, Float(N)), rng.range(0, Float(N)))
        let a = rng.range(0, 2 * .pi), l = rng.range(50, 240)
        let q = p + float2(cos(a), sin(a)) * l
        let col = mix(float3(0.12, 0.08, 0.05), float3(0.25, 0.2, 0.15), rng.float())
        c.line(p, q, r0: rng.range(1.8, 3.8), r1: 1.2, color: col, h0: 12, dome: 1)
    }
    return c
}

func genNeedleLitter(_ N: Int, seed: UInt64) -> Canvas {
    let c = Canvas(w: N, h: N, wrap: true, base: .zero)
    let s = UInt32(truncatingIfNeeded: seed)
    c.fill { u, v in
        let n = fbm(u * 6, v * 6, octaves: 5, period: 6, seed: s)
        return (mix(float3(0.04, 0.022, 0.012), float3(0.09, 0.05, 0.028), n * 0.5 + 0.5), n * 2)
    }
    var rng = RNG(seed)
    let pal: [float3] = [float3(0.3, 0.15, 0.07), float3(0.4, 0.23, 0.11), float3(0.2, 0.11, 0.055),
                         float3(0.3, 0.26, 0.2), float3(0.48, 0.3, 0.15)]
    let count = N * N / 22
    for i in 0..<count {
        let age = Float(i) / Float(count)
        let p = float2(rng.range(0, Float(N)), rng.range(0, Float(N)))
        let l = rng.chance(0.4) ? rng.range(24, 44) : rng.range(8, 18)
        let col = pal[rng.int(0, pal.count - 1)] * rng.range(0.7, 1.1) * mix(0.5, 1.0, age)
        c.shape(base: p, angle: rng.range(0, 2 * .pi), length: l, width: rng.range(1.7, 2.6),
                profile: needleProfile, color: col, h0: 1 + age * 8, dome: 1.0, edgeDark: 0.3)
    }
    // cones
    for _ in 0..<max(2, N * N / 90000) {
        let p = float2(rng.range(0, Float(N)), rng.range(0, Float(N)))
        let a = rng.range(0, 2 * .pi), l = rng.range(40, 70)
        c.shape(base: p, angle: a, length: l, width: l * 0.45, profile: { t in pow(sin(.pi * pow(max(t, 0.001), 0.8)), 0.6) },
                color: float3(0.2, 0.11, 0.055), h0: 9, dome: 12, edgeDark: 0.5)
        for k in 0..<10 {
            let t = Float(k) / 10 + 0.05
            let q = p + float2(cos(a), sin(a)) * l * t
            c.line(q - float2(-sin(a), cos(a)) * l * 0.18, q + float2(-sin(a), cos(a)) * l * 0.18, r0: 1, r1: 1,
                   color: float3(0.09, 0.05, 0.03), h0: 18, dome: 0.5)
        }
    }
    for _ in 0..<(N * N / 15000) {
        let p = float2(rng.range(0, Float(N)), rng.range(0, Float(N)))
        let a = rng.range(0, 2 * .pi), l = rng.range(40, 180)
        c.line(p, p + float2(cos(a), sin(a)) * l, r0: rng.range(1.5, 3), r1: 1, color: float3(0.18, 0.13, 0.09), h0: 10, dome: 1)
    }
    return c
}

func genMoss(_ N: Int, seed: UInt64) -> Canvas {
    let c = Canvas(w: N, h: N, wrap: true, base: .zero)
    let s = UInt32(truncatingIfNeeded: seed)
    c.fill { u, v in
        let n1 = fbm(u * 12, v * 12, octaves: 5, period: 12, seed: s)
        let (f1, _, id) = voronoi(u * 28, v * 28, period: 28, seed: s &+ 3)
        let clump = max(0, 1 - f1 * 1.4)
        let fine = fbm(u * 160, v * 160, octaves: 2, period: 160, seed: s &+ 9)
        let t = smoothstep(-0.35, 0.6, n1 * 0.8 + fine * 0.5 + clump * 0.4 + (id - 0.5) * 0.3)
        var col = mix(float3(0.03, 0.05, 0.012), float3(0.13, 0.2, 0.03), t)
        col += float3(0.04, 0.035, 0.0) * max(fine, 0)
        return (col, clump * 10 + fine * 2.5 + n1 * 4)
    }
    return c
}

func genRock(_ N: Int, seed: UInt64, lichen: Float) -> Canvas {
    let c = Canvas(w: N, h: N, wrap: true, base: .zero)
    let s = UInt32(truncatingIfNeeded: seed)
    c.fill { u, v in
        let n = fbm(u * 5, v * 5, octaves: 7, period: 5, seed: s)
        let wu = u + fbm(u * 3, v * 3, octaves: 3, period: 3, seed: s &+ 40) * 0.08
        let wv = v + fbm(u * 3 + 5, v * 3, octaves: 3, period: 3, seed: s &+ 41) * 0.08
        let (f1, f2, id) = voronoi(wu * 3, wv * 3, period: 3, seed: s &+ 11)
        let crackMask = smoothstep(0.1, 0.4, fbm(u * 5, v * 5, octaves: 3, period: 5, seed: s &+ 42))
        let crack = (1 - smoothstep(0, 0.025, f2 - f1)) * crackMask
        let speck = gradNoise(u * 300, v * 300, period: 300, seed: s &+ 5)
        var col = mix(float3(0.13, 0.125, 0.115), float3(0.32, 0.31, 0.29), smoothstep(-0.6, 0.6, n + (id - 0.5) * 0.4))
        col *= 0.85 + 0.25 * speck
        let l = fbm(u * 9, v * 9, octaves: 4, period: 9, seed: s &+ 21)
        if l > 0.35 - lichen * 0.3 { col = mix(col, float3(0.42, 0.45, 0.36), smoothstep(0.35 - lichen * 0.3, 0.55, l) * 0.8) }
        let l2 = fbm(u * 14, v * 14, octaves: 3, period: 14, seed: s &+ 31)
        if l2 > 0.55 { col = mix(col, float3(0.45, 0.3, 0.1), 0.6) }
        col = mix(col, float3(0.05, 0.05, 0.045), crack * 0.6)
        return (col, n * 34 + (1 - f1) * 3 - crack * 8 + speck * 0.8)
    }
    return c
}

func genDirt(_ N: Int, seed: UInt64) -> Canvas {
    let c = Canvas(w: N, h: N, wrap: true, base: .zero)
    let s = UInt32(truncatingIfNeeded: seed)
    c.fill { u, v in
        let n = fbm(u * 7, v * 7, octaves: 6, period: 7, seed: s)
        let n2 = gradNoise(u * 200, v * 200, period: 200, seed: s &+ 1)
        return (mix(float3(0.07, 0.05, 0.032), float3(0.16, 0.115, 0.07), n * 0.5 + 0.5) * (0.9 + 0.2 * n2), n * 6 + n2)
    }
    var rng = RNG(seed)
    for _ in 0..<(N * N / 700) {
        let p = float2(rng.range(0, Float(N)), rng.range(0, Float(N)))
        let r = rng.range(3, 14)
        let col = mix(float3(0.12, 0.1, 0.08), float3(0.32, 0.29, 0.25), rng.float())
        c.shape(base: p, angle: rng.range(0, 2 * .pi), length: r * 2, width: r * rng.range(1.2, 2),
                profile: { t in sqrt(max(0, sin(.pi * t))) }, color: col, h0: 4, dome: r * 0.6, edgeDark: 0.5)
    }
    for _ in 0..<(N * N / 40000) {
        var p = float2(rng.range(0, Float(N)), rng.range(0, Float(N)))
        var a = rng.range(0, 2 * .pi)
        var pts = [p]
        for _ in 0..<8 { a += rng.range(-0.4, 0.4); p += float2(cos(a), sin(a)) * 30; pts.append(p) }
        c.polyline(pts, r0: 4, r1: 1.5, color: float3(0.14, 0.09, 0.06), h0: 6, dome: 1)
    }
    return c
}

// MARK: - Bark layers. Index: 0 spruce, 1 pine, 2 oak, 3 birch, 4 redwood, 5 beech, 6 rock

func genBark(_ kind: Int, _ N: Int, seed: UInt64) -> Canvas {
    let c = Canvas(w: N, h: N, wrap: true, base: .zero)
    let s = UInt32(truncatingIfNeeded: seed)
    switch kind {
    case 0: // spruce: thin, irregular, flaky scales with a vertical grain
        c.fill { u, v in
            let wu = u + fbm(u * 6, v * 3, octaves: 3, px: 6, py: 3, seed: s &+ 20) * 0.025
            let wv = v + fbm(u * 6 + 3, v * 3, octaves: 3, px: 6, py: 3, seed: s &+ 21) * 0.02
            let (f1, f2, id) = voronoi(wu * 34, wv * 44, px: 34, py: 44, seed: s)
            let edge = smoothstep(0.0, 0.22, f2 - f1)
            let (g1, g2, gid) = voronoi(wu * 13, wv * 20, px: 13, py: 20, seed: s &+ 5)
            let plate = smoothstep(0.0, 0.12, g2 - g1)
            let n = fbm(u * 10, v * 10, octaves: 5, period: 10, seed: s &+ 1)
            let fine = gradNoise(u * 180, v * 60, px: 180, py: 60, seed: s &+ 2)
            var col = mix(float3(0.1, 0.075, 0.055), float3(0.26, 0.2, 0.15), saturate(id * 0.35 + gid * 0.35 + n * 0.4 + 0.15))
            col *= 0.9 + 0.15 * fine
            col = mix(col * 0.45, col, edge * 0.5 + plate * 0.5)
            let lich = fbm(u * 7, v * 7, octaves: 4, period: 7, seed: s &+ 9)
            if lich > 0.32 { col = mix(col, float3(0.36, 0.38, 0.31), smoothstep(0.32, 0.55, lich) * 0.7) }
            let resin = fbm(u * 20, v * 8, octaves: 2, px: 20, py: 8, seed: s &+ 13)
            if resin > 0.6 { col = mix(col, float3(0.32, 0.2, 0.08), (resin - 0.6) * 1.2) }
            return (col, edge * 2 + plate * 3.5 + (1 - f1) * 1.5 + n * 3 + fine * 0.6)
        }
    case 1: // scots pine: thick plates with deep cracks
        c.fill { u, v in
            let wu = u + fbm(u * 3, v * 3, octaves: 3, period: 3, seed: s &+ 4) * 0.03
            let (f1, f2, id) = voronoi(wu * 7, v * 4, px: 7, py: 4, seed: s)
            let edge = smoothstep(0, 0.2, f2 - f1)
            let n = fbm(u * 20, v * 12, octaves: 4, px: 20, py: 12, seed: s &+ 1)
            var col = mix(float3(0.2, 0.11, 0.07), float3(0.36, 0.27, 0.22), id * 0.7 + n * 0.3)
            col = mix(float3(0.025, 0.015, 0.01), col, edge)
            return (col, edge * 16 + n * 3 + (1 - f1) * 4)
        }
    case 2: // oak: deep vertical furrows
        c.fill { u, v in
            let warp = fbm(u * 4, v * 2, octaves: 3, px: 4, py: 2, seed: s &+ 3) * 0.06
            let r = ridged(u * 10 + warp * 10, v * 2, octaves: 4, px: 10, py: 2, seed: s)
            let n = fbm(u * 30, v * 15, octaves: 3, px: 30, py: 15, seed: s &+ 1)
            var col = mix(float3(0.035, 0.03, 0.025), float3(0.25, 0.23, 0.2), smoothstep(0.2, 0.8, r) + n * 0.15)
            let moss = fbm(u * 5, v * 5, octaves: 3, period: 5, seed: s &+ 7)
            if moss > 0.25 { col = mix(col, float3(0.1, 0.14, 0.04), smoothstep(0.25, 0.5, moss) * 0.6) }
            return (col, r * 18 + n * 2)
        }
    case 3: // birch: white with lenticels and black patches
        c.fill { u, v in
            let n = fbm(u * 4, v * 8, octaves: 5, px: 4, py: 8, seed: s)
            let p = fbm(u * 3, v * 10, octaves: 4, px: 3, py: 10, seed: s &+ 5)
            var col = mix(float3(0.55, 0.53, 0.49), float3(0.78, 0.76, 0.72), n * 0.5 + 0.5)
            if p > 0.28 { col = mix(col, float3(0.03, 0.03, 0.028), smoothstep(0.28, 0.4, p)) }
            let peel = fbm(u * 6, v * 20, octaves: 3, px: 6, py: 20, seed: s &+ 8)
            if peel > 0.45 { col = mix(col, float3(0.55, 0.42, 0.32), 0.5) }
            return (col, n * 1.5 + (p > 0.28 ? -2 + p * 4 : 0))
        }
        var rng = RNG(seed)
        for _ in 0..<(N * N / 1800) {
            let p = float2(rng.range(0, Float(N)), rng.range(0, Float(N)))
            let l = rng.range(10, 70)
            c.shape(base: p, angle: rng.range(-0.05, 0.05), length: l, width: rng.range(2.5, 6),
                    profile: { t in pow(sin(.pi * t), 0.5) }, color: float3(0.09, 0.08, 0.07) * rng.range(0.6, 1.4),
                    h0: -1, dome: -1, edgeDark: 0.2)
        }
    case 4: // redwood: fibrous reddish strands
        c.fill { u, v in
            let warp = fbm(u * 6, v * 2, octaves: 3, px: 6, py: 2, seed: s &+ 2) * 0.04
            let r = ridged(u * 16 + warp * 16, v * 1, octaves: 5, px: 16, py: 1, seed: s)
            let fib = gradNoise(u * 120 + warp * 60, v * 4, px: 120, py: 4, seed: s &+ 3)
            var col = mix(float3(0.13, 0.05, 0.03), float3(0.42, 0.21, 0.12), smoothstep(0.15, 0.85, r) * 0.8 + fib * 0.15 + 0.1)
            col = mix(col, float3(0.33, 0.27, 0.23), smoothstep(0.75, 1.0, r) * 0.4)
            return (col, r * 20 + fib * 2)
        }
    case 5: // beech: smooth gray, mottled lichen
        c.fill { u, v in
            let n = fbm(u * 5, v * 5, octaves: 6, period: 5, seed: s)
            let l = fbm(u * 8, v * 8, octaves: 4, period: 8, seed: s &+ 6)
            var col = mix(float3(0.24, 0.24, 0.22), float3(0.42, 0.42, 0.39), n * 0.5 + 0.5)
            if l > 0.2 { col = mix(col, float3(0.32, 0.4, 0.26), smoothstep(0.2, 0.45, l) * 0.6) }
            if l < -0.45 { col = mix(col, float3(0.6, 0.6, 0.56), 0.5) }
            let ring = smoothstep(0.92, 1.0, abs(sin(v * .pi * 9 + n)))
            col *= 1 - ring * 0.1
            return (col, n * 1.2 + l * 0.6)
        }
    default:
        return genRock(N, seed: seed, lichen: 0.6)
    }
    return c
}

// MARK: - Foliage cards. Index: 0 spruce, 1 pine, 2 oak, 3 birch, 4 fern, 5 beech, 6 redwood
// Card convention: attachment/base at the bottom of the image (v = 1), tip at the top (v = 0).

func genSpray(_ N: Int, seed: UInt64, redwood: Bool) -> Canvas {
    let c = Canvas(w: N, h: N, wrap: false, base: float3(0.04, 0.07, 0.03), alpha: 0)
    var rng = RNG(seed)
    let F = Float(N)
    let base = float2(F * 0.5, F - 4), tip = float2(F * 0.5 + rng.range(-0.03, 0.03) * F, F * 0.04)
    let bend = rng.range(-0.04, 0.04) * F
    func stem(_ t: Float) -> float2 { mix2(base, tip, t) + float2(sin(t * .pi) * bend, 0) }
    var twigs: [[float2]] = []
    twigs.append((0...12).map { stem(Float($0) / 12) })
    var t: Float = 0.05
    var side: Float = 1
    while t < 0.9 {
        for sd: Float in [-1, 1] {
            let p0 = stem(t)
            let stemDir = simd_normalize(stem(t + 0.01) - p0)
            let ang = atan2(stemDir.y, stemDir.x) + sd * (redwood ? rng.range(1.3, 1.5) : rng.range(0.6, 0.9))
            let len = (redwood ? 0.26 : 0.36) * F * pow(1 - t, 0.7) * rng.range(0.8, 1.1) + 0.04 * F
            let curve = rng.range(-0.25, 0.25)
            var pts: [float2] = []
            for k in 0...5 {
                let kk = Float(k) / 5
                let a = ang + curve * kk
                pts.append(p0 + float2(cos(a), sin(a)) * len * kk)
            }
            twigs.append(pts)
        }
        t += redwood ? 0.055 : 0.052
        side = -side
    }
    // twigs (wood) first
    for (i, tw) in twigs.enumerated() {
        c.polyline(tw, r0: i == 0 ? 5 : 3, r1: 1.5, color: float3(0.11, 0.065, 0.035), h0: 0, dome: 1)
    }
    // needles
    for tw in twigs {
        var total: Float = 0
        for i in 0..<(tw.count - 1) { total += simd_length(tw[i + 1] - tw[i]) }
        var acc: Float = 0
        for i in 0..<(tw.count - 1) {
            let a = tw[i], b = tw[i + 1]
            let seg = simd_length(b - a)
            let dir = (b - a) / max(seg, 0.001)
            let segAng = atan2(dir.y, dir.x)
            var d: Float = 0
            let step: Float = redwood ? 4.5 : 3.0
            while d < seg {
                let p = a + dir * d
                let along = (acc + d) / max(total, 1)
                for sd: Float in [-1, 1] {
                    let fresh = smoothstep(0.7, 1.0, along) * (redwood ? 0.4 : 1)
                    var col = redwood ? mix(float3(0.04, 0.075, 0.025), float3(0.075, 0.115, 0.035), rng.float())
                                      : mix(float3(0.022, 0.05, 0.025), float3(0.05, 0.09, 0.04), rng.float())
                    col = mix(col, float3(0.11, 0.19, 0.06), fresh * rng.range(0.4, 1))
                    let ang = segAng + sd * (redwood ? rng.range(1.25, 1.6) : rng.range(0.45, 1.15))
                    var len = redwood ? mix(34, 16, along) * rng.range(0.85, 1.1) : rng.range(26, 40) * rng.range(0.55, 1.0)
                    len *= Float(N) / 1024
                    c.shape(base: p, angle: ang, length: len, width: (redwood ? 4.0 : 3.6) * Float(N) / 1024,
                            profile: needleProfile, color: col, h0: 2 + rng.float(), dome: 1.2,
                            variation: rng.float(), translucency: 0.55, edgeDark: 0.25)
                }
                d += step * Float(N) / 1024
            }
            acc += seg
        }
    }
    return c
}

@inline(__always) func mix2(_ a: float2, _ b: float2, _ t: Float) -> float2 { a + (b - a) * t }

func genPineTufts(_ N: Int, seed: UInt64) -> Canvas {
    let c = Canvas(w: N, h: N, wrap: false, base: float3(0.04, 0.07, 0.04), alpha: 0)
    var rng = RNG(seed)
    let F = Float(N)
    let base = float2(F * 0.5, F - 4)
    let main = float2(F * 0.5 + rng.range(-0.05, 0.05) * F, F * 0.42)
    var tips: [float2] = [main]
    c.line(base, main, r0: 7, r1: 4, color: float3(0.16, 0.09, 0.05), h0: 0, dome: 1)
    for sd: Float in [-1, 1] {
        let p0 = mix2(base, main, rng.range(0.35, 0.6))
        let p1 = p0 + float2(sd * rng.range(0.18, 0.28) * F, -rng.range(0.15, 0.3) * F)
        c.line(p0, p1, r0: 4.5, r1: 3, color: float3(0.16, 0.09, 0.05), h0: 0, dome: 1)
        tips.append(p1)
    }
    tips.append(mix2(base, main, 0.75) + float2(rng.range(-0.1, 0.1) * F, 0))
    for tp in tips {
        for _ in 0..<95 {
            var d = rng.unitVector()
            d.y = abs(d.y) * 1.3 + 0.2
            d = simd_normalize(d)
            let proj = float2(d.x, -d.y)
            let pl = simd_length(proj)
            let L = rng.range(0.17, 0.25) * F * pl
            let ang = atan2(proj.y, proj.x)
            let col = mix(float3(0.03, 0.06, 0.035), float3(0.065, 0.1, 0.055), rng.float())
            c.shape(base: tp, angle: ang, length: L, width: 2.6 * F / 1024, profile: needleProfile, color: col,
                    h0: 2 + rng.float() * 2, dome: 1, variation: rng.float(), translucency: 0.5, edgeDark: 0.2)
        }
    }
    return c
}

func genBroadleaf(_ N: Int, seed: UInt64, kind: LeafKind) -> Canvas {
    let baseCol: float3 = kind == .oak ? float3(0.055, 0.1, 0.025) : (kind == .birch ? float3(0.075, 0.13, 0.03) : float3(0.065, 0.12, 0.03))
    let c = Canvas(w: N, h: N, wrap: false, base: baseCol, alpha: 0)
    var rng = RNG(seed)
    let F = Float(N), S = F / 1024
    let base = float2(F * 0.5, F - 4)
    let tip = float2(F * 0.5 + rng.range(-60, 60) * S, F * 0.16)
    struct Leaf { var p: float2; var a: Float; var len: Float; var wid: Float; var petiole: Float }
    var leaves: [Leaf] = []
    let (nodes, len0, wr, pet): (Int, Float, Float, Float) = {
        switch kind {
        case .oak: return (7, 270, 0.56, 10)
        case .birch: return (13, 140, 0.68, 26)
        case .beech: return (9, 200, 0.6, 12)
        }
    }()
    func addBranch(_ p0: float2, _ p1: float2, _ n: Int, _ lenScale: Float, width: Float) {
        c.line(p0, p1, r0: width * S, r1: width * 0.45 * S, color: float3(0.13, 0.08, 0.05), h0: 0, dome: 1)
        let dir = simd_normalize(p1 - p0)
        let ang0 = atan2(dir.y, dir.x)
        var side: Float = rng.chance(0.5) ? 1 : -1
        for i in 0..<n {
            let t = 0.15 + 0.85 * Float(i) / Float(max(n - 1, 1))
            let p = mix2(p0, p1, t)
            let a = ang0 + side * rng.range(0.45, 1.05)
            let l = len0 * S * rng.range(0.75, 1.05) * (1 - 0.25 * t)
            leaves.append(Leaf(p: p, a: a, len: l, wid: l * wr * rng.range(0.5, 1.0), petiole: pet * S * rng.range(0.7, 1.2)))
            side = -side
        }
        leaves.append(Leaf(p: p1, a: ang0 + rng.range(-0.2, 0.2), len: len0 * S * 0.85, wid: len0 * S * wr * 0.8, petiole: pet * S))
    }
    addBranch(base, tip, nodes, 1, width: 6)
    for _ in 0..<(kind == .birch ? 3 : 2) {
        let t = rng.range(0.25, 0.6)
        let p0 = mix2(base, tip, t)
        let sd: Float = rng.chance(0.5) ? 1 : -1
        let p1 = p0 + float2(sd * rng.range(0.22, 0.34) * F, -rng.range(0.2, 0.32) * F)
        addBranch(p0, simd_clamp(p1, float2(40, 40), float2(F - 40, F - 40)), max(nodes / 2, 3), 0.8, width: 4)
    }
    leaves.shuffle(using: &rng)
    let prof = leafProfile(kind)
    for (i, lf) in leaves.enumerated() {
        let dir = float2(cos(lf.a), sin(lf.a))
        c.line(lf.p, lf.p + dir * lf.petiole, r0: 2 * S, r1: 1.5 * S, color: float3(0.1, 0.12, 0.04), h0: 1, dome: 1)
        var col = baseCol * rng.range(0.8, 1.2)
        col.y *= rng.range(0.92, 1.08)
        col.x *= rng.range(0.85, 1.2)
        c.shape(base: lf.p + dir * lf.petiole, angle: lf.a, length: lf.len, width: lf.wid, profile: prof,
                color: col, h0: 2 + Float(i) * 0.4, dome: 3, variation: rng.float(), translucency: 0.8, vein: 1)
    }
    return c
}

func genFern(_ N: Int, seed: UInt64) -> Canvas {
    let c = Canvas(w: N, h: N, wrap: false, base: float3(0.07, 0.14, 0.03), alpha: 0)
    var rng = RNG(seed)
    let F = Float(N), S = F / 1024
    let base = float2(F * 0.5, F - 2), tip = float2(F * 0.5 + rng.range(-40, 40) * S, 12)
    let bend = rng.range(-30, 30) * S
    func rach(_ t: Float) -> float2 { mix2(base, tip, t) + float2(sin(t * .pi) * bend, 0) }
    c.polyline((0...16).map { rach(Float($0) / 16) }, r0: 5 * S, r1: 1.5 * S, color: float3(0.1, 0.15, 0.04), h0: 0, dome: 1)
    var t: Float = 0.12
    var flip: Float = 1
    let pinnule: (Float) -> Float = { tt in
        pow(max(1 - tt, 0), 0.6) * min(1, tt * 6 + 0.15) * (0.5 + 0.5 * pow(abs(sin(tt * .pi * 11)), 0.3))
    }
    while t < 0.97 {
        for sd: Float in [-1, 1] {
            let tt = t + (sd > 0 ? 0 : 0.012) * flip
            let p = rach(tt)
            let rd = simd_normalize(rach(tt + 0.01) - p)
            let ra = atan2(rd.y, rd.x)
            let env = pow(sin(.pi * pow(min(max((tt - 0.1) / 0.88, 0), 1), 0.5)), 1.1)
            let len = F * 0.4 * max(env, 0.04) * rng.range(0.92, 1.05)
            let a = ra + sd * (Float.pi / 2 - rng.range(0.4, 0.6))
            let col = mix(float3(0.05, 0.12, 0.025), float3(0.09, 0.17, 0.035), rng.float())
            c.shape(base: p, angle: a, length: len, width: max(len * 0.15, 6 * S), profile: pinnule, color: col,
                    h0: 1, dome: 2.5, variation: rng.float(), translucency: 0.85, vein: 0.6)
        }
        t += 0.04
        flip = -flip
    }
    return c
}

// MARK: - Texture library

final class TextureLibrary {
    let groundAlbedo: MTLTexture, groundExtra: MTLTexture
    let barkAlbedo: MTLTexture, barkExtra: MTLTexture
    let leafAlbedo: MTLTexture, leafExtra: MTLTexture

    init(device: MTLDevice, queue: MTLCommandQueue, size: Int = 1024) {
        let N = size
        let lock = NSLock()
        var ground = [LayerData?](repeating: nil, count: 6)
        var bark = [LayerData?](repeating: nil, count: 7)
        var leaves = [LayerData?](repeating: nil, count: 7)
        // 20 independent jobs, run in parallel on all cores
        DispatchQueue.concurrentPerform(iterations: 20) { job in
            switch job {
            case 0..<6:
                let canvas: Canvas
                let rough: Float
                switch job {
                case 0: canvas = genGroundGrass(N, seed: 11); rough = 0.85
                case 1: canvas = genLeafLitter(N, seed: 12); rough = 0.75
                case 2: canvas = genNeedleLitter(N, seed: 13); rough = 0.8
                case 3: canvas = genMoss(N, seed: 14); rough = 0.95
                case 4: canvas = genRock(N, seed: 15, lichen: 0.4); rough = 0.75
                default: canvas = genDirt(N, seed: 16); rough = 0.95
                }
                let d = encodeSurface(canvas, normalStrength: 0.9, roughness: rough)
                lock.lock(); ground[job] = d; lock.unlock()
            case 6..<13:
                let k = job - 6
                let canvas = genBark(k, N, seed: UInt64(100 + k))
                let d = encodeSurface(canvas, normalStrength: k == 3 || k == 5 ? 0.6 : 0.8, roughness: k == 3 ? 0.6 : 0.88, aoStrength: 0.06)
                lock.lock(); bark[k] = d; lock.unlock()
            default:
                let k = job - 13
                let canvas: Canvas
                switch k {
                case 0: canvas = genSpray(N, seed: 200, redwood: false)
                case 1: canvas = genPineTufts(N, seed: 201)
                case 2: canvas = genBroadleaf(N, seed: 202, kind: .oak)
                case 3: canvas = genBroadleaf(N, seed: 203, kind: .birch)
                case 4: canvas = genFern(N, seed: 204)
                case 5: canvas = genBroadleaf(N, seed: 205, kind: .beech)
                default: canvas = genSpray(N, seed: 206, redwood: true)
                }
                let d = encodeFoliage(canvas, normalStrength: 0.6)
                lock.lock(); leaves[k] = d; lock.unlock()
            }
        }
        func make(_ layers: [LayerData], extra: Bool, srgb: Bool, cpuMips: Bool) -> MTLTexture {
            let w = layers[0].w, h = layers[0].h
            let desc = MTLTextureDescriptor()
            desc.textureType = .type2DArray
            desc.pixelFormat = srgb ? .rgba8Unorm_srgb : .rgba8Unorm
            desc.width = w; desc.height = h
            desc.arrayLength = layers.count
            desc.mipmapLevelCount = Int(floor(log2(Double(max(w, h))))) + 1
            desc.usage = [.shaderRead]
            desc.storageMode = .shared
            let tex = device.makeTexture(descriptor: desc)!
            for (i, l) in layers.enumerated() {
                let data = extra ? l.extra : l.albedo
                data.withUnsafeBytes { p in
                    tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, slice: i, withBytes: p.baseAddress!,
                                bytesPerRow: w * 4, bytesPerImage: w * h * 4)
                }
                if cpuMips {
                    var mw = w, mh = h
                    for (m, mip) in l.mips.enumerated() {
                        mw = max(mw / 2, 1); mh = max(mh / 2, 1)
                        let md = extra ? mip.extra : mip.albedo
                        md.withUnsafeBytes { p in
                            tex.replace(region: MTLRegionMake2D(0, 0, mw, mh), mipmapLevel: m + 1, slice: i, withBytes: p.baseAddress!,
                                        bytesPerRow: mw * 4, bytesPerImage: mw * mh * 4)
                        }
                    }
                }
            }
            if !cpuMips {
                let cb = queue.makeCommandBuffer()!
                let blit = cb.makeBlitCommandEncoder()!
                blit.generateMipmaps(for: tex)
                blit.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
            }
            return tex
        }
        let g = ground.map { $0! }, b = bark.map { $0! }, l = leaves.map { $0! }
        groundAlbedo = make(g, extra: false, srgb: true, cpuMips: false)
        groundExtra = make(g, extra: true, srgb: false, cpuMips: false)
        barkAlbedo = make(b, extra: false, srgb: true, cpuMips: false)
        barkExtra = make(b, extra: true, srgb: false, cpuMips: false)
        leafAlbedo = make(l, extra: false, srgb: true, cpuMips: true)
        leafExtra = make(l, extra: true, srgb: false, cpuMips: true)
    }
}
