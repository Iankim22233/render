import Foundation
import simd

/// GPU vertex layout (80 bytes) — must match `TreeVertex` in Shaders/Common.metal.
struct TreeVertex {
    var position: float3
    var normal: float3
    var tangent: float3
    var uv: float2
    var extra: float4   // x: wind flex, y: wind phase, z: material layer, w: baked AO
}

enum Species: Int, CaseIterable {
    case spruce, pine, oak, birch, beech, redwood, fern, shrub, rock, log

    var displayName: String {
        switch self {
        case .spruce: return "Norway Spruce"
        case .pine: return "Scots Pine"
        case .oak: return "Oak"
        case .birch: return "Silver Birch"
        case .beech: return "Beech"
        case .redwood: return "Redwood"
        case .fern: return "Fern"
        case .shrub: return "Shrub"
        case .rock: return "Rock"
        case .log: return "Fallen Log"
        }
    }
    var isConifer: Bool { self == .spruce || self == .pine || self == .redwood }
    var isTree: Bool { rawValue <= Species.redwood.rawValue }
}

struct Branch {
    var points: [float3]
    var radii: [Float]
    var flex: [Float]
    var phase: Float
    var ao: Float
    var capEnds: Bool = false
}

struct LeafSpot {
    var pos: float3
    var up: float3
    var right: float3
    var width: Float
    var length: Float
    var flex: Float
    var phase: Float
    var importance: Float
    var droop: Float
    var cross: Bool
}

final class Skeleton {
    var branches: [Branch] = []
    var leaves: [LeafSpot] = []
    var barkLayer = 0
    var leafLayer = 0
    var barkTileU: Float = 0.7
    var barkTileV: Float = 1.4
    var cardSegments = 1
    var crownBlend: Float = 0.65
    var height: Float = 1
}

struct ModelLOD {
    var barkStart = 0, barkCount = 0
    var leafStart = 0, leafCount = 0
}

struct TreeModel {
    var species: Species
    var lods: [ModelLOD]
    var height: Float
    var radius: Float
    var crownRadius: Float
    var lod0: Float
    var lod1: Float
    var maxDistance: Float
    var castShadow: Bool
    var impostorLayer: Int = -1
    var impostor = float4(0, 0, 0, 0)   // half width, center y, half height, layer
}

// MARK: - Mesh accumulation

final class MeshAccumulator {
    var vertices: [TreeVertex] = []
    var indices: [UInt32] = []

    struct LODSpec { var minRadius: Float; var sides: Float; var keep: Float; var scale: Float; var cross: Bool; var ringStep: Int }
    static let specs: [LODSpec] = [
        LODSpec(minRadius: 0.005, sides: 1.0, keep: 1.0, scale: 1.0, cross: true, ringStep: 1),
        LODSpec(minRadius: 0.035, sides: 0.55, keep: 0.36, scale: 1.8, cross: false, ringStep: 1),
        LODSpec(minRadius: 0.14, sides: 0.4, keep: 0.05, scale: 4.3, cross: false, ringStep: 2),
    ]

    func emitTube(_ b: Branch, sides baseSides: Int, layer: Int, tileU: Float, tileV: Float, ringStep: Int) -> [UInt32] {
        var idx: [UInt32] = []
        var pts: [float3] = [], rad: [Float] = [], flx: [Float] = []
        for i in stride(from: 0, to: b.points.count, by: ringStep) { pts.append(b.points[i]); rad.append(b.radii[i]); flx.append(b.flex[i]) }
        if (b.points.count - 1) % ringStep != 0 { pts.append(b.points.last!); rad.append(b.radii.last!); flx.append(b.flex.last!) }
        let n = pts.count
        guard n >= 2 else { return idx }
        let sides = max(3, baseSides)
        var dirs: [float3] = []
        for i in 0..<n {
            let a = pts[max(i - 1, 0)], c = pts[min(i + 1, n - 1)]
            dirs.append(simd_normalize(c - a))
        }
        var A = anyPerpendicular(dirs[0])
        let uTiles = max(1, (2 * Float.pi * rad[0] / tileU).rounded())
        var v: Float = 0
        let base = UInt32(vertices.count)
        for i in 0..<n {
            let D = dirs[i]
            A = simd_normalize(A - D * simd_dot(A, D))
            let B = simd_cross(D, A)
            if i > 0 { v += simd_length(pts[i] - pts[i - 1]) / tileV }
            for s in 0...sides {
                let th = Float(s) / Float(sides) * 2 * .pi
                let N = A * cos(th) + B * sin(th)
                let T = A * -sin(th) + B * cos(th)
                vertices.append(TreeVertex(position: pts[i] + N * rad[i], normal: N, tangent: T,
                                           uv: float2(Float(s) / Float(sides) * uTiles, v),
                                           extra: float4(flx[i], b.phase, Float(layer), b.ao)))
            }
        }
        let ring = UInt32(sides + 1)
        for i in 0..<(n - 1) {
            for s in 0..<sides {
                let a = base + UInt32(i) * ring + UInt32(s)
                let bb = a + 1, c = a + ring + 1, d = a + ring
                idx += [a, bb, c, a, c, d]
            }
        }
        if b.capEnds {
            for (ri, sign) in [(0, Float(-1)), (n - 1, Float(1))] {
                let center = pts[ri]
                let D = dirs[ri] * sign
                let cbase = UInt32(vertices.count)
                let T = anyPerpendicular(D)
                vertices.append(TreeVertex(position: center, normal: D, tangent: T, uv: float2(0.5, 0.5), extra: float4(flx[ri], b.phase, Float(layer), b.ao * 0.8)))
                for s in 0...sides {
                    let p = vertices[Int(base) + ri * Int(ring) + s].position
                    vertices.append(TreeVertex(position: p, normal: D, tangent: T, uv: float2(Float(s) / Float(sides), 0), extra: float4(flx[ri], b.phase, Float(layer), b.ao * 0.8)))
                }
                for s in 0..<UInt32(sides) {
                    if sign > 0 { idx += [cbase, cbase + 1 + s, cbase + 2 + s] } else { idx += [cbase, cbase + 2 + s, cbase + 1 + s] }
                }
            }
        }
        return idx
    }

    func emitCard(_ sp: LeafSpot, scale: Float, segments: Int, layer: Int, crownC: float3, crownR: float3, blend: Float, rot: Float) -> [UInt32] {
        var idx: [UInt32] = []
        var up = sp.up, right = sp.right
        if rot != 0 { right = rotate(right, axis: up, angle: rot) }
        right = simd_normalize(right - up * simd_dot(right, up))
        up = simd_normalize(up)
        let w = sp.width * scale, L = sp.length * scale
        var geoN = simd_normalize(simd_cross(up, right))
        let crownN0 = simd_normalize((sp.pos - crownC) / max(crownR * crownR, float3(repeating: 0.01)))
        if simd_dot(geoN, crownN0) < 0 { geoN = -geoN }
        let base = UInt32(vertices.count)
        for k in 0...segments {
            let t = Float(k) / Float(segments)
            let center = sp.pos + up * (L * t) + float3(0, -1, 0) * (sp.droop * L * t * t)
            for sd in [Float(-1), 1] {
                let p = center + right * (w * 0.5 * sd)
                let cn = simd_normalize((p - crownC) / max(crownR * crownR, float3(repeating: 0.01)))
                let n = simd_normalize(geoN * (1 - blend) + cn * blend)
                let rr = simd_length((p - crownC) / max(crownR, float3(repeating: 0.01)))
                var ao = clampf(0.28 + 0.72 * smoothstep(0.05, 1.05, rr), 0.25, 1)
                ao *= clampf(0.82 + 0.25 * (p.y - crownC.y) / max(crownR.y, 0.1), 0.6, 1.05)
                vertices.append(TreeVertex(position: p, normal: n, tangent: right,
                                           uv: float2(sd < 0 ? 0 : 1, 1 - t),
                                           extra: float4(sp.flex + t * 0.12 * L, sp.phase, Float(layer), min(ao, 1))))
            }
        }
        for k in 0..<UInt32(segments) {
            let a = base + k * 2
            idx += [a, a + 1, a + 3, a, a + 3, a + 2]
        }
        return idx
    }

    func emit(_ sk: Skeleton, lod: Int) -> ModelLOD {
        let spec = MeshAccumulator.specs[lod]
        var out = ModelLOD()
        var bark: [UInt32] = []
        for b in sk.branches where b.radii[0] >= spec.minRadius {
            let r = b.radii[0]
            let s0: Int = r > 0.6 ? 18 : (r > 0.25 ? 14 : (r > 0.1 ? 10 : (r > 0.04 ? 6 : (r > 0.02 ? 4 : 3))))
            bark += emitTube(b, sides: Int((Float(s0) * spec.sides).rounded()), layer: sk.barkLayer, tileU: sk.barkTileU, tileV: sk.barkTileV,
                             ringStep: b.points.count > 6 ? spec.ringStep : 1)
        }
        out.barkStart = indices.count
        out.barkCount = bark.count
        indices += bark

        // leaf cards
        var crownC = float3(0, sk.height * 0.6, 0), crownR = float3(1, 1, 1) * sk.height * 0.3
        if !sk.leaves.isEmpty {
            var m = float3(0, 0, 0)
            for l in sk.leaves { m += l.pos }
            m /= Float(sk.leaves.count)
            var v = float3(0, 0, 0)
            for l in sk.leaves { let d = l.pos - m; v += d * d }
            v /= Float(sk.leaves.count)
            crownC = m
            crownR = simd_max(float3(sqrt(v.x), sqrt(v.y), sqrt(v.z)) * 1.65, float3(repeating: 0.3))
        }
        var leaf: [UInt32] = []
        let segs = lod == 0 ? sk.cardSegments : max(1, sk.cardSegments - 1)
        for sp in sk.leaves where sp.importance < spec.keep {
            leaf += emitCard(sp, scale: spec.scale, segments: segs, layer: sk.leafLayer, crownC: crownC, crownR: crownR, blend: sk.crownBlend, rot: 0)
            if spec.cross && sp.cross {
                leaf += emitCard(sp, scale: 0.85, segments: segs, layer: sk.leafLayer, crownC: crownC, crownR: crownR, blend: sk.crownBlend, rot: .pi / 2)
            }
        }
        out.leafStart = indices.count
        out.leafCount = leaf.count
        indices += leaf
        return out
    }

    func emitRaw(bark: [TreeVertex], barkIdx: [UInt32]) -> ModelLOD {
        let base = UInt32(vertices.count)
        vertices += bark
        var out = ModelLOD()
        out.barkStart = indices.count
        out.barkCount = barkIdx.count
        indices += barkIdx.map { $0 + base }
        out.leafStart = indices.count
        return out
    }
}

// MARK: - Species generators

private func trunkBranch(_ rng: inout RNG, height H: Float, baseR: Float, segs: Int, flare: Float, flareDecay: Float,
                         wobble: Float, tipR: Float = 0.015) -> Branch {
    var pts: [float3] = [], rad: [Float] = [], flex: [Float] = []
    let lean = float2(rng.range(-0.012, 0.012), rng.range(-0.012, 0.012))
    let s1 = UInt32(truncatingIfNeeded: rng.next())
    for i in 0...segs {
        let t = Float(i) / Float(segs)
        let y = H * t - 0.6
        let wx = gradNoise(t * 4, 0.5, seed: s1) * wobble, wz = gradNoise(t * 4, 7.5, seed: s1) * wobble
        pts.append(float3(lean.x * y + wx, y, lean.y * y + wz))
        var r = baseR * pow(1 - t, 0.85) + tipR
        r *= 1 + flare * exp(-max(y, 0) * flareDecay)
        rad.append(r)
        flex.append(0.42 * t * t * (H / 25))
    }
    return Branch(points: pts, radii: rad, flex: flex, phase: rng.float(), ao: 0.95)
}

@inline(__always) private func samplePath(_ b: Branch, _ s: Float) -> (float3, float3, Float, Float) {
    let n = b.points.count
    let f = min(max(s, 0), 1) * Float(n - 1)
    let i = min(Int(f), n - 2)
    let t = f - Float(i)
    let p = b.points[i] + (b.points[i + 1] - b.points[i]) * t
    let d = simd_normalize(b.points[i + 1] - b.points[i])
    let r = b.radii[i] + (b.radii[i + 1] - b.radii[i]) * t
    let fl = b.flex[i] + (b.flex[i + 1] - b.flex[i]) * t
    return (p, d, r, fl)
}

func genConifer(_ rng: inout RNG, redwood: Bool, scale: Float = 1) -> Skeleton {
    let sk = Skeleton()
    let H = (redwood ? rng.range(52, 74) : rng.range(19, 30)) * scale
    sk.height = H
    sk.barkLayer = redwood ? 4 : 0
    sk.leafLayer = redwood ? 6 : 0
    sk.barkTileU = redwood ? 1.6 : 0.7
    sk.barkTileV = redwood ? 3.0 : 1.3
    sk.cardSegments = 1
    sk.crownBlend = 0.55
    let baseR = redwood ? H * 0.026 : H * 0.016
    let trunk = trunkBranch(&rng, height: H, baseR: baseR, segs: redwood ? 22 : 16, flare: redwood ? 0.9 : 0.45,
                            flareDecay: redwood ? 0.45 : 1.6, wobble: redwood ? 0.25 : 0.12)
    sk.branches.append(trunk)
    func trunkAt(_ y: Float) -> (float3, Float, Float) {
        let (p, _, r, f) = samplePath(trunk, (y + 0.6) / H)
        return (p, r, f)
    }
    let crownBase = H * (redwood ? rng.range(0.38, 0.5) : rng.range(0.05, 0.2))
    let crownR = H * (redwood ? 0.14 : 0.2)
    // dead lower twigs
    if !redwood {
        var y: Float = 1.5
        while y < crownBase {
            let (tp, tr, tf) = trunkAt(y)
            for _ in 0..<rng.int(0, 2) {
                let az = rng.range(0, 2 * .pi)
                let d = float3(cos(az), rng.range(-0.3, 0.1), sin(az))
                let L = rng.range(0.4, 1.4)
                let p0 = tp + simd_normalize(float3(d.x, 0, d.z)) * tr * 0.8
                sk.branches.append(Branch(points: [p0, p0 + d * L * 0.5, p0 + d * L], radii: [0.018, 0.01, 0.004],
                                          flex: [tf, tf + 0.05, tf + 0.1], phase: rng.float(), ao: 0.6))
            }
            y += rng.range(0.4, 0.9)
        }
    }
    var y = crownBase
    while y < H - 0.5 {
        let t = (y - crownBase) / (H - crownBase)
        let n = rng.int(redwood ? 4 : 4, redwood ? 7 : 7)
        let az0 = rng.range(0, 2 * .pi)
        let maxL = crownR * pow(1 - t, redwood ? 0.65 : 0.92) * rng.range(0.85, 1.1) + 0.35
        let (tp, tr, tf) = trunkAt(y)
        for k in 0..<n {
            let az = az0 + 2 * .pi * Float(k) / Float(n) + rng.range(-0.35, 0.35)
            let L = maxL * rng.range(0.78, 1.05)
            let pitch = (redwood ? mix(-0.18, 0.15, t) : mix(-0.3, 0.38, t)) + rng.range(-0.12, 0.12)
            let dir0 = float3(cos(az) * cos(pitch), sin(pitch), sin(az) * cos(pitch))
            let horiz = simd_normalize(float3(dir0.x, 0, dir0.z))
            let start = tp + horiz * tr * 0.7
            var pts: [float3] = [], rad: [Float] = [], flex: [Float] = []
            let r0 = min(tr * 0.35, redwood ? 0.02 + 0.014 * L : 0.01 + 0.0065 * L)
            for j in 0..<5 {
                let s = Float(j) / 4
                var p = start + dir0 * (L * s)
                p.y -= L * (redwood ? 0.22 : 0.17) * s * s
                if !redwood { p.y += L * 0.07 * s * s * s }
                pts.append(p)
                rad.append(mix(r0, 0.005, s))
                flex.append(tf + s * (0.12 + 0.07 * L))
            }
            let br = Branch(points: pts, radii: rad, flex: flex, phase: rng.float(), ao: 0.55 + 0.3 * t)
            sk.branches.append(br)
            let sideAxis = simd_normalize(simd_cross(float3(0, 1, 0), horiz))
            _ = sideAxis
            var s: Float = redwood ? 0.12 : (t < 0.3 ? 0.28 : 0.1)
            let spacing: Float = redwood ? 0.5 : 0.3
            while s <= 1.0 {
                let (p, d, _, fl) = samplePath(br, s)
                let up = rotate(d, axis: float3(0, 1, 0), angle: rng.range(-0.55, 0.55))
                var right = simd_normalize(simd_cross(up, float3(0, 1, 0)))
                right = rotate(right, axis: up, angle: rng.range(-0.8, 0.8))
                let w = (redwood ? 1.3 : 0.85) * rng.range(0.85, 1.15) * mix(1, 0.62, t)
                sk.leaves.append(LeafSpot(pos: p - up * w * 0.2, up: up, right: right, width: w, length: w * 1.25,
                                          flex: fl, phase: rng.float(), importance: rng.float(), droop: redwood ? 0.3 : 0.14, cross: true))
                if !redwood && rng.chance(0.3) {
                    let hang = simd_normalize(d * 0.45 + float3(0, -1, 0))
                    let hr = simd_normalize(simd_cross(hang, horiz))
                    sk.leaves.append(LeafSpot(pos: p, up: hang, right: hr, width: w * 0.7, length: w * 0.9, flex: fl + 0.05,
                                              phase: rng.float(), importance: rng.float(), droop: 0, cross: false))
                }
                s += spacing / L
            }
        }
        y += redwood ? rng.range(0.45, 0.7) : rng.range(0.34, 0.5)
    }
    // leader
    for j in 0..<5 {
        let (tp, _, tf) = trunkAt(H - 1.6 + Float(j) * 0.35)
        let up = simd_normalize(float3(rng.range(-0.15, 0.15), 1, rng.range(-0.15, 0.15)))
        let az = rng.range(0, 2 * .pi)
        sk.leaves.append(LeafSpot(pos: tp, up: up, right: simd_normalize(simd_cross(up, float3(cos(az), 0, sin(az)))),
                                  width: 0.45, length: 0.6, flex: tf, phase: rng.float(), importance: rng.float() * 0.5, droop: 0, cross: true))
    }
    return sk
}

func genPine(_ rng: inout RNG) -> Skeleton {
    let sk = Skeleton()
    let H = rng.range(18, 28)
    sk.height = H
    sk.barkLayer = 1; sk.leafLayer = 1
    sk.barkTileU = 0.9; sk.barkTileV = 1.6
    sk.cardSegments = 1
    sk.crownBlend = 0.6
    let trunk = trunkBranch(&rng, height: H, baseR: H * 0.015, segs: 16, flare: 0.35, flareDecay: 1.8, wobble: 0.4)
    sk.branches.append(trunk)
    let crownBase = H * rng.range(0.5, 0.64)

    // A dense clump of needle tufts at a shoot tip (Scots pine foliage grows in clumps at branch ends).
    func clump(_ p: float3, _ dir: float3, _ flex: Float, _ n: Int) {
        for _ in 0..<n {
            let up = simd_normalize(dir * 0.5 + float3(0, 0.9, 0) + rng.unitVector() * 0.45)
            let right = simd_normalize(simd_cross(up, rng.unitVector()))
            let sz = rng.range(1.0, 1.35)
            sk.leaves.append(LeafSpot(pos: p - dir * rng.range(0, 0.45) + rng.unitVector() * 0.2, up: up, right: right,
                                      width: sz, length: sz, flex: flex, phase: rng.float(), importance: rng.float(), droop: 0, cross: true))
        }
    }

    let nMain = rng.int(16, 22)
    for k in 0..<nMain {
        let t = Float(k) / Float(nMain - 1)
        let y = mix(crownBase, H - 1.0, pow(t, 0.8))
        let (tp, _, tr, tf) = samplePath(trunk, (y + 0.6) / H)
        let az = Float(k) * 2.4 + rng.range(-0.4, 0.4)
        let L = H * 0.24 * (1 - t * 0.55) * rng.range(0.75, 1.1)
        let pitch0 = rng.range(0.05, 0.4) + t * 0.35
        var pts: [float3] = [], rad: [Float] = [], flex: [Float] = []
        var p = tp + float3(cos(az), 0, sin(az)) * tr * 0.6
        let r0 = min(tr * 0.5, 0.025 + 0.018 * L)
        for j in 0..<6 {
            let s = Float(j) / 5
            pts.append(p); rad.append(mix(r0, 0.01, s)); flex.append(tf + s * (0.15 + 0.06 * L))
            let pitch = pitch0 + s * 0.5 + rng.range(-0.18, 0.18)
            let a = az + rng.range(-0.25, 0.25)
            p += float3(cos(a) * cos(pitch), sin(pitch), sin(a) * cos(pitch)) * (L / 5)
        }
        let br = Branch(points: pts, radii: rad, flex: flex, phase: rng.float(), ao: 0.7)
        sk.branches.append(br)
        clump(pts.last!, simd_normalize(pts[5] - pts[4]), flex.last!, 6)
        for _ in 0..<rng.int(4, 6) {
            let s = rng.range(0.25, 0.9)
            let (sp, sd, sr, sf) = samplePath(br, s)
            let dir = simd_normalize(rotate(sd, axis: float3(0, 1, 0), angle: rng.range(-1.0, 1.0)) + float3(0, 0.3, 0))
            let L2 = L * rng.range(0.35, 0.55)
            let e = sp + dir * L2
            let sub = Branch(points: [sp, sp + dir * L2 * 0.5 + float3(0, 0.06, 0), e], radii: [sr * 0.6, sr * 0.4, 0.008],
                             flex: [sf, sf + 0.06, sf + 0.12], phase: rng.float(), ao: 0.75)
            sk.branches.append(sub)
            clump(e, dir, sf + 0.12, 5)
            for _ in 0..<rng.int(1, 2) {
                let (mp, md, mr, mf) = samplePath(sub, rng.range(0.35, 0.7))
                let d2 = simd_normalize(rotate(md, axis: float3(0, 1, 0), angle: rng.range(-1.1, 1.1)) + float3(0, 0.4, 0))
                let e2 = mp + d2 * L2 * 0.5
                sk.branches.append(Branch(points: [mp, e2], radii: [mr * 0.6, 0.006], flex: [mf, mf + 0.1], phase: rng.float(), ao: 0.75))
                clump(e2, d2, mf + 0.1, 4)
            }
        }
        var sAlong: Float = 0.45
        while sAlong < 0.95 {
            let (ap, ad, _, af) = samplePath(br, sAlong)
            let up = simd_normalize(ad * 0.5 + float3(0, 0.8, 0) + rng.unitVector() * 0.35)
            sk.leaves.append(LeafSpot(pos: ap, up: up, right: simd_normalize(simd_cross(up, rng.unitVector())), width: 0.9 * rng.range(0.85, 1.15),
                                      length: 0.9, flex: af, phase: rng.float(), importance: rng.float(), droop: 0, cross: true))
            sAlong += 0.4 / max(L, 0.5)
        }
    }
    let (top, _, _, tf) = samplePath(trunk, 1)
    clump(top - float3(0, 0.4, 0), float3(0, 1, 0), tf, 7)
    return sk
}

struct BroadParams {
    var H: Float
    var trunkFrac: Float
    var depthMax: Int
    var childMin: Int, childMax: Int
    var angleMin: Float, angleMax: Float
    var lenRatio: Float
    var droop: Float
    var upward: Float
    var leafSize: Float
    var leafSpacing: Float
    var apical: Bool
    var baseR: Float
    var leafDepth = 1     // leaves on branches with depth >= depthMax - leafDepth
    var limbScale: Float = 1
}

func genBroadleaf(_ rng: inout RNG, species: Species) -> Skeleton {
    let sk = Skeleton()
    var p: BroadParams
    switch species {
    case .oak:
        p = BroadParams(H: rng.range(15, 22), trunkFrac: rng.range(0.22, 0.32), depthMax: 4, childMin: 3, childMax: 4,
                        angleMin: 0.55, angleMax: 0.95, lenRatio: 0.64, droop: 0.15, upward: 0.25, leafSize: 0.95, leafSpacing: 0.42, apical: false, baseR: 0)
        sk.barkLayer = 2; sk.leafLayer = 2
    case .birch:
        p = BroadParams(H: rng.range(15, 23), trunkFrac: rng.range(0.25, 0.4), depthMax: 4, childMin: 3, childMax: 4,
                        angleMin: 0.45, angleMax: 0.8, lenRatio: 0.6, droop: 0.7, upward: 0.2, leafSize: 0.7, leafSpacing: 0.3, apical: true, baseR: 0)
        sk.barkLayer = 3; sk.leafLayer = 3
        p.leafDepth = 2; p.limbScale = 0.72
    case .shrub:
        p = BroadParams(H: rng.range(1.6, 3.2), trunkFrac: 0.02, depthMax: 3, childMin: 3, childMax: 5,
                        angleMin: 0.5, angleMax: 1.0, lenRatio: 0.72, droop: 0.2, upward: 0.3, leafSize: 0.55, leafSpacing: 0.25, apical: false, baseR: 0)
        sk.barkLayer = 5; sk.leafLayer = rng.chance(0.5) ? 5 : 2
    default: // beech
        p = BroadParams(H: rng.range(22, 30), trunkFrac: rng.range(0.32, 0.45), depthMax: 4, childMin: 3, childMax: 4,
                        angleMin: 0.4, angleMax: 0.75, lenRatio: 0.66, droop: 0.2, upward: 0.35, leafSize: 0.9, leafSpacing: 0.36, apical: true, baseR: 0)
        sk.barkLayer = 5; sk.leafLayer = 5
    }
    p.baseR = species == .shrub ? 0.04 : p.H * (species == .oak ? 0.026 : 0.017)
    sk.height = p.H
    sk.cardSegments = 2
    sk.crownBlend = 0.7
    sk.barkTileU = species == .birch ? 0.5 : 0.8
    sk.barkTileV = species == .birch ? 1.0 : 1.5

    func grow(start: float3, dir: float3, length: Float, radius: Float, depth: Int, flexBase: Float) {
        let segs = depth == 0 ? 10 : max(3, 7 - depth)
        var pts = [start], rad = [radius], flex = [flexBase]
        var d = dir, pos = start
        let droopK = p.droop * Float(depth) / Float(p.depthMax)
        for i in 1...segs {
            let s = Float(i) / Float(segs)
            d = simd_normalize(d + rng.unitVector() * (depth == 0 ? 0.06 : 0.16) + float3(0, -1, 0) * droopK * 0.35 + float3(0, 1, 0) * p.upward * 0.08)
            pos += d * (length / Float(segs))
            pts.append(pos)
            rad.append(radius * mix(1, 0.3, pow(s, 0.9)) + 0.003)
            flex.append(flexBase + s * length * (0.015 + 0.035 * Float(depth)))
        }
        let br = Branch(points: pts, radii: rad, flex: flex, phase: rng.float(), ao: depth == 0 ? 0.92 : 0.7)
        sk.branches.append(br)
        if depth < p.depthMax {
            let nC = rng.int(p.childMin, p.childMax) + (depth == 0 && !p.apical ? 1 : 0)
            let golden: Float = 2.399
            var rot = rng.range(0, 2 * .pi)
            for c in 0..<nC {
                let sA: Float
                if depth == 0 {
                    sA = p.apical ? mix(p.trunkFrac, 0.92, Float(c) / Float(max(nC - 1, 1))) * rng.range(0.95, 1.05)
                                  : (c == 0 ? 1.0 : rng.range(max(p.trunkFrac, 0.8), 1.0))
                } else {
                    sA = rng.range(0.3, 0.98)
                }
                let (ap, ad, ar, af) = samplePath(br, sA)
                rot += golden
                let perp = rotate(anyPerpendicular(ad), axis: ad, angle: rot)
                var angle = rng.range(p.angleMin, p.angleMax)
                if depth == 0 && !p.apical && c == 0 { angle *= 0.4 }
                var cd = rotate(ad, axis: simd_cross(ad, perp), angle: angle)
                if depth == 0 {
                    let out = simd_normalize(float3(ap.x - start.x, 0, ap.z - start.z) + perp * 0.5)
                    cd = simd_normalize(cd + out * 0.3)
                }
                var cl = length * p.lenRatio * rng.range(0.85, 1.1)
                if depth == 0 { cl = p.H * (1 - p.trunkFrac) * 0.62 * p.limbScale * rng.range(0.8, 1.05) * (p.apical ? mix(1.15, 0.45, (sA - p.trunkFrac) / (1 - p.trunkFrac)) : 1) }
                else { cl *= 1 - 0.35 * sA }
                let cr = min(ar * (depth == 0 ? 0.55 : 0.62), ar)
                grow(start: ap, dir: cd, length: cl, radius: cr, depth: depth + 1, flexBase: af)
            }
        }
        if depth >= p.depthMax - p.leafDepth && depth > 0 {
            var s: Float = depth == p.depthMax ? 0.15 : (depth == p.depthMax - 1 ? 0.45 : 0.6)
            while s <= 1.0 {
                let (lp, ld, _, lf) = samplePath(br, s)
                for _ in 0..<(depth == p.depthMax ? 3 : 2) {
                    let off = rng.unitVector() * p.leafSize * 0.45
                    let up = simd_normalize(ld + rng.unitVector() * 0.9 + float3(0, 0.25 - p.droop * 0.5, 0))
                    let right = simd_normalize(simd_cross(up, rng.unitVector()))
                    let sz = p.leafSize * rng.range(0.8, 1.2)
                    sk.leaves.append(LeafSpot(pos: lp + off, up: up, right: right, width: sz, length: sz * 1.1, flex: lf + 0.05,
                                              phase: rng.float(), importance: rng.float(), droop: 0.2 + p.droop * 0.3, cross: false))
                }
                s += p.leafSpacing / max(length, 0.3)
            }
        }
    }
    let trunkLen = species == .shrub ? p.H * 0.4 : p.H * (p.apical ? 0.97 : p.trunkFrac + 0.05)
    let start = float3(0, -0.5, 0)
    let dir = simd_normalize(float3(rng.range(-0.06, 0.06), 1, rng.range(-0.06, 0.06)))
    if species == .shrub {
        for _ in 0..<rng.int(3, 5) {
            let d = simd_normalize(float3(rng.range(-0.6, 0.6), 1, rng.range(-0.6, 0.6)))
            grow(start: start + float3(rng.range(-0.15, 0.15), 0, rng.range(-0.15, 0.15)), dir: d, length: p.H * rng.range(0.5, 0.75),
                 radius: p.baseR, depth: 1, flexBase: 0)
        }
    } else {
        grow(start: start, dir: dir, length: trunkLen, radius: p.baseR, depth: 0, flexBase: 0)
        // flare the trunk base
        sk.branches[0].radii = sk.branches[0].radii.enumerated().map { i, r in
            let y = sk.branches[0].points[i].y
            return r * (1 + 0.5 * exp(-max(y, 0) * 1.5))
        }
        sk.branches[0].flex = sk.branches[0].points.map { 0.38 * pow(max($0.y, 0) / p.H, 2) * (p.H / 22) }
    }
    return sk
}

func genFernSkeleton(_ rng: inout RNG) -> Skeleton {
    let sk = Skeleton()
    sk.leafLayer = 4
    sk.cardSegments = 3
    sk.crownBlend = 0.35
    sk.height = 1.1
    let n = rng.int(9, 15)
    let az0 = rng.range(0, 2 * .pi)
    for k in 0..<n {
        let az = az0 + 2 * .pi * Float(k) / Float(n) + rng.range(-0.25, 0.25)
        let el = rng.range(0.6, 1.2)
        let L = rng.range(0.75, 1.35)
        let up = float3(cos(az) * cos(el), sin(el), sin(az) * cos(el))
        let right = simd_normalize(simd_cross(up, float3(0, 1, 0)))
        sk.leaves.append(LeafSpot(pos: float3(0, 0.02, 0), up: up, right: rotate(right, axis: up, angle: rng.range(-0.3, 0.3)),
                                  width: L * 0.42, length: L, flex: 0.02, phase: rng.float(), importance: rng.float(),
                                  droop: rng.range(0.45, 0.8), cross: false))
    }
    return sk
}

private func noise3(_ p: float3, _ seed: UInt32) -> Float {
    (gradNoise(p.x, p.y, seed: seed) + gradNoise(p.y + 3.1, p.z, seed: seed &+ 1) + gradNoise(p.z + 7.7, p.x, seed: seed &+ 2)) / 3
}

/// Displaced icosphere rock, emitted directly per LOD.
func genRockMesh(seed: UInt64, lod: Int) -> ([TreeVertex], [UInt32], Float) {
    var rng = RNG(seed)
    let t: Float = (1 + sqrt(5)) / 2
    var verts: [float3] = [float3(-1, t, 0), float3(1, t, 0), float3(-1, -t, 0), float3(1, -t, 0),
                           float3(0, -1, t), float3(0, 1, t), float3(0, -1, -t), float3(0, 1, -t),
                           float3(t, 0, -1), float3(t, 0, 1), float3(-t, 0, -1), float3(-t, 0, 1)].map { simd_normalize($0) }
    var faces: [(Int, Int, Int)] = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
                                    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    let subdiv = [4, 3, 1][lod]
    for _ in 0..<subdiv {
        var cache: [Int: Int] = [:]
        func mid(_ a: Int, _ b: Int) -> Int {
            let key = min(a, b) * 100000 + max(a, b)
            if let v = cache[key] { return v }
            verts.append(simd_normalize((verts[a] + verts[b]) * 0.5))
            cache[key] = verts.count - 1
            return verts.count - 1
        }
        var nf: [(Int, Int, Int)] = []
        for (a, b, c) in faces {
            let ab = mid(a, b), bc = mid(b, c), ca = mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        }
        faces = nf
    }
    let s = UInt32(truncatingIfNeeded: seed)
    let scale = float3(rng.range(0.9, 1.4), rng.range(0.5, 0.85), rng.range(0.8, 1.2))
    let sq = rng.range(0.15, 0.4)
    var pos = verts.map { v -> float3 in
        var r: Float = 1 + 0.32 * noise3(v * 1.3, s) + 0.14 * noise3(v * 3.7, s &+ 9) + 0.06 * noise3(v * 9, s &+ 5) + 0.025 * noise3(v * 21, s &+ 7)
        // faceted, chunky look
        let f = abs(noise3(v * 2.2, s &+ 13))
        r -= f * 0.12
        var p = v * r * scale
        if p.y < -sq { p.y = -sq + (p.y + sq) * 0.3 }
        return p
    }
    var minY = Float.greatestFiniteMagnitude
    for p in pos { minY = min(minY, p.y) }
    pos = pos.map { $0 - float3(0, minY + 0.25 * scale.y, 0) }
    var normals = [float3](repeating: .zero, count: pos.count)
    for (a, b, c) in faces {
        let n = simd_cross(pos[b] - pos[a], pos[c] - pos[a])
        normals[a] += n; normals[b] += n; normals[c] += n
    }
    var out: [TreeVertex] = []
    var maxR: Float = 0
    for i in 0..<pos.count {
        let n = simd_normalize(normals[i])
        let ao = clampf(0.55 + 0.45 * (pos[i].y / (scale.y * 1.2) + 0.3), 0.45, 1)
        out.append(TreeVertex(position: pos[i], normal: n, tangent: anyPerpendicular(n), uv: float2(0, 0),
                              extra: float4(0, 0, 6, ao)))
        maxR = max(maxR, simd_length(pos[i]))
    }
    // Faces are CCW when seen from outside (icosahedron table is CCW-outward).
    var idx: [UInt32] = []
    for (a, b, c) in faces { idx += [UInt32(a), UInt32(b), UInt32(c)] }
    return (out, idx, maxR)
}

func genLogSkeleton(_ rng: inout RNG, barkLayer: Int) -> Skeleton {
    let sk = Skeleton()
    sk.barkLayer = barkLayer
    sk.barkTileU = 0.8; sk.barkTileV = 1.4
    let L = rng.range(4, 11)
    let r = rng.range(0.18, 0.38)
    sk.height = r * 2
    var pts: [float3] = [], rad: [Float] = [], flex: [Float] = []
    let segs = 10
    let s1 = UInt32(truncatingIfNeeded: rng.next())
    for i in 0...segs {
        let t = Float(i) / Float(segs)
        pts.append(float3(-L / 2 + L * t, r * 0.8 + gradNoise(t * 3, 0, seed: s1) * 0.08, gradNoise(t * 3, 5, seed: s1) * 0.25))
        rad.append(r * mix(1, 0.75, t))
        flex.append(0)
    }
    var trunk = Branch(points: pts, radii: rad, flex: flex, phase: 0, ao: 0.8)
    trunk.capEnds = true
    sk.branches.append(trunk)
    for _ in 0..<rng.int(2, 5) {
        let t = rng.range(0.2, 0.9)
        let (p, _, pr, _) = samplePath(trunk, t)
        let az = rng.range(0, 2 * .pi)
        let d = simd_normalize(float3(rng.range(-0.3, 0.3), cos(az), sin(az)))
        let l = rng.range(0.3, 1.2)
        sk.branches.append(Branch(points: [p, p + d * l * 0.5, p + d * l], radii: [pr * 0.3, pr * 0.2, 0.02], flex: [0, 0, 0], phase: 0, ao: 0.7))
    }
    return sk
}

/// Builds all variants for a species. Returns models appended to the accumulator.
func buildModels(species: Species, variants: Int, seedBase: UInt64, acc: MeshAccumulator, logBark: Int = 0) -> [TreeModel] {
    var models: [TreeModel] = []
    for v in 0..<variants {
        var rng = RNG(seedBase &+ UInt64(species.rawValue) &* 1000 &+ UInt64(v) &* 7919)
        if species == .rock {
            var lods: [ModelLOD] = []
            var R: Float = 1
            let rockSeed = rng.next() | 1
            for l in 0..<3 {
                let (vv, ii, r) = genRockMesh(seed: rockSeed, lod: l)
                lods.append(acc.emitRaw(bark: vv, barkIdx: ii))
                R = r
            }
            models.append(TreeModel(species: .rock, lods: lods, height: R, radius: R * 1.1, crownRadius: 0, lod0: 25, lod1: 90, maxDistance: 700, castShadow: true))
            continue
        }
        let sk: Skeleton
        switch species {
        case .spruce: sk = genConifer(&rng, redwood: false)
        case .redwood: sk = genConifer(&rng, redwood: true)
        case .pine: sk = genPine(&rng)
        case .fern: sk = genFernSkeleton(&rng)
        case .log: sk = genLogSkeleton(&rng, barkLayer: logBark)
        default: sk = genBroadleaf(&rng, species: species)
        }
        var lods: [ModelLOD] = []
        let lodCount = (species == .fern || species == .log) ? 2 : 3
        for l in 0..<3 { lods.append(acc.emit(sk, lod: min(l, lodCount - 1))) }
        var maxR: Float = 0, maxY: Float = 0, crown: Float = 0
        for b in sk.branches { for p in b.points { maxR = max(maxR, simd_length(p - float3(0, sk.height * 0.5, 0))); maxY = max(maxY, p.y) } }
        for l in sk.leaves {
            maxR = max(maxR, simd_length(l.pos - float3(0, sk.height * 0.5, 0)) + l.length * 3.2)
            crown = max(crown, simd_length(float2(l.pos.x, l.pos.z)))
        }
        let h = max(sk.height, maxY)
        let sizeK = min(max(h / 22, 0.5), 1.6)
        switch species {
        case .fern:
            models.append(TreeModel(species: species, lods: lods, height: 1.2, radius: 1.6, crownRadius: 1, lod0: 18, lod1: 45, maxDistance: 75, castShadow: true))
        case .shrub:
            models.append(TreeModel(species: species, lods: lods, height: h, radius: maxR, crownRadius: crown, lod0: 22, lod1: 70, maxDistance: 260, castShadow: true))
        case .log:
            models.append(TreeModel(species: species, lods: lods, height: h, radius: maxR + 6, crownRadius: 0, lod0: 30, lod1: 100, maxDistance: 320, castShadow: true))
        default:
            models.append(TreeModel(species: species, lods: lods, height: h, radius: maxR, crownRadius: crown,
                                    lod0: 38 * sizeK, lod1: 130 * sizeK, maxDistance: 5000, castShadow: true))
        }
    }
    return models
}
