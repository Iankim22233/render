import Foundation
import Metal
import simd

/// Per-instance data (32 bytes) — must match `TreeInstance` in the shaders.
struct TreeInstance {
    var posScale: float4   // xyz world position, w uniform scale
    var params: float4     // x rotation (rad), y tint seed, z wind phase, w LOD fade (+v: visible if dither < v, -v: visible if dither >= v)
}

struct CellRange { var model: Int; var start: Int; var count: Int }

final class World {
    let preset: ForestPreset
    let size: Float = 1536
    let hRes = 1024
    let canopyRes = 512
    var heights: [Float] = []
    private(set) var heightTexture: MTLTexture!
    private(set) var canopyTexture: MTLTexture!
    var models: [TreeModel] = []
    private(set) var vertexBuffer: MTLBuffer!
    private(set) var indexBuffer: MTLBuffer!
    var instances: [TreeInstance] = []
    let cellSize: Float = 64
    var cellsPerSide: Int = 24
    var cellRanges: [[CellRange]] = []
    var cellMin: [float3] = []
    var cellMax: [float3] = []
    var spawn = float3(0, 0, 0)
    var spawnYaw: Float = 0
    var fogBaseHeight: Float = 0
    var minHeight: Float = 0, maxHeight: Float = 0

    // Lake
    private(set) var hasLake = false
    private(set) var waterLevel: Float = -1000
    private(set) var lakeCenter = float2(0, 0)
    private(set) var lakeRadius: Float = 0
    private var lakeSeed: UInt32 = 0
    private(set) var lakeTexture: MTLTexture!

    private struct Clearing { var c: float2; var r: Float }
    private var clearings: [Clearing] = []

    init(device: MTLDevice, preset: ForestPreset, progress: (String) -> Void) {
        self.preset = preset
        cellsPerSide = Int(size / cellSize)
        let N = hRes
        let S = size
        let seed = UInt32(truncatingIfNeeded: preset.seed)
        progress("Sculpting terrain…")

        // --- Terrain heightfield
        var hs = [Float](repeating: 0, count: N * N)
        let hills = preset.hills
        let baseHeight: (Float, Float) -> Float = { x, z in
            var h = fbm(x * 0.0021, z * 0.0021, octaves: 5, seed: seed) * hills
            h += ridged(x * 0.0048, z * 0.0048, octaves: 4, seed: seed &+ 1) * hills * 0.28
            h += fbm(x * 0.022, z * 0.022, octaves: 3, seed: seed &+ 2) * 1.6
            h += fbm(x * 0.08, z * 0.08, octaves: 2, seed: seed &+ 3) * 0.35
            // gently flatten the spawn area so the camera starts in a walkable spot
            let d = sqrt(x * x + z * z)
            return h * mix(0.35, 1, smoothstep(10, 120, d))
        }
        // Lake: carved basin with an irregular shoreline, placed relative to the sun as seen from spawn
        let sunAz0 = preset.sunAzimuth * .pi / 180
        if preset.lakeRadius > 0 {
            hasLake = true
            lakeSeed = seed &+ 77
            lakeRadius = preset.lakeRadius
            let a = sunAz0 + preset.lakeAzimuthOffset * .pi / 180
            lakeCenter = float2(sin(a), -cos(a)) * (preset.lakeRadius + 16)
            waterLevel = baseHeight(lakeCenter.x, lakeCenter.y) - 0.6
        }
        let hasLake = self.hasLake, wl = waterLevel, lc = lakeCenter, lr = lakeRadius, ls = lakeSeed, depth = preset.lakeDepth
        let carve: (Float, Float, Float) -> Float = { h, x, z in
            guard hasLake else { return h }
            let dx = x - lc.x, dz = z - lc.y
            let d = sqrt(dx * dx + dz * dz)
            if d > lr * 2.2 { return h }
            let rr = World.edgeRadius(dx, dz, lr, ls)
            if d < rr {
                let t = d / rr
                let bottom = wl + 0.25 - (depth + 0.25) * pow(max(0, 1 - t * t), 0.6) + fbm(x * 0.05, z * 0.05, octaves: 2, seed: ls &+ 3) * 0.6
                return min(h, bottom)
            }
            let wgt = 1 - smoothstep(rr, rr * 1.7, d)
            let target = wl + 0.25 + (d - rr) * 0.12
            return mix(h, target, wgt)
        }
        hs.withUnsafeMutableBufferPointer { buf in
            let base = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: N) { j in
                for i in 0..<N {
                    let x = (Float(i) / Float(N - 1) - 0.5) * S
                    let z = (Float(j) / Float(N - 1) - 0.5) * S
                    base[j * N + i] = carve(baseHeight(x, z), x, z)
                }
            }
        }
        heights = hs
        minHeight = hs.min() ?? 0
        maxHeight = hs.max() ?? 0

        let float32Filter = device.supports32BitFloatFiltering
        let hd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: float32Filter ? .r32Float : .r16Float, width: N, height: N, mipmapped: false)
        hd.usage = [.shaderRead]
        hd.storageMode = .shared
        heightTexture = device.makeTexture(descriptor: hd)!
        if float32Filter {
            hs.withUnsafeBytes { p in
                heightTexture.replace(region: MTLRegionMake2D(0, 0, N, N), mipmapLevel: 0, withBytes: p.baseAddress!, bytesPerRow: N * 4)
            }
        } else {
            let h16 = hs.map { Float16($0) }
            h16.withUnsafeBytes { p in
                heightTexture.replace(region: MTLRegionMake2D(0, 0, N, N), mipmapLevel: 0, withBytes: p.baseAddress!, bytesPerRow: N * 2)
            }
        }

        // --- Clearings
        var rng = RNG(preset.seed &* 31)
        let sunAz = preset.sunAzimuth * .pi / 180
        let sunH = float2(sin(sunAz), -cos(sunAz))
        clearings.append(Clearing(c: float2(0, 0), r: preset.clearingRadius))
        if hasLake {
            let toLake = simd_normalize(lakeCenter)
            let rr = World.edgeRadius(-toLake.x * lakeRadius, -toLake.y * lakeRadius, lakeRadius, lakeSeed)
            clearings.append(Clearing(c: lakeCenter - toLake * (rr + 9), r: 12))
        }
        if !hasLake {
            clearings.append(Clearing(c: sunH * 75, r: 26))
            clearings.append(Clearing(c: sunH * 60, r: 12))
        }
        for _ in 0..<14 {
            clearings.append(Clearing(c: float2(rng.range(-650, 650), rng.range(-650, 650)), r: rng.range(14, 45)))
        }

        // --- Models
        progress("Growing trees…")
        let acc = MeshAccumulator()
        var speciesModels: [Species: [Int]] = [:]
        func addSpecies(_ s: Species, _ n: Int) {
            let ms = buildModels(species: s, variants: n, seedBase: preset.seed, acc: acc, logBark: preset.logBark)
            var ids: [Int] = []
            for m in ms { ids.append(models.count); models.append(m) }
            speciesModels[s] = ids
        }
        for (s, _) in preset.species { addSpecies(s, s == .redwood ? 4 : 5) }
        if preset.fernDensity > 0 { addSpecies(.fern, 4) }
        if preset.shrubDensity > 0 { addSpecies(.shrub, 3) }
        if preset.rockDensity > 0 { addSpecies(.rock, 5) }
        if preset.logDensity > 0 { addSpecies(.log, 3) }

        if ProcessInfo.processInfo.environment["FOREST_STATS"] != nil {
            for (i, m) in models.enumerated() {
                let t = m.lods.map { "\(($0.barkCount) / 3)+\(($0.leafCount) / 3)" }.joined(separator: " | ")
                print("model \(i) \(m.species) h=\(m.height) r=\(m.radius) lod0=\(m.lod0) lod1=\(m.lod1) tris(bark+leaf): \(t)")
            }
        }
        vertexBuffer = acc.vertices.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        indexBuffer = acc.indices.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }

        // --- Placement
        progress("Planting the forest…")
        var placed: [(TreeInstance, Int)] = []
        let half = S / 2 - 6
        let totalW = preset.species.reduce(0) { $0 + $1.1 }
        let sp = preset.treeSpacing
        let n = Int(S / sp)
        for j in 0..<n {
            for i in 0..<n {
                let x = (Float(i) + 0.5 + rng.range(-0.45, 0.45)) * sp - S / 2
                let z = (Float(j) + 0.5 + rng.range(-0.45, 0.45)) * sp - S / 2
                if abs(x) > half || abs(z) > half { continue }
                if lakeDistance(x, z) < rng.range(2, 6) { continue }
                let clear = clearingMask(x, z)
                let cluster = fbm(x * 0.011, z * 0.011, octaves: 3, seed: seed &+ 5) * 0.5 + 0.5
                var p = preset.treeCoverage * (0.4 + 1.0 * cluster) * (1 - clear)
                let slope = slopeAt(x, z)
                if slope > 0.55 { p *= 0.3 }
                if rng.float() > p { continue }
                // weighted pick, modulated by low-frequency patches so species form natural stands
                var pick = preset.species[0].0
                var bestScore: Float = -1
                for (k, (s, w)) in preset.species.enumerated() {
                    let patch = fbm(x * 0.009 + Float(k) * 37, z * 0.009, octaves: 2, seed: seed &+ 40 &+ UInt32(k)) * 0.5 + 0.5
                    let score = w * (0.25 + 1.5 * patch * patch) * rng.range(0.55, 1.45)
                    if score > bestScore { bestScore = score; pick = s }
                }
                guard let ids = speciesModels[pick], !ids.isEmpty else { continue }
                let m = ids[rng.int(0, ids.count - 1)]
                let scale = rng.range(0.72, 1.18) * mix(0.85, 1.1, cluster)
                let y = heightAt(x, z) - 0.25
                placed.append((TreeInstance(posScale: float4(x, y, z, scale),
                                            params: float4(rng.range(0, 2 * .pi), rng.float(), rng.range(0, 6.28), 1)), m))
            }
        }
        let treeCount = placed.count

        func scatter(_ species: Species, spacing: Float, prob: (Float, Float) -> Float, scale: ClosedRange<Float>, sink: Float) {
            guard let ids = speciesModels[species], !ids.isEmpty else { return }
            let n = Int(S / spacing)
            for j in 0..<n {
                for i in 0..<n {
                    let x = (Float(i) + 0.5 + rng.range(-0.5, 0.5)) * spacing - S / 2
                    let z = (Float(j) + 0.5 + rng.range(-0.5, 0.5)) * spacing - S / 2
                    if abs(x) > half || abs(z) > half { continue }
                    if species != .rock && lakeDistance(x, z) < 1.5 { continue }
                    if species == .rock && lakeDistance(x, z) < -12 { continue }
                    if rng.float() > prob(x, z) { continue }
                    let m = ids[rng.int(0, ids.count - 1)]
                    let sc = rng.range(scale.lowerBound, scale.upperBound)
                    var y = heightAt(x, z) - sink * sc
                    let rot = rng.range(0, 2 * .pi)
                    if species == .log {
                        let L: Float = 5 * sc
                        let dx = cos(rot) * L, dz = -sin(rot) * L
                        y = min(heightAt(x + dx, z + dz), heightAt(x - dx, z - dz), heightAt(x, z)) - 0.12
                    }
                    placed.append((TreeInstance(posScale: float4(x, y, z, sc), params: float4(rot, rng.float(), rng.range(0, 6.28), 1)), m))
                }
            }
        }
        scatter(.fern, spacing: 2.3, prob: { x, z in
            let moist = fbm(x * 0.02, z * 0.02, octaves: 3, seed: seed &+ 60) * 0.5 + 0.5
            return preset.fernDensity * 0.85 * smoothstep(0.25, 0.7, moist) * (1 - 0.6 * self.clearingMask(x, z))
        }, scale: 0.65...1.35, sink: 0.05)
        scatter(.shrub, spacing: 6, prob: { x, z in
            preset.shrubDensity * 0.35 * (fbm(x * 0.015, z * 0.015, octaves: 2, seed: seed &+ 70) * 0.5 + 0.5)
        }, scale: 0.7...1.4, sink: 0.1)
        scatter(.rock, spacing: 9, prob: { x, z in
            let rocky = fbm(x * 0.006, z * 0.006, octaves: 3, seed: seed &+ 80) * 0.5 + 0.5
            return preset.rockDensity * (0.03 + 0.35 * smoothstep(0.55, 0.85, rocky) + 0.4 * smoothstep(0.3, 0.6, self.slopeAt(x, z)))
        }, scale: 0.35...2.4, sink: 0.12)
        scatter(.log, spacing: 17, prob: { x, z in preset.logDensity * 0.28 * (1 - self.clearingMask(x, z)) }, scale: 0.7...1.3, sink: 0)

        // --- Canopy / ground-cover map
        progress("Mapping canopy…")
        let C = canopyRes
        var dens = [Float](repeating: 0, count: C * C)
        var top = [Float](repeating: 0, count: C * C)
        var conif = [Float](repeating: 0, count: C * C)
        var broad = [Float](repeating: 0, count: C * C)
        let px = S / Float(C)
        for (inst, m) in placed[0..<treeCount] {
            let model = models[m]
            let sc = inst.posScale.w
            let R = max(model.crownRadius * sc * 1.05, 2)
            let H = model.height * sc
            let cx = (inst.posScale.x + S / 2) / px, cz = (inst.posScale.z + S / 2) / px
            let rp = Int(ceil(R / px)) + 1
            for dz in -rp...rp {
                for dx in -rp...rp {
                    let ix = Int(cx) + dx, iz = Int(cz) + dz
                    if ix < 0 || iz < 0 || ix >= C || iz >= C { continue }
                    let wx = (Float(ix) + 0.5 - cx) * px, wz = (Float(iz) + 0.5 - cz) * px
                    let d = sqrt(wx * wx + wz * wz) / R
                    if d >= 1 { continue }
                    let w = 1 - d * d
                    let k = iz * C + ix
                    dens[k] += w * 0.6
                    top[k] = max(top[k], H * (0.75 + 0.25 * w))
                    if model.species.isConifer { conif[k] += w } else { broad[k] += w }
                }
            }
        }
        dens = boxBlurWrap(dens, w: C, h: C, radius: 2)
        top = boxBlurWrap(top, w: C, h: C, radius: 1)
        let defaultNeedle: Float = preset.species.filter { $0.0.isConifer }.reduce(0) { $0 + $1.1 } / max(totalW, 0.001)
        var tex = [Float16](repeating: 0, count: C * C * 4)
        for iz in 0..<C {
            for ix in 0..<C {
                let k = iz * C + ix
                let x = (Float(ix) + 0.5) * px - S / 2, z = (Float(iz) + 0.5) * px - S / 2
                let d = min(dens[k], 1)
                let patch = fbm(x * 0.03, z * 0.03, octaves: 3, seed: seed &+ 90) * 0.5 + 0.5
                var grass = preset.grassAmount * (1 - smoothstep(0.1, 0.85, d)) * smoothstep(0.15, 0.6, patch + 0.25)
                grass = max(grass, clearingMask(x, z) * preset.grassAmount * 0.9 + clearingMask(x, z) * 0.25)
                grass = min(grass * (1 - smoothstep(0.35, 0.6, slopeAt(x, z))), 1)
                if hasLake {
                    let gh = heightAt(x, z), ld = lakeDistance(x, z)
                    if ld < 6 {
                        let reedPatch = fbm(x * 0.06, z * 0.06, octaves: 2, seed: lakeSeed &+ 9) * 0.5 + 0.5
                        let band = smoothstep(waterLevel - 0.9, waterLevel - 0.4, gh) * (1 - smoothstep(waterLevel + 0.4, waterLevel + 0.9, gh))
                        grass = max(grass * smoothstep(waterLevel + 0.2, waterLevel + 0.7, gh), band * smoothstep(0.35, 0.65, reedPatch))
                    }
                }
                let tot = conif[k] + broad[k]
                let needle = tot > 0.05 ? conif[k] / tot : defaultNeedle
                tex[k * 4 + 0] = Float16(d)
                tex[k * 4 + 1] = Float16(top[k] / 100)
                tex[k * 4 + 2] = Float16(grass)
                tex[k * 4 + 3] = Float16(needle)
            }
        }
        let cd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: C, height: C, mipmapped: false)
        cd.usage = [.shaderRead]
        cd.storageMode = .shared
        canopyTexture = device.makeTexture(descriptor: cd)!
        tex.withUnsafeBytes { p in
            canopyTexture.replace(region: MTLRegionMake2D(0, 0, C, C), mipmapLevel: 0, withBytes: p.baseAddress!, bytesPerRow: C * 8)
        }

        // --- Lake mask (water is only rendered where this is > 0.5)
        var lm = [UInt8](repeating: 0, count: C * C)
        if hasLake {
            for iz in 0..<C {
                for ix in 0..<C {
                    let x = (Float(ix) + 0.5) * px - S / 2, z = (Float(iz) + 0.5) * px - S / 2
                    lm[iz * C + ix] = lakeDistance(x, z) < 1.5 ? 255 : 0
                }
            }
        }
        let ld = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: C, height: C, mipmapped: false)
        ld.usage = [.shaderRead]
        ld.storageMode = .shared
        lakeTexture = device.makeTexture(descriptor: ld)!
        lm.withUnsafeBytes { p in
            lakeTexture.replace(region: MTLRegionMake2D(0, 0, C, C), mipmapLevel: 0, withBytes: p.baseAddress!, bytesPerRow: C)
        }

        // --- Spatial cells
        let nc = cellsPerSide * cellsPerSide
        var buckets = [[(TreeInstance, Int)]](repeating: [], count: nc)
        for (inst, m) in placed {
            let cx = min(max(Int((inst.posScale.x + S / 2) / cellSize), 0), cellsPerSide - 1)
            let cz = min(max(Int((inst.posScale.z + S / 2) / cellSize), 0), cellsPerSide - 1)
            buckets[cz * cellsPerSide + cx].append((inst, m))
        }
        cellRanges = Array(repeating: [], count: nc)
        cellMin = Array(repeating: float3(repeating: .greatestFiniteMagnitude), count: nc)
        cellMax = Array(repeating: float3(repeating: -.greatestFiniteMagnitude), count: nc)
        for c in 0..<nc {
            let sorted = buckets[c].sorted { $0.1 < $1.1 }
            var k = 0
            while k < sorted.count {
                let m = sorted[k].1
                let start = instances.count
                while k < sorted.count && sorted[k].1 == m {
                    let inst = sorted[k].0
                    instances.append(inst)
                    let model = models[m]
                    let r = model.radius * inst.posScale.w
                    let center = float3(inst.posScale.x, inst.posScale.y + model.height * inst.posScale.w * 0.5, inst.posScale.z)
                    cellMin[c] = simd_min(cellMin[c], center - float3(repeating: r))
                    cellMax[c] = simd_max(cellMax[c], center + float3(repeating: r))
                    k += 1
                }
                cellRanges[c].append(CellRange(model: m, start: start, count: instances.count - start))
            }
        }

        if hasLake {
            // Spawn on the near shore looking across the lake
            let toLake = simd_normalize(lakeCenter)
            let rr = World.edgeRadius(-toLake.x * lakeRadius, -toLake.y * lakeRadius, lakeRadius, lakeSeed)
            let sp2 = lakeCenter - toLake * (rr + 5)
            let spawnY = max(heightAt(sp2.x, sp2.y), waterLevel)
            spawn = float3(sp2.x, spawnY + 1.7, sp2.y)
            spawnYaw = atan2(toLake.x, -toLake.y)
            fogBaseHeight = waterLevel - 2
        } else {
            // Spawn: inside the sunward clearing, looking towards the sun through the far tree line (best for god rays)
            let sp2 = sunH * 60
            let spawnY = heightAt(sp2.x, sp2.y)
            spawn = float3(sp2.x, spawnY + 1.7, sp2.y)
            spawnYaw = sunAz + 0.25
            fogBaseHeight = heightAt(0, 0) - 6
        }
    }

    static func edgeRadius(_ dx: Float, _ dz: Float, _ r: Float, _ seed: UInt32) -> Float {
        let th = atan2(dz, dx)
        let n = gradNoise(cos(th) * 1.8 + 3, sin(th) * 1.8, seed: seed) * 0.22 + gradNoise(cos(th) * 5, sin(th) * 5 + 9, seed: seed &+ 1) * 0.08
        return r * (1 + n)
    }

    /// Signed distance to the shoreline (negative inside the lake). Large positive when there is no lake.
    func lakeDistance(_ x: Float, _ z: Float) -> Float {
        guard hasLake else { return 1e9 }
        let dx = x - lakeCenter.x, dz = z - lakeCenter.y
        return sqrt(dx * dx + dz * dz) - World.edgeRadius(dx, dz, lakeRadius, lakeSeed)
    }

    /// Height the walking camera stands on (allows wading up to ~1 m deep).
    func walkHeight(_ x: Float, _ z: Float) -> Float {
        let g = heightAt(x, z)
        return hasLake && lakeDistance(x, z) < 2 ? max(g, waterLevel - 0.9) : g
    }

    private func clearingMask(_ x: Float, _ z: Float) -> Float {
        var m: Float = 0
        for c in clearings {
            let d = simd_length(float2(x, z) - c.c)
            m = max(m, 1 - smoothstep(c.r * 0.6, c.r, d))
        }
        return m
    }

    func heightAt(_ x: Float, _ z: Float) -> Float {
        let N = hRes
        let fx = clampf((x / size + 0.5) * Float(N - 1), 0, Float(N - 1) - 0.001)
        let fz = clampf((z / size + 0.5) * Float(N - 1), 0, Float(N - 1) - 0.001)
        let ix = Int(fx), iz = Int(fz)
        let tx = fx - Float(ix), tz = fz - Float(iz)
        let h00 = heights[iz * N + ix], h10 = heights[iz * N + ix + 1]
        let h01 = heights[(iz + 1) * N + ix], h11 = heights[(iz + 1) * N + ix + 1]
        return mix(mix(h00, h10, tx), mix(h01, h11, tx), tz)
    }

    func slopeAt(_ x: Float, _ z: Float) -> Float {
        let e: Float = 2
        let dx = heightAt(x + e, z) - heightAt(x - e, z)
        let dz = heightAt(x, z + e) - heightAt(x, z - e)
        let n = simd_normalize(float3(-dx, 2 * e, -dz))
        return 1 - n.y
    }
}
