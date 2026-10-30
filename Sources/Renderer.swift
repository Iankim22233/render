import Foundation
#if os(iOS)
import UIKit
#endif
import Metal
import MetalKit
import QuartzCore
import simd

/// Must match `FrameUniforms` in Shaders/Common.metal.
struct FrameUniforms {
    var view = float4x4.identity
    var proj = float4x4.identity
    var viewProj = float4x4.identity
    var invViewProj = float4x4.identity
    var invProj = float4x4.identity
    var invView = float4x4.identity
    var viewProjNoJitter = float4x4.identity
    var invViewProjNoJitter = float4x4.identity
    var prevViewProjNoJitter = float4x4.identity
    var cascadeViewProj = (float4x4.identity, float4x4.identity, float4x4.identity, float4x4.identity)
    var cascadeInfo = (float4(), float4(), float4(), float4())
    var cameraPos = float4()
    var sunDir = float4()
    var sunColor = float4()
    var screenSize = float4()
    var jitter = float4()
    var fog = float4()
    var fogColor = float4()
    var wind = float4()
    var terrain = float4()
    var shadowParams = float4()
    var qualityParams = float4()
    var grassParams = float4()
    var sceneParams = float4()
    var postParams = float4()
    var postParams2 = float4()
    var colorGrass = float4()
    var misc = float4()
    var water = float4()        // x level, y enabled, z planar reflections, w unused
    var waterColor = float4()   // rgb absorption per meter, w unused
    var weather = float4()      // x rain, y wetness, z overcast, w lightning flash
    var flags = float4()        // x reflection pass
    var snow = float4()         // x snowfall, y snow cover, z stars, w night
    var moon = float4()         // xyz moon direction, w moon is key light
    var moonParams = float4()   // x elongation, y disk brightness, z moonlight irradiance
    var skySun = float4()       // xyz real sun direction
}

struct DrawItem { var model: Int; var lod: Int; var offset: Int; var count: Int }
struct ChunkGroup { var res: Int; var offset: Int; var count: Int }

final class Culler {
    var buckets: [[TreeInstance]] = []
    func reset(models: Int) {
        if buckets.count != models * 4 { buckets = Array(repeating: [], count: models * 4) }
        for i in buckets.indices { buckets[i].removeAll(keepingCapacity: true) }
    }
    var total: Int { buckets.reduce(0) { $0 + $1.count } }
}

final class FrameSlot {
    var instanceBuffer: MTLBuffer
    var chunkBuffer: MTLBuffer
    init(device: MTLDevice) {
        instanceBuffer = device.makeBuffer(length: 1 << 22, options: .storageModeShared)!
        chunkBuffer = device.makeBuffer(length: 64 * 1024, options: .storageModeShared)!
    }
}

final class RenderTargets {
    let width: Int, height: Int
    let gAlbedo, gNormal, gMaterial, depth: MTLTexture
    let ao, aoBlur: MTLTexture
    let vol, volDepth: MTLTexture
    let hdr: MTLTexture
    let history: [MTLTexture]
    let bloom: [MTLTexture]
    let rain: MTLTexture
    let rAlbedo, rNormal, rMaterial, rDepth, rHDR: MTLTexture
    let aoQuality: Int, volQuality: Int, reflQuality: Int

    init(device: MTLDevice, width w: Int, height h: Int, aoQuality: Int, volQuality: Int, reflQuality: Int) {
        width = w; height = h
        self.aoQuality = aoQuality; self.volQuality = volQuality; self.reflQuality = reflQuality
        func tex(_ f: MTLPixelFormat, _ tw: Int, _ th: Int) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: max(tw, 1), height: max(th, 1), mipmapped: false)
            d.usage = [.renderTarget, .shaderRead]
            d.storageMode = .private
            return device.makeTexture(descriptor: d)!
        }
        gAlbedo = tex(.rgba8Unorm_srgb, w, h)
        gNormal = tex(.rgba16Float, w, h)
        gMaterial = tex(.rgba8Unorm, w, h)
        depth = tex(.depth32Float, w, h)
        let aoDiv = aoQuality <= 2 ? 2 : 1
        ao = tex(.r16Float, w / aoDiv, h / aoDiv)
        aoBlur = tex(.r16Float, w / aoDiv, h / aoDiv)
        let volDiv = volQuality <= 3 ? 2 : 1
        vol = tex(.rgba16Float, w / volDiv, h / volDiv)
        volDepth = tex(.r32Float, w / volDiv, h / volDiv)
        hdr = tex(.rgba16Float, w, h)
        history = [tex(.rgba16Float, w, h), tex(.rgba16Float, w, h)]
        var b: [MTLTexture] = []
        var bw = w / 2, bh = h / 2
        for _ in 0..<6 {
            b.append(tex(.rgba16Float, bw, bh))
            bw = max(bw / 2, 1); bh = max(bh / 2, 1)
        }
        bloom = b
        rain = tex(.rgba16Float, w, h)
        let rDiv = reflQuality >= 2 ? 1 : 2
        let rw = reflQuality == 0 ? 1 : w / rDiv, rh = reflQuality == 0 ? 1 : h / rDiv
        rAlbedo = tex(.rgba8Unorm_srgb, rw, rh)
        rNormal = tex(.rgba16Float, rw, rh)
        rMaterial = tex(.rgba8Unorm, rw, rh)
        rDepth = tex(.depth32Float, rw, rh)
        rHDR = tex(.rgba16Float, rw, rh)
    }
}

enum RendererError: Error { case shaderSourceMissing, functionMissing(String) }

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let state: AppState
    let camera = Camera()
    let input = InputState()
    private(set) var world: World?
    private(set) var textures: TextureLibrary?
    private var library: MTLLibrary!

    private var psoTerrain, psoBark, psoLeaf, psoLeafDepth, psoGrass, psoImpostor, psoImpostorShadow: MTLRenderPipelineState!
    private var psoTerrainR, psoBarkR, psoLeafDepthR, psoImpostorR, psoRain, psoSnow: MTLRenderPipelineState!
    private var psoTerrainShadow, psoBarkShadow, psoLeafShadow: MTLRenderPipelineState!
    private var psoGTAO, psoAOBlur, psoVolumetric, psoLighting, psoTAA, psoBloomDown, psoBloomUp, psoFinal: MTLRenderPipelineState!
    private var cpsSkyLUT, cpsSkySH, cpsExposure: MTLComputePipelineState!
    private var dsGBuffer, dsEqual, dsShadow, dsNone, dsTestOnly: MTLDepthStencilState!
    private var barkWinding: MTLWinding = .counterClockwise
    private var materialSampler: MTLSamplerState!
    private var samplerAniso = -1

    private var targets: RenderTargets?
    private var shadowMap: MTLTexture?
    private var skyLUT: MTLTexture!
    private var shBuffer: MTLBuffer!
    private var exposureBuffer: MTLBuffer!
    private var slots: [FrameSlot] = []
    private let inflight = DispatchSemaphore(value: 3)
    private var slotIndex = 0
    private var cullers = (0..<6).map { _ in Culler() }

    private var frameIndex = 0
    private var time: Float = 0
    private var lastTime: CFTimeInterval = CACurrentMediaTime()
    private var prevVPNoJitter = float4x4.identity
    private var historyValid = false
    private var historyIndex = 0
    private var generating = false
    private var worldPresetIndex = -1
    private var exposureInstant = true
    // Shadow cascade cache: far cascades are re-rendered less often (staggered) and keep the matrices they were rendered with.
    private var cachedCascadeVP = [float4x4](repeating: .identity, count: 4)
    private var cachedCascadeInfo = [float4](repeating: .zero, count: 4)
    private var cascadeValid = [false, false, false, false]
    private var lastSunDir = float3(0, 0, 0)
    private var lastShadowKey = ""
    private var lastCascadeCamPos = [float3](repeating: .zero, count: 4)
    // Impostors
    private var impA, impN, impM: MTLTexture?
    private var impDepth: MTLTexture?
    private var impostorAutumn: Float = -1
    private weak var impostorWorld: World?
    private lazy var impInstance: MTLBuffer = {
        var inst = TreeInstance(posScale: float4(0, 0, 0, 1), params: float4(0, 0.5, 0, 1))
        return device.makeBuffer(bytes: &inst, length: 32, options: .storageModeShared)!
    }()
    var pixelScale: Float = 1
    let gamepad = GamepadInput()
    /// Procedural texture resolution: smaller on low-memory mobile devices.
    static var textureSize: Int {
        #if os(iOS)
        return ProcessInfo.processInfo.physicalMemory < 5_500_000_000 ? 512 : 1024
        #else
        return 1024
        #endif
    }
    // Weather
    private var wetness: Float = 0
    private var snowCover: Float = 0
    private var nightExposure = float4(0, 1, 0, 0)
    private var weatherInstant = true
    private var lightningT: Float = -1
    private var nextLightning: Float = 3
    private var lightningRNG = RNG(4242)
    private var pendingShot: MTLTexture?
    var outputFormat: MTLPixelFormat = .bgra8Unorm_srgb

    // stats
    private var statAccumTime: Double = 0
    private var statFrames = 0
    private var gpuTimeAccum: Double = 0
    private var gpuSamples = 0
    private var cpuTimeAccum: Double = 0
    private var lastTriangles = 0
    private var lastInstances = 0
    private let statLock = NSLock()
    private var lastGpuEnd: CFTimeInterval = 0

    init(device: MTLDevice, state: AppState) throws {
        self.device = device
        self.queue = device.makeCommandQueue()!
        self.state = state
        super.init()
        try buildPipelines()
        for _ in 0..<3 { slots.append(FrameSlot(device: device)) }
        let ld = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 256, height: 128, mipmapped: false)
        ld.usage = [.shaderRead, .shaderWrite]
        ld.storageMode = .private
        skyLUT = device.makeTexture(descriptor: ld)
        shBuffer = device.makeBuffer(length: 16 * 8, options: .storageModePrivate)
        exposureBuffer = device.makeBuffer(length: 16, options: .storageModeShared)
        exposureBuffer.contents().storeBytes(of: Float(0), as: Float.self)
    }

    // MARK: - Setup

    static func shaderSource() throws -> String {
        let names = ["Common", "Geometry", "Lighting", "Post", "Weather"]
        var dirs: [URL] = []
        if let r = Bundle.main.resourceURL { dirs.append(r.appendingPathComponent("Shaders")) }
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        dirs.append(exe.appendingPathComponent("../Resources/Shaders").standardized)
        dirs.append(URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("Shaders"))
        for dir in dirs {
            let urls = names.map { dir.appendingPathComponent("\($0).metal") }
            if urls.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) {
                return try urls.map { try String(contentsOf: $0, encoding: .utf8) }.joined(separator: "\n\n")
            }
        }
        throw RendererError.shaderSourceMissing
    }

    private func buildPipelines() throws {
        let src = try Renderer.shaderSource()
        let opts = MTLCompileOptions()
        if #available(macOS 15.0, iOS 18.0, *) { opts.mathMode = .fast }
        opts.preserveInvariance = true
        library = try device.makeLibrary(source: src, options: opts)

        func fn(_ n: String) throws -> MTLFunction {
            // every function is specialized (reflection clipping off) because some reference function constants
            let cv = MTLFunctionConstantValues()
            var off = false
            cv.setConstantValue(&off, type: .bool, index: 0)
            guard let f = try? library.makeFunction(name: n, constantValues: cv) else { throw RendererError.functionMissing(n) }
            return f
        }
        func reflFn(_ n: String) throws -> MTLFunction {
            let cv = MTLFunctionConstantValues()
            var on = true
            cv.setConstantValue(&on, type: .bool, index: 0)
            return try library.makeFunction(name: n, constantValues: cv)
        }
        func pso(_ v: String, _ f: String?, colors: [MTLPixelFormat], depth: MTLPixelFormat = .invalid, additive: Bool = false,
                 noColorWrite: Bool = false, reflect: Bool = false, premultiplied: Bool = false) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = try fn(v)
            d.fragmentFunction = try f.map { reflect ? try reflFn($0) : try fn($0) }
            for (i, c) in colors.enumerated() {
                d.colorAttachments[i].pixelFormat = c
                if noColorWrite { d.colorAttachments[i].writeMask = [] }
                if premultiplied {
                    d.colorAttachments[i].isBlendingEnabled = true
                    d.colorAttachments[i].sourceRGBBlendFactor = .one
                    d.colorAttachments[i].destinationRGBBlendFactor = .oneMinusSourceAlpha
                    d.colorAttachments[i].sourceAlphaBlendFactor = .one
                    d.colorAttachments[i].destinationAlphaBlendFactor = .oneMinusSourceAlpha
                }
                if additive {
                    d.colorAttachments[i].isBlendingEnabled = true
                    d.colorAttachments[i].rgbBlendOperation = .add
                    d.colorAttachments[i].alphaBlendOperation = .add
                    d.colorAttachments[i].sourceRGBBlendFactor = .one
                    d.colorAttachments[i].destinationRGBBlendFactor = .one
                    d.colorAttachments[i].sourceAlphaBlendFactor = .one
                    d.colorAttachments[i].destinationAlphaBlendFactor = .one
                }
            }
            d.depthAttachmentPixelFormat = depth
            return try device.makeRenderPipelineState(descriptor: d)
        }
        let gb: [MTLPixelFormat] = [.rgba8Unorm_srgb, .rgba16Float, .rgba8Unorm]
        psoTerrain = try pso("terrainVS", "terrainFS", colors: gb, depth: .depth32Float)
        psoBark = try pso("barkVS", "barkFS", colors: gb, depth: .depth32Float)
        psoLeaf = try pso("leafVS", "leafFS", colors: gb, depth: .depth32Float)
        psoLeafDepth = try pso("leafVS", "leafDepthFS", colors: gb, depth: .depth32Float, noColorWrite: true)
        psoGrass = try pso("grassVS", "grassFS", colors: gb, depth: .depth32Float)
        psoImpostor = try pso("impostorVS", "impostorFS", colors: gb, depth: .depth32Float)
        psoTerrainR = try pso("terrainVS", "terrainFS", colors: gb, depth: .depth32Float, reflect: true)
        psoBarkR = try pso("barkVS", "barkFS", colors: gb, depth: .depth32Float, reflect: true)
        psoLeafDepthR = try pso("leafVS", "leafDepthFS", colors: gb, depth: .depth32Float, noColorWrite: true, reflect: true)
        psoImpostorR = try pso("impostorVS", "impostorFS", colors: gb, depth: .depth32Float, reflect: true)
        psoRain = try pso("rainVS", "rainFS", colors: [.rgba16Float], depth: .depth32Float, premultiplied: true)
        psoSnow = try pso("snowVS", "snowFS", colors: [.rgba16Float], depth: .depth32Float, premultiplied: true)
        psoImpostorShadow = try pso("impostorVS", "impostorShadowFS", colors: [], depth: .depth32Float)
        psoTerrainShadow = try pso("terrainVS", nil, colors: [], depth: .depth32Float)
        psoBarkShadow = try pso("barkVS", nil, colors: [], depth: .depth32Float)
        psoLeafShadow = try pso("leafVS", "leafShadowFS", colors: [], depth: .depth32Float)
        psoGTAO = try pso("fullscreenVS", "gtaoFS", colors: [.r16Float])
        psoAOBlur = try pso("fullscreenVS", "aoBlurFS", colors: [.r16Float])
        psoVolumetric = try pso("fullscreenVS", "volumetricFS", colors: [.rgba16Float, .r32Float])
        psoLighting = try pso("fullscreenVS", "lightingFS", colors: [.rgba16Float])
        psoTAA = try pso("fullscreenVS", "taaFS", colors: [.rgba16Float])
        psoBloomDown = try pso("fullscreenVS", "bloomDownFS", colors: [.rgba16Float])
        psoBloomUp = try pso("fullscreenVS", "bloomUpFS", colors: [.rgba16Float], additive: true)
        psoFinal = try pso("fullscreenVS", "finalFS", colors: [outputFormat])
        cpsSkyLUT = try device.makeComputePipelineState(function: try fn("skyLUTKernel"))
        cpsSkySH = try device.makeComputePipelineState(function: try fn("skySHKernel"))
        cpsExposure = try device.makeComputePipelineState(function: try fn("exposureKernel"))

        let dg = MTLDepthStencilDescriptor()
        dg.depthCompareFunction = .greater
        dg.isDepthWriteEnabled = true
        dsGBuffer = device.makeDepthStencilState(descriptor: dg)
        let de = MTLDepthStencilDescriptor()
        de.depthCompareFunction = .equal
        de.isDepthWriteEnabled = false
        dsEqual = device.makeDepthStencilState(descriptor: de)
        let dt = MTLDepthStencilDescriptor()
        dt.depthCompareFunction = .greater
        dt.isDepthWriteEnabled = false
        dsTestOnly = device.makeDepthStencilState(descriptor: dt)
        let ds = MTLDepthStencilDescriptor()
        ds.depthCompareFunction = .lessEqual
        ds.isDepthWriteEnabled = true
        dsShadow = device.makeDepthStencilState(descriptor: ds)
        let dn = MTLDepthStencilDescriptor()
        dn.depthCompareFunction = .always
        dn.isDepthWriteEnabled = false
        dsNone = device.makeDepthStencilState(descriptor: dn)
    }

    private func ensureSampler(_ aniso: Int) {
        if aniso == samplerAniso { return }
        let d = MTLSamplerDescriptor()
        d.minFilter = .linear; d.magFilter = .linear; d.mipFilter = .linear
        d.sAddressMode = .repeat; d.tAddressMode = .repeat
        d.maxAnisotropy = max(1, min(aniso, 16))
        materialSampler = device.makeSamplerState(descriptor: d)
        samplerAniso = aniso
    }

    // MARK: - Content loading

    func loadContentAsync() {
        let st = state
        DispatchQueue.global(qos: .userInitiated).async {
            let t0 = CACurrentMediaTime()
            let tex = TextureLibrary(device: self.device, queue: self.queue, size: Renderer.textureSize)
            let t1 = CACurrentMediaTime()
            print(String(format: "[ForestRender] textures generated in %.2fs", t1 - t0))
            DispatchQueue.main.async {
                self.textures = tex
                self.startWorldGeneration(index: st.presetIndex)
            }
        }
    }

    func loadContentSync(presetIndex: Int) {
        textures = TextureLibrary(device: device, queue: queue)
        let w = World(device: device, preset: ForestPreset.all[presetIndex]) { print("[ForestRender] \($0)") }
        installWorld(w, index: presetIndex)
    }

    private func startWorldGeneration(index: Int) {
        guard !generating else { return }
        generating = true
        worldPresetIndex = index
        state.loading = true
        state.loadingText = "Growing \(ForestPreset.all[index].name)…"
        let preset = ForestPreset.all[index]
        DispatchQueue.global(qos: .userInitiated).async {
            let t0 = CACurrentMediaTime()
            let w = World(device: self.device, preset: preset) { msg in
                DispatchQueue.main.async { self.state.loadingText = msg }
            }
            print(String(format: "[ForestRender] world '%@' generated in %.2fs (%d instances, %d models)", preset.name, CACurrentMediaTime() - t0, w.instances.count, w.models.count))
            DispatchQueue.main.async {
                self.installWorld(w, index: index)
                self.generating = false
                self.state.loading = false
            }
        }
    }

    private func installWorld(_ w: World, index: Int) {
        world = w
        worldPresetIndex = index
        camera.position = w.spawn
        camera.yaw = w.spawnYaw
        camera.pitch = 0.05
        camera.velocity = .zero
        historyValid = false
        exposureInstant = true
        weatherInstant = true
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        if world != nil && !generating && state.presetIndex != worldPresetIndex {
            startWorldGeneration(index: state.presetIndex)
        }
        #if os(macOS)
        pixelScale = Float(view.window?.backingScaleFactor ?? 1)
        #else
        pixelScale = Float(view.contentScaleFactor)
        #endif
        gamepad.poll(input: input, state: state)
        guard let drawable = view.currentDrawable else { return }
        inflight.wait()
        let cpuStart = CACurrentMediaTime()
        let now = CACurrentMediaTime()
        let dt = Float(min(max(now - lastTime, 0.0001), 0.1))
        lastTime = now
        let cb = queue.makeCommandBuffer()!
        cb.addCompletedHandler { [weak self] cb in
            guard let self else { return }
            self.statLock.lock()
            // command buffers can overlap on the GPU; count only the non-overlapping busy span
            let gpu = cb.gpuEndTime - max(cb.gpuStartTime, self.lastGpuEnd)
            self.lastGpuEnd = max(self.lastGpuEnd, cb.gpuEndTime)
            if gpu > 0 && gpu < 2 { self.gpuTimeAccum += gpu; self.gpuSamples += 1 }
            self.statLock.unlock()
            self.inflight.signal()
        }
        if state.screenshotRequested {
            state.screenshotRequested = false
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: outputFormat, width: drawable.texture.width,
                                                             height: drawable.texture.height, mipmapped: false)
            d.usage = [.renderTarget]
            d.storageMode = .shared
            if let shot = device.makeTexture(descriptor: d) {
                pendingShot = shot
                cb.addCompletedHandler { [weak self] _ in
                    DispatchQueue.global(qos: .userInitiated).async { self?.saveScreenshot(shot) }
                }
            }
        }
        encodeFrame(cb, output: drawable.texture, dt: dt)
        cb.present(drawable)
        cb.commit()
        updateStats(dt: Double(dt), cpu: CACurrentMediaTime() - cpuStart)
    }

    private func updateStats(dt: Double, cpu: Double) {
        statAccumTime += dt
        statFrames += 1
        cpuTimeAccum += cpu
        if statAccumTime >= 0.4 {
            statLock.lock()
            let gpu = gpuSamples > 0 ? gpuTimeAccum / Double(gpuSamples) : 0
            gpuTimeAccum = 0; gpuSamples = 0
            statLock.unlock()
            state.fps = Double(statFrames) / statAccumTime
            state.gpuMs = gpu * 1000
            state.cpuMs = cpuTimeAccum / Double(statFrames) * 1000
            state.triangles = lastTriangles
            state.instancesDrawn = lastInstances
            state.wetness = wetness
            if let t = targets { state.internalSize = "\(t.width)×\(t.height)" }
            if ProcessInfo.processInfo.environment["FOREST_TRACE"] != nil {
                print(String(format: "[stats] fps %.1f gpu %.1f ms internal %@ preset %d world %d", state.fps, state.gpuMs, state.internalSize, state.presetIndex, worldPresetIndex))
            }
            statAccumTime = 0; statFrames = 0; cpuTimeAccum = 0
        }
    }

    /// Offscreen render used by `--snapshot` mode.
    func renderOffscreen(width: Int, height: Int, frames: Int) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: outputFormat, width: width, height: height, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .shared
        let out = device.makeTexture(descriptor: d)!
        var gpuTotal: Double = 0
        for i in 0..<frames {
            let cb = queue.makeCommandBuffer()!
            encodeFrame(cb, output: out, dt: 1.0 / 60.0)
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { print("[ForestRender] GPU error: \(e)") }
            if i >= frames / 2 { gpuTotal += cb.gpuEndTime - cb.gpuStartTime }
            if ProcessInfo.processInfo.environment["FOREST_TRACE"] != nil && i % 30 == 0 {
                print(String(format: "frame %d gpu %.1f ms", i, (cb.gpuEndTime - cb.gpuStartTime) * 1000))
            }
        }
        print(String(format: "[ForestRender] avg GPU frame time: %.2f ms, triangles ~%d, instances %d", gpuTotal / Double(frames - frames / 2) * 1000, lastTriangles, lastInstances))
        return out
    }

    // MARK: - Frame

    private func ensureTargets(_ w: Int, _ h: Int, _ g: GraphicsSettings) {
        if let t = targets, t.width == w, t.height == h, t.aoQuality == g.aoQuality, t.volQuality == g.volumetricQuality,
           t.reflQuality == g.waterReflections { return }
        targets = RenderTargets(device: device, width: w, height: h, aoQuality: g.aoQuality, volQuality: g.volumetricQuality,
                                reflQuality: g.waterReflections)
        historyValid = false
    }

    private func ensureShadowMap(_ resIn: Int) {
        #if os(iOS)
        let res = min(resIn, 4096)   // 8K shadow arrays don't fit in mobile memory budgets
        #else
        let res = resIn
        #endif
        if let s = shadowMap, s.width == res { return }
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.pixelFormat = .depth32Float
        d.width = res; d.height = res
        d.arrayLength = 4
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        shadowMap = device.makeTexture(descriptor: d)
    }

    private func clearOutput(_ cb: MTLCommandBuffer, _ output: MTLTexture) {
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = output
        rp.colorAttachments[0].loadAction = .clear
        rp.colorAttachments[0].storeAction = .store
        rp.colorAttachments[0].clearColor = MTLClearColor(red: 0.02, green: 0.025, blue: 0.02, alpha: 1)
        cb.makeRenderCommandEncoder(descriptor: rp)?.endEncoding()
    }

    static let cascadeFraction: [Float] = [1, 0.5, 0.5, 0.5]

    private func computeCascades(view: float4x4, fovY: Float, aspect: Float, near: Float, far: Float, sunDir: float3, res: Int)
        -> ([float4x4], [float4]) {
        let lambda: Float = 0.85
        var splits: [Float] = [near]
        for i in 1...4 {
            let f = Float(i) / 4
            let lg = near * pow(far / near, f)
            let ln = near + (far - near) * f
            splits.append(mix(ln, lg, lambda))
        }
        let pos = camera.position, fw = camera.forward, rt = simd_normalize(simd_cross(fw, float3(0, 1, 0)))
        let up = simd_cross(rt, fw)
        let tanH = tan(fovY * 0.5)
        let lz = sunDir
        let upRef: float3 = abs(lz.y) > 0.99 ? float3(0, 0, 1) : float3(0, 1, 0)
        let lx = simd_normalize(simd_cross(upRef, lz))
        let ly = simd_cross(lz, lx)
        var vps: [float4x4] = [], infos: [float4] = []
        let back: Float = 450
        for c in 0..<4 {
            let d0 = c == 0 ? near : splits[c] * 0.97, d1 = splits[c + 1]
            var corners: [float3] = []
            for d in [d0, d1] {
                for sx: Float in [-1, 1] {
                    for sy: Float in [-1, 1] {
                        corners.append(pos + fw * d + rt * (sx * d * tanH * aspect) + up * (sy * d * tanH))
                    }
                }
            }
            var center = corners.reduce(float3(0, 0, 0), +) / 8
            var radius: Float = 0
            for p in corners { radius = max(radius, simd_length(p - center)) }
            radius = ceil(radius * 8) / 8
            let frac = Renderer.cascadeFraction[c]
            let texel = 2 * radius / (Float(res) * frac)
            // snap the center to the shadow-map texel grid in light space (stable shadows while moving)
            var cx = simd_dot(center, lx), cy = simd_dot(center, ly)
            let cz = simd_dot(center, lz)
            cx = floor(cx / texel) * texel
            cy = floor(cy / texel) * texel
            center = lx * cx + ly * cy + lz * cz
            let eye = center + lz * (radius + back)
            let lv = float4x4.lookAt(eye: eye, center: center, up: upRef)
            let range = 2 * radius + back
            let lp = float4x4.orthographic(left: -radius, right: radius, bottom: -radius, top: radius, near: 0, far: range)
            vps.append(lp * lv)
            infos.append(float4(texel, range, radius, frac))
        }
        return (vps, infos)
    }

    private func cull(_ c: Culler, world: World, vp: float4x4, camPos: float3, lodBias: Float, drawDistance: Float, cascade: Int) {
        c.reset(models: world.models.count)
        let planes = frustumPlanes(vp)
        let minLodForCascade = [0, 1, 2, 2]
        let models = world.models
        planes.withUnsafeBufferPointer { pl in
            world.instances.withUnsafeBufferPointer { inst in
                for cell in 0..<world.cellRanges.count {
                    let mn = world.cellMin[cell], mx = world.cellMax[cell]
                    if mn.x > mx.x { continue }
                    let q = simd_clamp(camPos, mn, mx)
                    let cellDist = simd_length(q - camPos)
                    if cellDist > drawDistance { continue }
                    if !aabbVisible(pl, mn, mx) { continue }
                    for r in world.cellRanges[cell] {
                        let m = models[r.model]
                        if cascade >= 0 {
                            if m.species == .fern && cascade > 1 { continue }
                            if m.species == .shrub && cascade > 1 { continue }
                            if cascade == 3 && !m.species.isTree { continue }
                        }
                        let isPlant = !m.species.isTree
                        let maxD = min(isPlant ? m.maxDistance * max(lodBias, 0.6) : m.maxDistance, drawDistance)
                        if cellDist > maxD { continue }
                        let l0 = m.lod0 * lodBias, l1 = m.lod1 * lodBias
                        let hasImp = m.impostorLayer >= 0
                        let l2 = l1 * 1.6
                        let b0 = l0 * 0.85, b1 = l1 * 0.88, b2 = l2 * 0.9
                        let base = r.model * 4
                        for i in r.start..<(r.start + r.count) {
                            var it = inst[i]
                            let s = it.posScale.w
                            let center = float3(it.posScale.x, it.posScale.y + m.height * s * 0.5, it.posScale.z)
                            let rad = m.radius * s
                            let d = simd_length(center - camPos)
                            if d - rad > maxD { continue }
                            if !sphereVisible(pl, center, rad) { continue }
                            if cascade >= 0 {
                                var lod = d < l0 ? 0 : (d < l1 ? 1 : 2)
                                lod = max(lod, minLodForCascade[cascade])
                                if hasImp && (cascade == 3 || d > l2) { lod = 3 }
                                it.params.w = 1
                                c.buckets[base + lod].append(it)
                                continue
                            }
                            var farFade: Float = 1
                            if isPlant && d > maxD * 0.8 { farFade = max((maxD - d) / (maxD * 0.2), 0.001) }
                            if d < b0 {
                                it.params.w = farFade; c.buckets[base].append(it)
                            } else if d < l0 {
                                let f = max(1 - (d - b0) / (l0 - b0), 0.001)
                                it.params.w = f; c.buckets[base].append(it)
                                it.params.w = -f; c.buckets[base + 1].append(it)
                            } else if d < b1 {
                                it.params.w = farFade; c.buckets[base + 1].append(it)
                            } else if d < l1 {
                                let f = max(1 - (d - b1) / (l1 - b1), 0.001)
                                it.params.w = f; c.buckets[base + 1].append(it)
                                it.params.w = -f; c.buckets[base + 2].append(it)
                            } else if !hasImp || d < b2 {
                                it.params.w = farFade; c.buckets[base + 2].append(it)
                            } else if d < l2 {
                                let f = max(1 - (d - b2) / (l2 - b2), 0.001)
                                it.params.w = f; c.buckets[base + 2].append(it)
                                it.params.w = -f; c.buckets[base + 3].append(it)
                            } else {
                                it.params.w = farFade; c.buckets[base + 3].append(it)
                            }
                        }
                    }
                }
            }
        }
    }

    private func flush(_ c: Culler, into ptr: UnsafeMutablePointer<TreeInstance>, cursor: inout Int) -> [DrawItem] {
        var items: [DrawItem] = []
        for (b, arr) in c.buckets.enumerated() where !arr.isEmpty {
            arr.withUnsafeBufferPointer { src in
                (ptr + cursor).update(from: src.baseAddress!, count: arr.count)
            }
            items.append(DrawItem(model: b / 4, lod: b % 4, offset: cursor, count: arr.count))
            cursor += arr.count
        }
        return items
    }

    private func terrainChunks(world: World, vp: float4x4, camPos: float3, shadow: Bool, detail: Float) -> [Int: [float4]] {
        let planes = frustumPlanes(vp)
        let size: Float = 64
        let n = Int(world.size / size)
        var groups: [Int: [float4]] = [:]
        planes.withUnsafeBufferPointer { pl in
            for j in 0..<n {
                for i in 0..<n {
                    let x0 = Float(i) * size - world.size / 2, z0 = Float(j) * size - world.size / 2
                    let mn = float3(x0, world.minHeight - 5, z0), mx = float3(x0 + size, world.maxHeight + 5, z0 + size)
                    if !aabbVisible(pl, mn, mx) { continue }
                    let q = simd_clamp(camPos, mn, mx)
                    let d = simd_length(q - camPos) / max(detail, 0.25)
                    var res = d < 60 ? 64 : (d < 160 ? 32 : (d < 380 ? 16 : 8))
                    if shadow { res = max(res / 2, 4) }
                    groups[res, default: []].append(float4(x0, z0, size, Float(res)))
                }
            }
        }
        return groups
    }

    private func writeChunks(_ groups: [Int: [float4]], slot: FrameSlot, cursor: inout Int) -> [ChunkGroup] {
        var out: [ChunkGroup] = []
        let cap = slot.chunkBuffer.length / 16
        let ptr = slot.chunkBuffer.contents().bindMemory(to: float4.self, capacity: cap)
        for (res, arr) in groups.sorted(by: { $0.key > $1.key }) {
            let n = min(arr.count, cap - cursor)
            if n <= 0 { break }
            arr.withUnsafeBufferPointer { (ptr + cursor).update(from: $0.baseAddress!, count: n) }
            out.append(ChunkGroup(res: res, offset: cursor, count: n))
            cursor += n
        }
        return out
    }

    private let skip = Set((ProcessInfo.processInfo.environment["FOREST_SKIP"] ?? "").split(separator: ",").map(String.init))

    private func encodeFrame(_ cb: MTLCommandBuffer, output: MTLTexture, dt: Float) {
        let st = state
        let g = st.graphics
        guard let world = world, let tex = textures else {
            clearOutput(cb, output)
            return
        }
        let preset = world.preset
        slotIndex = (slotIndex + 1) % 3
        let slot = slots[slotIndex]

        let outW = output.width, outH = output.height
        let scale = g.renderScale / max(pixelScale, 1)
        let iw = max(64, Int((Float(outW) * scale).rounded())), ih = max(64, Int((Float(outH) * scale).rounded()))
        ensureTargets(iw, ih, g)
        ensureShadowMap(g.shadowResolution)
        ensureSampler(g.anisotropy)
        guard let T = targets, let shadowMap = shadowMap else { return }

        if impostorWorld !== world || abs(st.autumn - impostorAutumn) > 0.005 {
            captureImpostors(world: world, tex: tex, cb: cb, autumn: st.autumn)
            impostorWorld = world
            impostorAutumn = st.autumn
        }
        ensureSampler(g.anisotropy)

        if !st.timePaused { time += dt }
        camera.update(dt: dt, input: input, speed: st.moveSpeed, walk: st.walkMode,
                      groundHeight: { world.walkHeight($0, $1) }, bounds: world.size * 0.46)

        // --- weather: surfaces wet up while it rains and slowly dry afterwards; lightning flashes in storms
        let rain = st.rain
        let wetTarget: Float = rain > 0.02 ? min(1, 0.4 + rain) : 0
        if weatherInstant { wetness = wetTarget }
        else { wetness += (wetTarget - wetness) * (1 - exp(-dt * (wetTarget > wetness ? 0.35 : 0.035))) }
        var flash: Float = 0
        if st.thunder && rain > 0.15 && !st.timePaused {
            nextLightning -= dt
            if nextLightning <= 0 { lightningT = 0; nextLightning = lightningRNG.range(4, 14) }
            if lightningT >= 0 {
                let t = lightningT
                flash = exp(-t * 14) + (t > 0.12 ? 0.7 * exp(-(t - 0.12) * 10) : 0) + (t > 0.32 ? 0.45 * exp(-(t - 0.32) * 7) : 0)
                flash *= 2.5 * lightningRNG.range(0.85, 1.15)
                lightningT += dt
                if lightningT > 1.2 { lightningT = -1 }
            }
        }
        // snow cover follows the Snow slider over a couple of seconds
        if weatherInstant { snowCover = st.snow }
        else { snowCover += (st.snow - snowCover) * (1 - exp(-dt * 0.8)) }
        let snowfall = st.snowFalling ? st.snow : 0
        let overcast = max(smoothstep(0.0, 0.35, rain) * 0.92, smoothstep(0.45, 1.0, snowfall) * 0.7)
        let night = smoothstep(-1, -12, st.sunElevation)

        // --- camera matrices
        let aspect = Float(iw) / Float(ih)
        let fovY = st.fov * .pi / 180
        let near: Float = 0.1
        let view = camera.viewMatrix
        let proj = float4x4.perspectiveReversedZInfinite(fovY: fovY, aspect: aspect, near: near)
        var projJ = proj
        var jitter = float2(0, 0)
        if g.taa {
            let k = frameIndex % 16 + 1
            jitter = float2((halton(k, 2) - 0.5) * 2 / Float(iw), (halton(k, 3) - 0.5) * 2 / Float(ih))
            projJ.columns.2.x += jitter.x
            projJ.columns.2.y += jitter.y
        }
        let viewProj = projJ * view
        let vpNJ = proj * view
        if !historyValid { prevVPNoJitter = vpNJ }

        // --- sun & atmosphere
        let el = st.sunElevation * .pi / 180, az = st.sunAzimuth * .pi / 180
        let realSunDir = simd_normalize(float3(cos(el) * sin(az), sin(el), -cos(el) * cos(az)))
        let sunT = Atmosphere.sunTransmittance(elevationDeg: st.sunElevation, turbidity: st.turbidity)
        let sunLight = sunT * smoothstep(-1.0, 1.5, st.sunElevation) * (1 - overcast * 0.85)
        // --- moon: opposite the sun in azimuth at a user-set height; the phase is its elongation from the sun.
        // At night it becomes the key light (shadows, volumetric moonbeams, glitter on water).
        let mEl = st.moonElevation * .pi / 180, mAz = az + .pi
        let moonDir = simd_normalize(float3(cos(mEl) * sin(mAz), sin(mEl), -cos(mEl) * cos(mAz)))
        let elongation = st.moonPhase * 2 * .pi
        let illuminated = (1 - cos(elongation)) / 2
        let moonT = Atmosphere.sunTransmittance(elevationDeg: st.moonElevation, turbidity: st.turbidity)
        let moonIrr = 0.02 * st.moonlight * pow(illuminated, 1.6) * smoothstep(-1, -6, st.sunElevation) * (1 - overcast * 0.85)
        let moonKey = st.sunElevation < -2 && moonIrr > 1e-7 && st.moonElevation > 0
        let moonVisible: Float = st.moonElevation > 0 ? 1 : 0
        nightExposure = float4(night, 0.07 + 0.6 * st.moonlight * pow(illuminated, 1.2) * moonVisible * (1 - overcast * 0.6), 0, 0)
        let sunDir = moonKey ? moonDir : realSunDir
        let sunColor = moonKey ? moonT * moonIrr * float3(0.92, 0.95, 1.0) : sunLight

        let (newVP, newInfo) = computeCascades(view: view, fovY: fovY, aspect: aspect, near: near, far: g.shadowDistance, sunDir: sunDir, res: g.shadowResolution)
        let shadowKey = "\(g.shadowResolution)-\(g.shadowDistance)-\(worldPresetIndex)"
        if simd_length(sunDir - lastSunDir) > 1e-5 || shadowKey != lastShadowKey {
            cascadeValid = [false, false, false, false]
            lastSunDir = sunDir
            lastShadowKey = shadowKey
        }
        let periods = [1, 1, 3, 6], phases = [0, 0, 1, 4]
        var cascadeUpdate = [false, false, false, false]
        for c in 0..<4 {
            let moved = simd_length(camera.position - lastCascadeCamPos[c]) > newInfo[c].z * 0.15
            cascadeUpdate[c] = !cascadeValid[c] || moved || frameIndex % periods[c] == phases[c]
            if cascadeUpdate[c] {
                cachedCascadeVP[c] = newVP[c]
                cachedCascadeInfo[c] = newInfo[c]
                cascadeValid[c] = true
                lastCascadeCamPos[c] = camera.position
            }
        }
        let cvp = cachedCascadeVP, cinfo = cachedCascadeInfo

        // --- uniforms
        var u = FrameUniforms()
        u.view = view
        u.proj = projJ
        u.viewProj = viewProj
        u.invViewProj = viewProj.inverse
        u.invProj = projJ.inverse
        u.invView = view.inverse
        u.viewProjNoJitter = vpNJ
        u.invViewProjNoJitter = vpNJ.inverse
        u.prevViewProjNoJitter = prevVPNoJitter
        u.cascadeViewProj = (cvp[0], cvp[1], cvp[2], cvp[3])
        u.cascadeInfo = (cinfo[0], cinfo[1], cinfo[2], cinfo[3])
        u.cameraPos = float4(camera.position, time)
        u.sunDir = float4(sunDir, 0.00465)
        u.sunColor = float4(sunColor, 1)
        u.screenSize = float4(Float(iw), Float(ih), 1 / Float(iw), 1 / Float(ih))
        u.jitter = float4(jitter.x, jitter.y, Float(frameIndex % 1024), near)
        u.fog = float4(st.fogDensity + snowfall * 0.004, preset.fogFalloff, world.fogBaseHeight, 0.72)
        u.fogColor = float4(preset.fogColor, 0.55)
        let windDir = simd_normalize(float2(0.8, 0.6))
        u.wind = float4(windDir.x, windDir.y, st.windStrength, time)
        u.terrain = float4(world.size, 1 / world.size, Float(world.hRes), 100)
        u.shadowParams = float4(Float(g.pcfSamples), g.pcss ? 1 : 0, g.sunSoftness, g.shadowDistance)
        if skip.contains("filter") { u.shadowParams.x = 1; u.shadowParams.y = 0 }
        let volSteps: [Float] = [0, 20, 40, 72, 128]
        let volDist: [Float] = [0, 120, 180, 260, 340]
        u.qualityParams = float4(g.contactShadows ? 12 : 0, volSteps[g.volumetricQuality], volDist[g.volumetricQuality], g.aoQuality > 0 ? 1 : 0)
        u.grassParams = float4(g.grassDistance, 0.08, preset.grassHeight, preset.grassDryness)
        u.sceneParams = float4(st.autumn, max(st.cloudCoverage, overcast), st.turbidity, preset.mossAmount)
        u.postParams = float4(st.exposureEV, g.bloom ? st.bloomStrength : 0, Float(st.tonemapper), st.vignette)
        u.postParams2 = float4(st.saturation, st.contrast, st.filmGrain, g.taa ? st.sharpen : 0)
        u.colorGrass = float4(preset.grassColor, g.fogNoise ? 1 : 0)
        u.misc = float4(g.lodBias, g.clouds ? 1 : 0, Float(st.debugView), 0.75)
        let reflOn = world.hasLake && g.waterReflections > 0
        u.water = float4(world.waterLevel, world.hasLake ? 1 : 0, reflOn ? 1 : 0, 0)
        u.waterColor = float4(preset.waterAbsorb, 0)
        u.weather = float4(rain, wetness, overcast, flash)
        u.flags = float4(0, 0, 0, 0)
        u.snow = float4(snowfall, snowCover, st.stars, night)
        u.moon = float4(moonDir, moonKey ? 1 : 0)
        u.moonParams = float4(elongation, mix(0.35, 2.5, night), moonKey ? moonIrr : 0, 0)
        u.skySun = float4(realSunDir, 0)
        weatherInstant = false

        // --- visibility
        let camPos = camera.position
        cull(cullers[0], world: world, vp: viewProj, camPos: camPos, lodBias: g.lodBias, drawDistance: g.drawDistance, cascade: -1)
        let castShadows = sunColor.x + sunColor.y + sunColor.z > 0.0001
        if castShadows {
            for c in 0..<4 {
                if cascadeUpdate[c] {
                    cull(cullers[c + 1], world: world, vp: cvp[c], camPos: camPos, lodBias: g.lodBias, drawDistance: min(g.drawDistance, g.shadowDistance + 200), cascade: c)
                } else {
                    cullers[c + 1].reset(models: world.models.count)
                }
            }
        } else {
            for c in 1..<5 { cullers[c].reset(models: world.models.count) }
        }
        // planar reflection view: mirror the camera about the water plane
        let wl = world.waterLevel
        let mirror = float4x4(columns: (float4(1, 0, 0, 0), float4(0, -1, 0, 0), float4(0, 0, 1, 0), float4(0, 2 * wl, 0, 1)))
        let viewR = view * mirror
        let vpR = proj * viewR
        let camR = float3(camPos.x, 2 * wl - camPos.y, camPos.z)
        if reflOn {
            cull(cullers[5], world: world, vp: vpR, camPos: camR, lodBias: g.lodBias * 0.6, drawDistance: g.drawDistance * 0.75, cascade: -1)
        } else {
            cullers[5].reset(models: world.models.count)
        }
        let needed = cullers.reduce(0) { $0 + $1.total }
        if needed * 32 > slot.instanceBuffer.length {
            slot.instanceBuffer = device.makeBuffer(length: max(needed * 32 * 3 / 2, 1 << 20), options: .storageModeShared)!
        }
        let iptr = slot.instanceBuffer.contents().bindMemory(to: TreeInstance.self, capacity: slot.instanceBuffer.length / 32)
        var cursor = 0
        let mainItems = flush(cullers[0], into: iptr, cursor: &cursor)
        var shadowItems: [[DrawItem]] = []
        for c in 0..<4 { shadowItems.append(flush(cullers[c + 1], into: iptr, cursor: &cursor)) }
        let reflItems = flush(cullers[5], into: iptr, cursor: &cursor)
        if ProcessInfo.processInfo.environment["FOREST_STATS"] != nil && frameIndex == 5 {
            for c in 0..<4 {
                var t = 0, n = 0
                for it in shadowItems[c] { if it.lod == 3 { t += 2 * it.count; n += it.count; continue }; let l = world.models[it.model].lods[it.lod]; t += (l.barkCount + l.leafCount) / 3 * it.count; n += it.count }
                print("cascade \(c): \(n) instances, \(t / 1000)k tris, radius \(cinfo[c].z)")
            }
        }

        var tris = 0
        for it in mainItems {
            if it.lod == 3 { tris += 2 * it.count; continue }
            let l = world.models[it.model].lods[it.lod]
            tris += (l.barkCount + l.leafCount) / 3 * it.count
        }
        lastInstances = mainItems.reduce(0) { $0 + $1.count }

        var ccursor = 0
        let mainChunks = writeChunks(terrainChunks(world: world, vp: viewProj, camPos: camPos, shadow: false, detail: g.lodBias), slot: slot, cursor: &ccursor)
        let reflChunks = reflOn ? writeChunks(terrainChunks(world: world, vp: vpR, camPos: camR, shadow: true, detail: g.lodBias), slot: slot, cursor: &ccursor) : []
        var shadowChunks: [[ChunkGroup]] = []
        for c in 0..<4 {
            shadowChunks.append(castShadows ? writeChunks(terrainChunks(world: world, vp: cvp[c], camPos: camPos, shadow: true, detail: g.lodBias), slot: slot, cursor: &ccursor) : [])
        }

        // --- sky LUT + ambient SH
        if let ce = cb.makeComputeCommandEncoder() {
            ce.setComputePipelineState(cpsSkyLUT)
            ce.setTexture(skyLUT, index: 0)
            ce.setBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
            ce.dispatchThreadgroups(MTLSize(width: (skyLUT.width + 7) / 8, height: (skyLUT.height + 7) / 8, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            ce.setComputePipelineState(cpsSkySH)
            ce.setTexture(skyLUT, index: 0)
            ce.setBuffer(shBuffer, offset: 0, index: 0)
            ce.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            ce.endEncoding()
        }

        // --- shadow cascades
        for c in 0..<4 where cascadeUpdate[c] {
            let rp = MTLRenderPassDescriptor()
            rp.depthAttachment.texture = shadowMap
            rp.depthAttachment.slice = c
            rp.depthAttachment.loadAction = .clear
            rp.depthAttachment.storeAction = .store
            rp.depthAttachment.clearDepth = 1.0
            guard let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { continue }
            enc.label = "Shadow cascade \(c)"
            let vpSize = Double(shadowMap.width) * Double(Renderer.cascadeFraction[c])
            enc.setViewport(MTLViewport(originX: 0, originY: 0, width: vpSize, height: vpSize, znear: 0, zfar: 1))
            enc.setDepthStencilState(dsShadow)
            enc.setDepthBias(0.5, slopeScale: 1.5, clamp: 0.002)
            var vp = cvp[c]
            enc.setVertexBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
            enc.setVertexBytes(&vp, length: 64, index: 3)
            if castShadows && !skip.contains("shadow") {
                enc.setRenderPipelineState(psoTerrainShadow)
                enc.setCullMode(.none)
                enc.setVertexTexture(world.heightTexture, index: 0)
                encodeTerrain(enc, groups: shadowChunks[c], slot: slot)
                enc.setFragmentTexture(tex.leafAlbedo, index: 0)
                enc.setFragmentSamplerState(materialSampler, index: 0)
                encodeTrees(enc, items: shadowItems[c], world: world, slot: slot, bark: psoBarkShadow, leaf: psoLeafShadow, barkCull: .none)
                if let impM {
                    enc.setFragmentTexture(impM, index: 2)
                    encodeImpostors(enc, items: shadowItems[c], world: world, slot: slot, facing: float4(sunDir, 1), pso: psoImpostorShadow)
                }
            }
            enc.endEncoding()
        }

        // --- G-buffer
        tris += encodeGBuffer(cb, targets: (T.gAlbedo, T.gNormal, T.gMaterial, T.depth), u: u, vp: viewProj, items: mainItems,
                              chunks: mainChunks, slot: slot, world: world, tex: tex, g: g, grass: true, reflection: false,
                              facing: float4(camPos, 0), label: "GBuffer")

        // --- planar water reflection: mirrored G-buffer + lighting at reduced resolution
        if reflOn {
            var uR = u
            uR.view = viewR
            uR.proj = proj
            uR.viewProj = vpR
            uR.invViewProj = vpR.inverse
            uR.invProj = proj.inverse
            uR.invView = viewR.inverse
            uR.cameraPos = float4(camR, time)
            uR.screenSize = float4(Float(T.rHDR.width), Float(T.rHDR.height), 1 / Float(T.rHDR.width), 1 / Float(T.rHDR.height))
            uR.qualityParams.x = 0
            uR.qualityParams.w = 0
            uR.shadowParams.x = min(uR.shadowParams.x, 8)
            uR.shadowParams.y = 0
            uR.water.y = 0
            uR.flags.x = 1
            uR.misc.z = 0
            barkWinding = .clockwise
            tris += encodeGBuffer(cb, targets: (T.rAlbedo, T.rNormal, T.rMaterial, T.rDepth), u: uR, vp: vpR, items: reflItems,
                                  chunks: reflChunks, slot: slot, world: world, tex: tex, g: g, grass: false, reflection: true,
                                  facing: float4(camR, 0), label: "Reflection GBuffer")
            barkWinding = .counterClockwise
            fullscreen(cb, target: T.rHDR, pso: psoLighting, label: "Reflection lighting") { enc in
                enc.setFragmentBytes(&uR, length: MemoryLayout<FrameUniforms>.stride, index: 2)
                enc.setFragmentBuffer(shBuffer, offset: 0, index: 1)
                enc.setFragmentTexture(T.rAlbedo, index: 0)
                enc.setFragmentTexture(T.rNormal, index: 1)
                enc.setFragmentTexture(T.rMaterial, index: 2)
                enc.setFragmentTexture(T.rDepth, index: 3)
                enc.setFragmentTexture(shadowMap, index: 4)
                enc.setFragmentTexture(T.aoBlur, index: 5)
                enc.setFragmentTexture(skyLUT, index: 6)
                enc.setFragmentTexture(T.vol, index: 7)
                enc.setFragmentTexture(T.volDepth, index: 8)
                enc.setFragmentTexture(world.canopyTexture, index: 9)
                enc.setFragmentTexture(world.heightTexture, index: 10)
                enc.setFragmentTexture(world.lakeTexture, index: 11)
                enc.setFragmentTexture(skyLUT, index: 12)
            }
        }
        lastTriangles = tris

        // --- ambient occlusion
        if g.aoQuality > 0 && !skip.contains("ao") {
            var p: float4
            switch g.aoQuality {
            case 1: p = float4(1.2, 2, 4, 1.4)
            case 2: p = float4(1.4, 3, 6, 1.5)
            default: p = float4(1.6, 4, 10, 1.5)
            }
            fullscreen(cb, target: T.ao, pso: psoGTAO, label: "GTAO") { enc in
                enc.setFragmentBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
                enc.setFragmentBytes(&p, length: 16, index: 3)
                enc.setFragmentTexture(T.depth, index: 0)
                enc.setFragmentTexture(T.gNormal, index: 1)
            }
            fullscreen(cb, target: T.aoBlur, pso: psoAOBlur, label: "AO blur") { enc in
                enc.setFragmentBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
                enc.setFragmentTexture(T.ao, index: 0)
                enc.setFragmentTexture(T.depth, index: 1)
            }
        }

        // --- volumetric light
        do {
            let rp = MTLRenderPassDescriptor()
            rp.colorAttachments[0].texture = T.vol
            rp.colorAttachments[0].loadAction = .dontCare
            rp.colorAttachments[0].storeAction = .store
            rp.colorAttachments[1].texture = T.volDepth
            rp.colorAttachments[1].loadAction = .dontCare
            rp.colorAttachments[1].storeAction = .store
            let enc = cb.makeRenderCommandEncoder(descriptor: rp)!
            enc.label = "Volumetrics"
            enc.setRenderPipelineState(psoVolumetric)
            enc.setFragmentBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
            enc.setFragmentBuffer(shBuffer, offset: 0, index: 1)
            enc.setFragmentTexture(T.depth, index: 0)
            enc.setFragmentTexture(shadowMap, index: 1)
            enc.setFragmentTexture(world.canopyTexture, index: 2)
            enc.setFragmentTexture(world.heightTexture, index: 3)
            enc.setFragmentTexture(world.lakeTexture, index: 4)
            if !skip.contains("vol") { enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3) }
            enc.endEncoding()
        }

        // --- deferred lighting + sky + fog composite
        fullscreen(cb, target: T.hdr, pso: psoLighting, label: "Lighting") { enc in
            enc.setFragmentBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
            enc.setFragmentBuffer(shBuffer, offset: 0, index: 1)
            enc.setFragmentTexture(T.gAlbedo, index: 0)
            enc.setFragmentTexture(T.gNormal, index: 1)
            enc.setFragmentTexture(T.gMaterial, index: 2)
            enc.setFragmentTexture(T.depth, index: 3)
            enc.setFragmentTexture(shadowMap, index: 4)
            enc.setFragmentTexture(T.aoBlur, index: 5)
            enc.setFragmentTexture(skyLUT, index: 6)
            enc.setFragmentTexture(T.vol, index: 7)
            enc.setFragmentTexture(T.volDepth, index: 8)
            enc.setFragmentTexture(world.canopyTexture, index: 9)
            enc.setFragmentTexture(world.heightTexture, index: 10)
            enc.setFragmentTexture(world.lakeTexture, index: 11)
            enc.setFragmentTexture(reflOn ? T.rHDR : skyLUT, index: 12)
        }

        // --- TAA
        var resolved = T.hdr
        if g.taa {
            let dst = T.history[historyIndex]
            let src = T.history[1 - historyIndex]
            var tp = float4(0.9, historyValid ? 0 : 1, 0, 0)
            fullscreen(cb, target: dst, pso: psoTAA, label: "TAA") { enc in
                enc.setFragmentBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
                enc.setFragmentBytes(&tp, length: 16, index: 3)
                enc.setFragmentTexture(T.hdr, index: 0)
                enc.setFragmentTexture(src, index: 1)
                enc.setFragmentTexture(T.depth, index: 2)
                enc.setFragmentTexture(world.lakeTexture, index: 3)
            }
            resolved = dst
            historyIndex = 1 - historyIndex
            historyValid = true
        } else {
            historyValid = false
        }

        // --- rain streaks (own buffer, composited after TAA so drops stay crisp)
        if rain > 0.001 || snowfall > 0.001 {
            let rp = MTLRenderPassDescriptor()
            rp.colorAttachments[0].texture = T.rain
            rp.colorAttachments[0].loadAction = .clear
            rp.colorAttachments[0].storeAction = .store
            rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            rp.depthAttachment.texture = T.depth
            rp.depthAttachment.loadAction = .load
            rp.depthAttachment.storeAction = .dontCare
            if let enc = cb.makeRenderCommandEncoder(descriptor: rp) {
                enc.label = "Rain"
                enc.setRenderPipelineState(psoRain)
                enc.setDepthStencilState(dsTestOnly)
                enc.setCullMode(.none)
                var vp = viewProj
                enc.setVertexBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
                enc.setVertexBytes(&vp, length: 64, index: 3)
                enc.setVertexTexture(world.canopyTexture, index: 0)
                enc.setVertexTexture(world.heightTexture, index: 1)
                enc.setFragmentBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
                enc.setFragmentBuffer(shBuffer, offset: 0, index: 1)
                let density = 0.5 + 0.5 * min(g.grassDensity, 1.5)
                if rain > 0.001 {
                    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: max(Int(rain * 110_000 * density), 1))
                }
                if snowfall > 0.001 {
                    enc.setRenderPipelineState(psoSnow)
                    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: max(Int(snowfall * 140_000 * density), 1))
                }
                enc.endEncoding()
            }
        }

        // --- bloom chain + auto exposure
        var srcTex = resolved
        for (i, dst) in T.bloom.enumerated() {
            var bp = float4(1 / Float(srcTex.width), 1 / Float(srcTex.height), i == 0 ? 1 : 0, 0)
            let s = srcTex
            fullscreen(cb, target: dst, pso: psoBloomDown, label: "Bloom down \(i)") { enc in
                enc.setFragmentTexture(s, index: 0)
                enc.setFragmentBytes(&bp, length: 16, index: 0)
            }
            srcTex = dst
        }
        if let ce = cb.makeComputeCommandEncoder() {
            var ep = float4(dt, exposureInstant ? 1000 : 1.6, st.exposureEV, st.autoExposure ? 1 : 0)
            ce.setComputePipelineState(cpsExposure)
            ce.setTexture(T.bloom[3], index: 0)
            ce.setBuffer(exposureBuffer, offset: 0, index: 0)
            ce.setBytes(&ep, length: 16, index: 1)
            var np = nightExposure
            ce.setBytes(&np, length: 16, index: 2)
            ce.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            ce.endEncoding()
            exposureInstant = false
        }
        for i in stride(from: T.bloom.count - 1, to: 0, by: -1) {
            let s = T.bloom[i]
            var bp = float4(1 / Float(s.width), 1 / Float(s.height), 1.0, 0)
            let rp = MTLRenderPassDescriptor()
            rp.colorAttachments[0].texture = T.bloom[i - 1]
            rp.colorAttachments[0].loadAction = .load
            rp.colorAttachments[0].storeAction = .store
            let enc = cb.makeRenderCommandEncoder(descriptor: rp)!
            enc.setRenderPipelineState(psoBloomUp)
            enc.setFragmentTexture(s, index: 0)
            enc.setFragmentBytes(&bp, length: 16, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }

        // --- final (also into the screenshot texture when one was requested — no UI is ever drawn into it)
        for target in [output, pendingShot].compactMap({ $0 }) {
            let rp = MTLRenderPassDescriptor()
            rp.colorAttachments[0].texture = target
            rp.colorAttachments[0].loadAction = .dontCare
            rp.colorAttachments[0].storeAction = .store
            let enc = cb.makeRenderCommandEncoder(descriptor: rp)!
            enc.label = "Final"
            enc.setRenderPipelineState(psoFinal)
            enc.setFragmentBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
            enc.setFragmentTexture(resolved, index: 0)
            enc.setFragmentTexture(T.bloom[0], index: 1)
            enc.setFragmentTexture(T.rain, index: 2)
            enc.setFragmentBuffer(exposureBuffer, offset: 0, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }
        pendingShot = nil

        prevVPNoJitter = vpNJ
        frameIndex += 1
    }

    /// Fills a G-buffer (main view or mirrored reflection view). Returns the terrain triangle count.
    private func encodeGBuffer(_ cb: MTLCommandBuffer, targets: (MTLTexture, MTLTexture, MTLTexture, MTLTexture), u uIn: FrameUniforms,
                               vp vpIn: float4x4, items: [DrawItem], chunks: [ChunkGroup], slot: FrameSlot, world: World,
                               tex: TextureLibrary, g: GraphicsSettings, grass: Bool, reflection: Bool, facing: float4, label: String) -> Int {
        var u = uIn, vp = vpIn
        var tris = 0
        let rp = MTLRenderPassDescriptor()
        for (i, t) in [targets.0, targets.1, targets.2].enumerated() {
            rp.colorAttachments[i].texture = t
            rp.colorAttachments[i].loadAction = .clear
            rp.colorAttachments[i].storeAction = .store
            rp.colorAttachments[i].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        }
        rp.depthAttachment.texture = targets.3
        rp.depthAttachment.loadAction = .clear
        rp.depthAttachment.storeAction = .store
        rp.depthAttachment.clearDepth = 0
        let enc = cb.makeRenderCommandEncoder(descriptor: rp)!
        enc.label = label
        enc.setDepthStencilState(dsGBuffer)
        enc.setVertexBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
        enc.setVertexBytes(&vp, length: 64, index: 3)
        enc.setFragmentBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
        enc.setFragmentSamplerState(materialSampler, index: 0)
        // terrain
        enc.setRenderPipelineState(reflection ? psoTerrainR : psoTerrain)
        enc.setCullMode(.none)
        enc.setVertexTexture(world.heightTexture, index: 0)
        enc.setFragmentTexture(tex.groundAlbedo, index: 0)
        enc.setFragmentTexture(tex.groundExtra, index: 1)
        enc.setFragmentTexture(world.heightTexture, index: 2)
        enc.setFragmentTexture(world.canopyTexture, index: 3)
        enc.setFragmentTexture(world.lakeTexture, index: 4)
        if !skip.contains("terrain") { encodeTerrain(enc, groups: chunks, slot: slot) }
        for gch in chunks { tris += (gch.res * gch.res * 2 + gch.res * 8) * gch.count }
        // trees, plants, rocks
        enc.setFragmentTexture(tex.barkAlbedo, index: 0)
        enc.setFragmentTexture(tex.barkExtra, index: 1)
        enc.setFragmentTexture(tex.groundAlbedo, index: 2)
        if !skip.contains("trees") {
            if !skip.contains("bark") { encodeTrees(enc, items: items, world: world, slot: slot, bark: reflection ? psoBarkR : psoBark, leaf: nil, barkCull: .back) }
            enc.setFragmentTexture(tex.leafAlbedo, index: 0)
            enc.setFragmentTexture(tex.leafExtra, index: 1)
            // foliage: alpha-tested depth prepass, then shade once per pixel with depth == equal
            if !skip.contains("leafdepth") { encodeTrees(enc, items: items, world: world, slot: slot, bark: nil, leaf: reflection ? psoLeafDepthR : psoLeafDepth, barkCull: .back) }
            enc.setDepthStencilState(dsEqual)
            if !skip.contains("leafshade") { encodeTrees(enc, items: items, world: world, slot: slot, bark: nil, leaf: psoLeaf, barkCull: .back) }
            enc.setDepthStencilState(dsGBuffer)
            if let impA, let impN, let impM {
                enc.setFragmentTexture(impA, index: 0)
                enc.setFragmentTexture(impN, index: 1)
                enc.setFragmentTexture(impM, index: 2)
                encodeImpostors(enc, items: items, world: world, slot: slot, facing: facing, pso: reflection ? psoImpostorR : psoImpostor)
            }
        }
        // grass
        if grass && world.preset.grassAmount > 0 && g.grassDensity > 0.01 && !skip.contains("grass") {
            enc.setRenderPipelineState(psoGrass)
            enc.setCullMode(.none)
            enc.setVertexTexture(world.heightTexture, index: 0)
            enc.setVertexTexture(world.canopyTexture, index: 1)
            let R = g.grassDistance
            let s0 = 0.085 / sqrt(g.grassDensity)
            let rings: [(Float, Float, Float, Float)] = [(0, R * 0.3, s0, 1.0), (R * 0.3, R * 0.62, s0 * 2, 1.65), (R * 0.62, R, s0 * 4, 2.8)]
            for (k, rg) in rings.enumerated() {
                let band = rg.1 * 0.08
                let inner = k == 0 ? 0 : rg.0 - band
                let gridN = Int(ceil(2 * rg.1 / rg.2)) + 1
                var ring = [float4(inner, rg.1, rg.2, rg.3), float4(Float(gridN), 0, band, 0)]
                enc.setVertexBytes(&ring, length: 32, index: 4)
                enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 9, instanceCount: gridN * gridN)
            }
        }
        enc.endEncoding()
        return tris
    }

    // MARK: - Screenshots

    private func saveScreenshot(_ tex: MTLTexture) {
        let w = tex.width, h = tex.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        #if os(iOS)
        guard let cg = makeCGImage(bytes: &bytes, width: w, height: h) else { return }
        let image = UIImage(cgImage: cg)
        DispatchQueue.main.async {
            UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
            self.state.toast = "Screenshot saved to Photos (\(w)×\(h))"
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) { self.state.toast = nil }
        }
        #else
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd 'at' HH.mm.ss"
        let dir = FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask).first!.appendingPathComponent("Forest Render")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("Forest Render \(fmt.string(from: Date())).png")
        let ok = writePNG(bytes: &bytes, width: w, height: h, to: url)
        DispatchQueue.main.async {
            self.state.toast = ok ? "Screenshot saved to Pictures/Forest Render (\(w)×\(h))" : "Could not save screenshot"
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) { self.state.toast = nil }
        }
        #endif
    }

    /// Renders every tree model (LOD1) once from the side into albedo/normal/material atlases used for far billboards.
    private func captureImpostors(world: World, tex: TextureLibrary, cb: MTLCommandBuffer, autumn: Float) {
        let trees = world.models.indices.filter { world.models[$0].species.isTree }
        guard !trees.isEmpty else { return }
        let W = 256, H = 512
        if impA == nil || impA!.arrayLength != trees.count {
            func make(_ f: MTLPixelFormat) -> MTLTexture {
                let d = MTLTextureDescriptor()
                d.textureType = .type2DArray
                d.pixelFormat = f
                d.width = W; d.height = H
                d.arrayLength = trees.count
                d.mipmapLevelCount = Int(log2(Double(H))) + 1
                d.usage = [.renderTarget, .shaderRead]
                d.storageMode = .private
                return device.makeTexture(descriptor: d)!
            }
            impA = make(.rgba8Unorm_srgb)
            impN = make(.rgba16Float)
            impM = make(.rgba8Unorm)
            let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: W, height: H, mipmapped: false)
            dd.usage = [.renderTarget]
            #if targetEnvironment(simulator)
            dd.storageMode = .private
            #else
            dd.storageMode = .memoryless
            #endif
            impDepth = device.makeTexture(descriptor: dd)
        }
        guard let impA, let impN, let impM, let impDepth else { return }
        let idxPtr = world.indexBuffer.contents().bindMemory(to: UInt32.self, capacity: world.indexBuffer.length / 4)
        let vtxPtr = world.vertexBuffer.contents().bindMemory(to: TreeVertex.self, capacity: world.vertexBuffer.length / MemoryLayout<TreeVertex>.stride)
        ensureSampler(16)
        for (layer, mi) in trees.enumerated() {
            let lod = world.models[mi].lods[1]
            var maxR: Float = 0.5, minY: Float = 0, maxY: Float = 1
            for (start, count) in [(lod.barkStart, lod.barkCount), (lod.leafStart, lod.leafCount)] {
                for i in start..<(start + count) {
                    let p = vtxPtr[Int(idxPtr[i])].position
                    maxR = max(maxR, abs(p.x), abs(p.z)); minY = min(minY, p.y); maxY = max(maxY, p.y)
                }
            }
            let hw = max(maxR * 1.03, (maxY - minY) * 0.5 * 1.03 / 2), hh = hw * 2
            let cy = (minY + maxY) * 0.5
            world.models[mi].impostorLayer = layer
            world.models[mi].impostor = float4(hw, cy, hh, Float(layer))
            let R = maxR + 5
            var u = FrameUniforms()
            u.view = float4x4.lookAt(eye: float3(0, cy, 200), center: float3(0, cy, 0), up: float3(0, 1, 0))
            u.proj = float4x4.orthographic(left: -hw, right: hw, bottom: -hh, top: hh, near: 200 + R, far: 200 - R)
            u.viewProj = u.proj * u.view
            u.invViewProj = u.viewProj.inverse
            u.cameraPos = float4(0, cy, 200, 0)
            u.wind = float4(1, 0, 0, 0)
            u.jitter = float4(0, 0, 0, 0.1)
            u.terrain = float4(world.size, 1 / world.size, Float(world.hRes), 100)
            u.sceneParams = float4(autumn, 0, 2, world.preset.mossAmount)
            u.colorGrass = float4(world.preset.grassColor, 0)
            u.misc = float4(1, 0, 0, 0)
            var vp = u.viewProj
            let rp = MTLRenderPassDescriptor()
            for (i, t) in [impA, impN, impM].enumerated() {
                rp.colorAttachments[i].texture = t
                rp.colorAttachments[i].slice = layer
                rp.colorAttachments[i].loadAction = .clear
                rp.colorAttachments[i].storeAction = .store
                rp.colorAttachments[i].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            }
            rp.depthAttachment.texture = impDepth
            rp.depthAttachment.loadAction = .clear
            rp.depthAttachment.storeAction = .dontCare
            rp.depthAttachment.clearDepth = 0
            guard let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { continue }
            enc.label = "Impostor capture \(layer)"
            enc.setDepthStencilState(dsGBuffer)
            enc.setVertexBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
            enc.setVertexBytes(&vp, length: 64, index: 3)
            enc.setFragmentBytes(&u, length: MemoryLayout<FrameUniforms>.stride, index: 2)
            enc.setFragmentSamplerState(materialSampler, index: 0)
            let items = [DrawItem(model: mi, lod: 1, offset: 0, count: 1)]
            enc.setFragmentTexture(tex.barkAlbedo, index: 0)
            enc.setFragmentTexture(tex.barkExtra, index: 1)
            enc.setFragmentTexture(tex.groundAlbedo, index: 2)
            encodeTrees(enc, items: items, world: world, instanceBuffer: impInstance, bark: psoBark, leaf: nil, barkCull: .none)
            enc.setFragmentTexture(tex.leafAlbedo, index: 0)
            enc.setFragmentTexture(tex.leafExtra, index: 1)
            encodeTrees(enc, items: items, world: world, instanceBuffer: impInstance, bark: nil, leaf: psoLeafDepth, barkCull: .none)
            enc.setDepthStencilState(dsEqual)
            encodeTrees(enc, items: items, world: world, instanceBuffer: impInstance, bark: nil, leaf: psoLeaf, barkCull: .none)
            enc.endEncoding()
        }
        if let blit = cb.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: impA)
            blit.generateMipmaps(for: impN)
            blit.generateMipmaps(for: impM)
            blit.endEncoding()
        }
        samplerAniso = -1
    }

    private func fullscreen(_ cb: MTLCommandBuffer, target: MTLTexture, pso: MTLRenderPipelineState, label: String, setup: (MTLRenderCommandEncoder) -> Void) {
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = target
        rp.colorAttachments[0].loadAction = .dontCare
        rp.colorAttachments[0].storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { return }
        enc.label = label
        enc.setRenderPipelineState(pso)
        setup(enc)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }

    private func encodeTerrain(_ enc: MTLRenderCommandEncoder, groups: [ChunkGroup], slot: FrameSlot) {
        enc.setVertexBuffer(slot.chunkBuffer, offset: 0, index: 0)
        for g in groups {
            enc.setVertexBufferOffset(g.offset * 16, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: g.res * g.res * 6 + 4 * g.res * 6, instanceCount: g.count)
        }
    }

    private func encodeTrees(_ enc: MTLRenderCommandEncoder, items: [DrawItem], world: World, slot: FrameSlot,
                             bark: MTLRenderPipelineState?, leaf: MTLRenderPipelineState?, barkCull: MTLCullMode) {
        encodeTrees(enc, items: items, world: world, instanceBuffer: slot.instanceBuffer, bark: bark, leaf: leaf, barkCull: barkCull)
    }

    private func encodeImpostors(_ enc: MTLRenderCommandEncoder, items: [DrawItem], world: World, slot: FrameSlot, facing: float4, pso: MTLRenderPipelineState) {
        let imps = items.filter { $0.lod == 3 }
        guard !imps.isEmpty else { return }
        enc.setRenderPipelineState(pso)
        enc.setCullMode(.none)
        enc.setVertexBuffer(slot.instanceBuffer, offset: 0, index: 1)
        for it in imps {
            var imp = [world.models[it.model].impostor, facing]
            enc.setVertexBytes(&imp, length: 32, index: 4)
            enc.setVertexBufferOffset(it.offset * 32, index: 1)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: it.count)
        }
    }

    private func encodeTrees(_ enc: MTLRenderCommandEncoder, items allItems: [DrawItem], world: World, instanceBuffer: MTLBuffer,
                             bark: MTLRenderPipelineState?, leaf: MTLRenderPipelineState?, barkCull: MTLCullMode) {
        let items = allItems.filter { $0.lod < 3 }
        guard !items.isEmpty else { return }
        enc.setVertexBuffer(world.vertexBuffer, offset: 0, index: 0)
        enc.setVertexBuffer(instanceBuffer, offset: 0, index: 1)
        if let bark {
            enc.setRenderPipelineState(bark)
            enc.setCullMode(barkCull)
            enc.setFrontFacing(barkWinding)
            for it in items {
                let l = world.models[it.model].lods[it.lod]
                if l.barkCount == 0 { continue }
                enc.setVertexBufferOffset(it.offset * 32, index: 1)
                enc.drawIndexedPrimitives(type: .triangle, indexCount: l.barkCount, indexType: .uint32,
                                          indexBuffer: world.indexBuffer, indexBufferOffset: l.barkStart * 4, instanceCount: it.count)
            }
        }
        if let leaf {
            enc.setRenderPipelineState(leaf)
            enc.setCullMode(.none)
            for it in items {
                let l = world.models[it.model].lods[it.lod]
                if l.leafCount == 0 { continue }
                enc.setVertexBufferOffset(it.offset * 32, index: 1)
                enc.drawIndexedPrimitives(type: .triangle, indexCount: l.leafCount, indexType: .uint32,
                                          indexBuffer: world.indexBuffer, indexBufferOffset: l.leafStart * 4, instanceCount: it.count)
            }
        }
    }
}
