#if os(macOS)
import AppKit
#endif
import CoreGraphics
import ImageIO
import Metal
import UniformTypeIdentifiers

#if os(macOS)
/// Headless mode for testing & screenshots:
///   ForestRender --snapshot out.png [--scene N] [--quality low|medium|high|ultra|insane]
///                [--width W] [--height H] [--frames N] [--sun-elev deg] [--sun-az deg]
///                [--yaw-offset deg] [--pitch deg] [--pos x,y,z] [--debug N] [--fog d]
func runSnapshot(_ args: [String]) -> Int32 {
    func val(_ k: String) -> String? {
        if let i = args.firstIndex(of: k), i + 1 < args.count { return args[i + 1] }
        return nil
    }
    let out = val("--snapshot")!
    let scene = Int(val("--scene") ?? "0") ?? 0
    let w = Int(val("--width") ?? "1600") ?? 1600
    let h = Int(val("--height") ?? "900") ?? 900
    let frames = Int(val("--frames") ?? "40") ?? 40
    let state = AppState()
    state.applyPreset(scene)
    if let q = val("--quality"), let ql = QualityLevel(rawValue: q.capitalized) { state.quality = ql }
    if let s = val("--sun-elev"), let v = Float(s) { state.sunElevation = v }
    if let s = val("--sun-az"), let v = Float(s) { state.sunAzimuth = v }
    if let s = val("--fog"), let v = Float(s) { state.fogDensity = v }
    if let s = val("--rain"), let v = Float(s) { state.rain = v }
    if let s = val("--snow"), let v = Float(s) { state.snow = v }
    if let s = val("--stars"), let v = Float(s) { state.stars = v }
    if let s = val("--moon-phase"), let v = Float(s) { state.moonPhase = v }
    if let s = val("--moonlight"), let v = Float(s) { state.moonlight = v }
    if let s = val("--moon-elev"), let v = Float(s) { state.moonElevation = v }
    if let s = val("--fov"), let v = Float(s) { state.fov = v }
    if let s = val("--flash"), let v = Float(s) { state.thunder = v > 0 }
    if let s = val("--debug"), let v = Int(s) { state.debugView = v }
    if let s = val("--scale"), let v = Float(s) { state.setGraphics { $0.renderScale = v } }
    if let s = val("--draw"), let v = Float(s) { state.setGraphics { $0.drawDistance = v } }
    if let s = val("--lod"), let v = Float(s) { state.setGraphics { $0.lodBias = v } }
    if let s = val("--shadowres"), let v = Int(s) { state.setGraphics { $0.shadowResolution = v } }
    if let s = val("--pcf"), let v = Int(s) { state.setGraphics { $0.pcfSamples = v } }
    guard let device = MTLCreateSystemDefaultDevice() else { print("no Metal device"); return 1 }
    let renderer: Renderer
    do {
        renderer = try Renderer(device: device, state: state)
    } catch {
        print("[ForestRender] renderer init failed: \(error)")
        return 2
    }
    let t0 = Date()
    renderer.loadContentSync(presetIndex: scene)
    print(String(format: "[ForestRender] content ready in %.2fs", Date().timeIntervalSince(t0)))
    if let s = val("--yaw-offset"), let v = Float(s) { renderer.camera.yaw += v * .pi / 180 }
    if let s = val("--pitch"), let v = Float(s) { renderer.camera.pitch = v * .pi / 180 }
    if let s = val("--pos") {
        let c = s.split(separator: ",").compactMap { Float($0) }
        if c.count == 3 {
            state.walkMode = false
            renderer.camera.position = float3(c[0], c[1], c[2])
        }
    }
    let tex = renderer.renderOffscreen(width: w, height: h, frames: frames)
    var bytes = [UInt8](repeating: 0, count: w * h * 4)
    tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    guard writePNG(bytes: &bytes, width: w, height: h, to: URL(fileURLWithPath: out)) else {
        print("failed to write image")
        return 3
    }
    print("[ForestRender] wrote \(out)")
    return 0
}

#endif

/// BGRA8 (sRGB) pixels -> opaque CGImage.
func makeCGImage(bytes: inout [UInt8], width w: Int, height h: Int) -> CGImage? {
    for i in stride(from: 3, to: bytes.count, by: 4) { bytes[i] = 255 }
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let info = CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue)
    guard let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: cs, bitmapInfo: info.rawValue) else { return nil }
    return ctx.makeImage()
}

/// Writes BGRA8 (sRGB) pixels as an opaque PNG.
func writePNG(bytes: inout [UInt8], width w: Int, height h: Int, to url: URL) -> Bool {
    guard let img = makeCGImage(bytes: &bytes, width: w, height: h),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return false }
    CGImageDestinationAddImage(dest, img, nil)
    return CGImageDestinationFinalize(dest)
}

/// Debug: dump a generated texture layer. --dump-tex kind index out.png  (kind: leaf|bark|ground)
func dumpTexture(_ args: [String]) -> Int32 {
    guard let i = args.firstIndex(of: "--dump-tex"), i + 3 < args.count, let idx = Int(args[i + 2]) else { return 1 }
    let kind = args[i + 1], out = args[i + 3]
    let N = 1024
    let c: Canvas
    switch kind {
    case "leaf":
        switch idx {
        case 0: c = genSpray(N, seed: 200, redwood: false)
        case 1: c = genPineTufts(N, seed: 201)
        case 2: c = genBroadleaf(N, seed: 202, kind: .oak)
        case 3: c = genBroadleaf(N, seed: 203, kind: .birch)
        case 4: c = genFern(N, seed: 204)
        case 5: c = genBroadleaf(N, seed: 205, kind: .beech)
        default: c = genSpray(N, seed: 206, redwood: true)
        }
    case "bark": c = genBark(idx, N, seed: UInt64(100 + idx))
    default:
        switch idx {
        case 0: c = genGroundGrass(N, seed: 11)
        case 1: c = genLeafLitter(N, seed: 12)
        case 2: c = genNeedleLitter(N, seed: 13)
        case 3: c = genMoss(N, seed: 14)
        case 4: c = genRock(N, seed: 15, lichen: 0.4)
        default: c = genDirt(N, seed: 16)
        }
    }
    var bytes = [UInt8](repeating: 0, count: N * N * 4)
    for p in 0..<(N * N) {
        let a = kind == "leaf" ? c.alpha[p] : 1
        let bg = float3(0.85, 0.85, 0.85) * (1 - a)
        let col = c.col[p] * a + bg
        bytes[p * 4 + 0] = u8(linearToSRGB(col.x)); bytes[p * 4 + 1] = u8(linearToSRGB(col.y))
        bytes[p * 4 + 2] = u8(linearToSRGB(col.z)); bytes[p * 4 + 3] = 255
    }
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let ctx = CGContext(data: &bytes, width: N, height: N, bitsPerComponent: 8, bytesPerRow: N * 4, space: cs,
                        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: out) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
    CGImageDestinationFinalize(dest)
    return 0
}

setvbuf(stdout, nil, _IOLBF, 0)

#if os(macOS)

if CommandLine.arguments.contains("--dump-tex") {
    exit(dumpTexture(CommandLine.arguments))
} else if CommandLine.arguments.contains("--snapshot") {
    exit(runSnapshot(CommandLine.arguments))
} else {
    ForestApp.main()
}
#else
ForestApp.main()
#endif
