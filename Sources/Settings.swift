import Foundation
import Combine
import simd

struct ForestPreset {
    var name: String
    var subtitle: String
    var seed: UInt64
    var species: [(Species, Float)]
    var treeSpacing: Float
    var treeCoverage: Float
    var hills: Float
    var sunElevation: Float
    var sunAzimuth: Float
    var fogDensity: Float
    var fogFalloff: Float
    var fogColor: float3
    var turbidity: Float
    var cloudCoverage: Float
    var autumn: Float
    var grassAmount: Float
    var grassHeight: Float
    var grassColor: float3
    var grassDryness: Float
    var mossAmount: Float
    var fernDensity: Float
    var shrubDensity: Float
    var rockDensity: Float
    var logDensity: Float
    var windStrength: Float
    var exposureEV: Float
    var clearingRadius: Float
    var logBark: Int
    // Lakes & weather (defaults: no lake, dry)
    var lakeRadius: Float = 0
    var lakeDepth: Float = 6
    var lakeAzimuthOffset: Float = 0      // lake direction relative to the sun (deg) as seen from spawn
    var waterAbsorb = float3(0.32, 0.11, 0.09)   // per-meter absorption (r, g, b)
    var rain: Float = 0
    var thunder = false
    var snow: Float = 0
    var stars: Float = 0.75
    var moonPhase: Float = 0.5      // 0 new, 0.25 first quarter, 0.5 full, 0.75 last quarter

    static let all: [ForestPreset] = [
        ForestPreset(name: "Nordic Spruce — Dawn Mist", subtitle: "Dense spruce taiga, low sun, mist and god rays",
                     seed: 1001, species: [(.spruce, 0.78), (.pine, 0.12), (.birch, 0.1)],
                     treeSpacing: 5.8, treeCoverage: 0.66, hills: 34, sunElevation: 9, sunAzimuth: 95,
                     fogDensity: 0.009, fogFalloff: 0.03, fogColor: float3(0.92, 0.95, 1.0), turbidity: 3.2, cloudCoverage: 0.25,
                     autumn: 0, grassAmount: 0.35, grassHeight: 0.45, grassColor: float3(1, 1, 1), grassDryness: 0.15,
                     mossAmount: 0.95, fernDensity: 0.45, shrubDensity: 0.04, rockDensity: 0.7, logDensity: 0.55,
                     windStrength: 0.3, exposureEV: 0, clearingRadius: 12, logBark: 0),
        ForestPreset(name: "Golden Birch Grove", subtitle: "Autumn birches in warm late-afternoon light",
                     seed: 2002, species: [(.birch, 0.8), (.spruce, 0.2)],
                     treeSpacing: 6.2, treeCoverage: 0.62, hills: 20, sunElevation: 13, sunAzimuth: 250,
                     fogDensity: 0.006, fogFalloff: 0.04, fogColor: float3(1.0, 0.97, 0.92), turbidity: 2.6, cloudCoverage: 0.35,
                     autumn: 0.88, grassAmount: 0.8, grassHeight: 0.62, grassColor: float3(1.15, 1.0, 0.65), grassDryness: 0.5,
                     mossAmount: 0.3, fernDensity: 0.25, shrubDensity: 0.12, rockDensity: 0.3, logDensity: 0.3,
                     windStrength: 0.55, exposureEV: 0, clearingRadius: 16, logBark: 5),
        ForestPreset(name: "Ancient Redwoods", subtitle: "Giant redwoods, fern understory, coastal fog",
                     seed: 3003, species: [(.redwood, 0.88), (.spruce, 0.12)],
                     treeSpacing: 11, treeCoverage: 0.62, hills: 46, sunElevation: 36, sunAzimuth: 160,
                     fogDensity: 0.016, fogFalloff: 0.012, fogColor: float3(0.94, 0.96, 1.0), turbidity: 4, cloudCoverage: 0.1,
                     autumn: 0, grassAmount: 0.08, grassHeight: 0.4, grassColor: float3(0.95, 1.05, 0.9), grassDryness: 0.1,
                     mossAmount: 0.85, fernDensity: 1.0, shrubDensity: 0.15, rockDensity: 0.25, logDensity: 0.9,
                     windStrength: 0.12, exposureEV: 0, clearingRadius: 10, logBark: 4),
        ForestPreset(name: "Autumn Oak Woodland", subtitle: "Oaks and beeches turning copper and gold",
                     seed: 4004, species: [(.oak, 0.6), (.beech, 0.25), (.birch, 0.15)],
                     treeSpacing: 9, treeCoverage: 0.62, hills: 24, sunElevation: 21, sunAzimuth: 205,
                     fogDensity: 0.005, fogFalloff: 0.04, fogColor: float3(1.0, 0.97, 0.93), turbidity: 2.8, cloudCoverage: 0.45,
                     autumn: 0.72, grassAmount: 0.5, grassHeight: 0.5, grassColor: float3(1.1, 1.0, 0.7), grassDryness: 0.55,
                     mossAmount: 0.4, fernDensity: 0.3, shrubDensity: 0.3, rockDensity: 0.4, logDensity: 0.45,
                     windStrength: 0.45, exposureEV: 0, clearingRadius: 15, logBark: 2),
        ForestPreset(name: "Summer Pine Heath — Midday", subtitle: "Open Scots pine forest under a high summer sun",
                     seed: 5005, species: [(.pine, 0.85), (.birch, 0.15)],
                     treeSpacing: 8, treeCoverage: 0.52, hills: 18, sunElevation: 58, sunAzimuth: 185,
                     fogDensity: 0.0018, fogFalloff: 0.02, fogColor: float3(0.95, 0.97, 1.0), turbidity: 2.0, cloudCoverage: 0.4,
                     autumn: 0, grassAmount: 0.9, grassHeight: 0.55, grassColor: float3(1.05, 1.0, 0.8), grassDryness: 0.3,
                     mossAmount: 0.5, fernDensity: 0.2, shrubDensity: 0.15, rockDensity: 0.5, logDensity: 0.2,
                     windStrength: 0.35, exposureEV: 0, clearingRadius: 14, logBark: 1),
        ForestPreset(name: "Misty Beech Cathedral", subtitle: "Tall spring beeches, thick morning fog",
                     seed: 6006, species: [(.beech, 0.88), (.oak, 0.12)],
                     treeSpacing: 8.5, treeCoverage: 0.78, hills: 28, sunElevation: 11, sunAzimuth: 80,
                     fogDensity: 0.02, fogFalloff: 0.025, fogColor: float3(0.95, 0.97, 1.0), turbidity: 3.5, cloudCoverage: 0.2,
                     autumn: 0, grassAmount: 0.2, grassHeight: 0.4, grassColor: float3(0.95, 1.08, 0.85), grassDryness: 0.05,
                     mossAmount: 0.55, fernDensity: 0.4, shrubDensity: 0.1, rockDensity: 0.3, logDensity: 0.4,
                     windStrength: 0.2, exposureEV: 0, clearingRadius: 12, logBark: 5),
        ForestPreset(name: "Mirror Lake — Alpine Sunset", subtitle: "Still mountain lake reflecting a golden spruce shore",
                     seed: 7007, species: [(.spruce, 0.62), (.pine, 0.28), (.birch, 0.1)],
                     treeSpacing: 6.5, treeCoverage: 0.6, hills: 70, sunElevation: 6, sunAzimuth: 255,
                     fogDensity: 0.0025, fogFalloff: 0.03, fogColor: float3(1.0, 0.96, 0.92), turbidity: 2.4, cloudCoverage: 0.4,
                     autumn: 0, grassAmount: 0.5, grassHeight: 0.45, grassColor: float3(1.05, 1.0, 0.8), grassDryness: 0.3,
                     mossAmount: 0.7, fernDensity: 0.3, shrubDensity: 0.08, rockDensity: 0.9, logDensity: 0.4,
                     windStrength: 0.08, exposureEV: 0, clearingRadius: 10, logBark: 0,
                     lakeRadius: 115, lakeDepth: 9, lakeAzimuthOffset: 150, waterAbsorb: float3(0.36, 0.08, 0.06)),
        ForestPreset(name: "Rainy Birch Lake", subtitle: "Steady rain on a birch-lined lake, wet ground and puddles",
                     seed: 8008, species: [(.birch, 0.55), (.spruce, 0.3), (.beech, 0.15)],
                     treeSpacing: 6.5, treeCoverage: 0.6, hills: 26, sunElevation: 28, sunAzimuth: 200,
                     fogDensity: 0.007, fogFalloff: 0.02, fogColor: float3(0.92, 0.95, 1.0), turbidity: 4.5, cloudCoverage: 0.9,
                     autumn: 0.25, grassAmount: 0.75, grassHeight: 0.55, grassColor: float3(1.0, 1.05, 0.85), grassDryness: 0.15,
                     mossAmount: 0.8, fernDensity: 0.45, shrubDensity: 0.15, rockDensity: 0.5, logDensity: 0.45,
                     windStrength: 0.45, exposureEV: 0, clearingRadius: 12, logBark: 5,
                     lakeRadius: 85, lakeDepth: 6, lakeAzimuthOffset: 60, waterAbsorb: float3(0.3, 0.13, 0.12), rain: 0.75),
        ForestPreset(name: "Misty Pine Lake — Morning", subtitle: "Fog drifting over a calm pine lake at sunrise",
                     seed: 9009, species: [(.pine, 0.75), (.birch, 0.15), (.spruce, 0.1)],
                     treeSpacing: 7.5, treeCoverage: 0.6, hills: 30, sunElevation: 5, sunAzimuth: 100,
                     fogDensity: 0.014, fogFalloff: 0.06, fogColor: float3(0.96, 0.97, 1.0), turbidity: 3.0, cloudCoverage: 0.15,
                     autumn: 0, grassAmount: 0.6, grassHeight: 0.5, grassColor: float3(1.05, 1.0, 0.8), grassDryness: 0.25,
                     mossAmount: 0.6, fernDensity: 0.25, shrubDensity: 0.12, rockDensity: 0.6, logDensity: 0.3,
                     windStrength: 0.06, exposureEV: 0, clearingRadius: 12, logBark: 1,
                     lakeRadius: 130, lakeDepth: 7, lakeAzimuthOffset: 15, waterAbsorb: float3(0.34, 0.1, 0.09)),
        ForestPreset(name: "Thunderstorm Taiga Lake", subtitle: "Heavy rain, wind and lightning over a dark spruce lake",
                     seed: 10010, species: [(.spruce, 0.85), (.birch, 0.15)],
                     treeSpacing: 6, treeCoverage: 0.68, hills: 40, sunElevation: 20, sunAzimuth: 300,
                     fogDensity: 0.006, fogFalloff: 0.02, fogColor: float3(0.9, 0.93, 1.0), turbidity: 6, cloudCoverage: 1.0,
                     autumn: 0, grassAmount: 0.5, grassHeight: 0.5, grassColor: float3(0.95, 1.0, 0.9), grassDryness: 0.1,
                     mossAmount: 0.95, fernDensity: 0.5, shrubDensity: 0.06, rockDensity: 0.7, logDensity: 0.6,
                     windStrength: 0.9, exposureEV: 0, clearingRadius: 10, logBark: 0,
                     lakeRadius: 100, lakeDepth: 8, lakeAzimuthOffset: 90, waterAbsorb: float3(0.38, 0.16, 0.14), rain: 1.0, thunder: true),
        ForestPreset(name: "Winter Taiga — Fresh Snow", subtitle: "Snow-laden spruces in low winter sun, flakes still falling",
                     seed: 11011, species: [(.spruce, 0.82), (.pine, 0.1), (.birch, 0.08)],
                     treeSpacing: 6.2, treeCoverage: 0.62, hills: 38, sunElevation: 9, sunAzimuth: 160,
                     fogDensity: 0.004, fogFalloff: 0.03, fogColor: float3(0.94, 0.96, 1.0), turbidity: 2.2, cloudCoverage: 0.3,
                     autumn: 0, grassAmount: 0.4, grassHeight: 0.45, grassColor: float3(1.0, 0.95, 0.75), grassDryness: 0.5,
                     mossAmount: 0.6, fernDensity: 0.1, shrubDensity: 0.05, rockDensity: 0.8, logDensity: 0.6,
                     windStrength: 0.2, exposureEV: 0, clearingRadius: 12, logBark: 0, snow: 0.8),
        ForestPreset(name: "Starry Night Lake", subtitle: "Thin crescent moon, the Milky Way mirrored in a still lake",
                     seed: 12012, species: [(.spruce, 0.55), (.pine, 0.35), (.birch, 0.1)],
                     treeSpacing: 6.8, treeCoverage: 0.6, hills: 55, sunElevation: -32, sunAzimuth: 250,
                     fogDensity: 0.0015, fogFalloff: 0.04, fogColor: float3(0.9, 0.93, 1.0), turbidity: 2.0, cloudCoverage: 0.15,
                     autumn: 0, grassAmount: 0.5, grassHeight: 0.45, grassColor: float3(1.0, 1.0, 0.85), grassDryness: 0.3,
                     mossAmount: 0.6, fernDensity: 0.25, shrubDensity: 0.08, rockDensity: 0.8, logDensity: 0.4,
                     windStrength: 0.04, exposureEV: 0, clearingRadius: 10, logBark: 0,
                     lakeRadius: 120, lakeDepth: 8, lakeAzimuthOffset: 180, waterAbsorb: float3(0.34, 0.09, 0.07), stars: 1.0, moonPhase: 0.16),
    ]
}

enum QualityLevel: String, CaseIterable, Identifiable {
    case low = "Low", medium = "Medium", high = "High", ultra = "Ultra", insane = "Insane", custom = "Custom"
    var id: String { rawValue }
}

struct GraphicsSettings: Equatable {
    var renderScale: Float = 1.0   // relative to window points: 2.0 = native Retina pixels
    var shadowResolution: Int = 4096
    var shadowDistance: Float = 260
    var pcfSamples: Int = 24
    var pcss: Bool = true
    var sunSoftness: Float = 1.0
    var contactShadows: Bool = true
    var volumetricQuality: Int = 2   // 0 off .. 4 extreme
    var aoQuality: Int = 2           // 0 off .. 3 ultra
    var grassDensity: Float = 1.0
    var grassDistance: Float = 60
    var lodBias: Float = 1.0
    var drawDistance: Float = 900
    var taa: Bool = true
    var bloom: Bool = true
    var anisotropy: Int = 16
    var fogNoise: Bool = true
    var clouds: Bool = true
    var waterReflections: Int = 1   // 0 sky only, 1 half-res planar, 2 full-res planar

    static func preset(_ q: QualityLevel) -> GraphicsSettings {
        var g = GraphicsSettings()
        switch q {
        case .low:
            g.renderScale = 0.65; g.shadowResolution = 2048; g.shadowDistance = 140; g.pcfSamples = 8; g.pcss = false
            g.contactShadows = false; g.volumetricQuality = 1; g.aoQuality = 1; g.grassDensity = 0.45; g.grassDistance = 30
            g.lodBias = 0.6; g.drawDistance = 500; g.anisotropy = 4; g.fogNoise = false; g.waterReflections = 0
        case .medium:
            g.renderScale = 0.85; g.shadowResolution = 2048; g.shadowDistance = 190; g.pcfSamples = 16; g.pcss = true
            g.contactShadows = true; g.volumetricQuality = 2; g.aoQuality = 1; g.grassDensity = 0.7; g.grassDistance = 45
            g.lodBias = 0.8; g.drawDistance = 700; g.anisotropy = 8
        case .high, .custom:
            break
        case .ultra:
            g.renderScale = 1.3; g.shadowResolution = 4096; g.shadowDistance = 340; g.pcfSamples = 32; g.volumetricQuality = 3
            g.aoQuality = 2; g.grassDensity = 1.35; g.grassDistance = 85; g.lodBias = 1.4; g.drawDistance = 1200; g.waterReflections = 2
        case .insane:
            g.renderScale = 1.6; g.shadowResolution = 8192; g.shadowDistance = 420; g.pcfSamples = 64; g.volumetricQuality = 4
            g.aoQuality = 3; g.grassDensity = 1.9; g.grassDistance = 115; g.lodBias = 2.2; g.drawDistance = 1600; g.waterReflections = 2
        }
        return g
    }
}

final class AppState: ObservableObject {
    init() {
        #if os(iOS)
        // mobile defaults: Medium quality at slightly above point resolution, panel closed
        quality = .medium
        var g = GraphicsSettings.preset(.medium)
        g.renderScale = 1.4
        graphics = g
        showPanel = false
        #endif
    }

    // Scene
    @Published var presetIndex: Int = 0
    @Published var sunElevation: Float = ForestPreset.all[0].sunElevation
    @Published var sunAzimuth: Float = ForestPreset.all[0].sunAzimuth
    @Published var fogDensity: Float = ForestPreset.all[0].fogDensity
    @Published var windStrength: Float = ForestPreset.all[0].windStrength
    @Published var cloudCoverage: Float = ForestPreset.all[0].cloudCoverage
    @Published var autumn: Float = ForestPreset.all[0].autumn
    @Published var turbidity: Float = ForestPreset.all[0].turbidity
    @Published var timePaused = false
    @Published var rain: Float = ForestPreset.all[0].rain
    @Published var thunder = ForestPreset.all[0].thunder
    @Published var snow: Float = ForestPreset.all[0].snow
    @Published var snowFalling = true
    @Published var stars: Float = ForestPreset.all[0].stars
    @Published var moonPhase: Float = ForestPreset.all[0].moonPhase
    @Published var moonlight: Float = 0.4
    @Published var moonElevation: Float = 32

    var moonPhaseName: String {
        let p = moonPhase
        switch p {
        case ..<0.03, 0.97...: return "New moon"
        case ..<0.22: return "Waxing crescent"
        case ..<0.28: return "First quarter"
        case ..<0.47: return "Waxing gibbous"
        case ..<0.53: return "Full moon"
        case ..<0.72: return "Waning gibbous"
        case ..<0.78: return "Last quarter"
        default: return "Waning crescent"
        }
    }
    @Published var wetness: Float = 0          // driven by the renderer (surfaces wet up / dry out over time)

    // Camera / post
    @Published var fov: Float = 62
    @Published var moveSpeed: Float = 6
    @Published var walkMode = true
    @Published var exposureEV: Float = 0
    @Published var autoExposure = true
    @Published var bloomStrength: Float = 0.05
    @Published var tonemapper: Int = 0   // 0 AgX, 1 ACES
    @Published var saturation: Float = 1.05
    @Published var contrast: Float = 1.05
    @Published var vignette: Float = 0.35
    @Published var filmGrain: Float = 0.25
    @Published var sharpen: Float = 0.25
    @Published var debugView: Int = 0

    // Graphics
    @Published var quality: QualityLevel = .high {
        didSet { if quality != .custom { graphics = GraphicsSettings.preset(quality) } }
    }
    @Published var graphics = GraphicsSettings.preset(.high)

    // UI / stats
    @Published var showPanel = ProcessInfo.processInfo.environment["FOREST_NOPANEL"] == nil
    @Published var hideUI = false
    @Published var toast: String?
    var screenshotRequested = false
    @Published var showHelp = true
    @Published var loading = true
    @Published var loadingText = "Generating textures…"
    @Published var fps: Double = 0
    @Published var gpuMs: Double = 0
    @Published var cpuMs: Double = 0
    @Published var internalSize = "-"
    @Published var triangles: Int = 0
    @Published var instancesDrawn: Int = 0

    func applyPreset(_ i: Int) {
        if ProcessInfo.processInfo.environment["FOREST_TRACE"] != nil {
            print("applyPreset(\(i)) from:\n" + Thread.callStackSymbols.prefix(12).joined(separator: "\n"))
        }
        let p = ForestPreset.all[i]
        sunElevation = p.sunElevation
        sunAzimuth = p.sunAzimuth
        fogDensity = p.fogDensity
        windStrength = p.windStrength
        cloudCoverage = p.cloudCoverage
        autumn = p.autumn
        turbidity = p.turbidity
        exposureEV = p.exposureEV
        rain = p.rain
        thunder = p.thunder
        snow = p.snow
        snowFalling = true
        stars = p.stars
        moonPhase = p.moonPhase
        presetIndex = i
    }

    func setGraphics(_ mutate: (inout GraphicsSettings) -> Void) {
        var g = graphics
        mutate(&g)
        graphics = g
        if quality != .custom { quality = .custom }
    }
}
