#if os(macOS)
import AppKit
#else
import UIKit
#endif
import Metal
import SwiftUI

#if os(macOS)
let panelWidth: CGFloat = 340
#else
let panelWidth: CGFloat = 320
#endif

struct ForestApp: App {
    static var sharedState: AppState?
    @StateObject private var state: AppState
    private let renderer: Renderer
    #if os(macOS)
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    #endif

    init() {
        let s = AppState()
        ForestApp.sharedState = s
        _state = StateObject(wrappedValue: s)
        guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal is not available on this device") }
        do {
            renderer = try Renderer(device: device, state: s)
        } catch {
            fatalError("Failed to build renderer: \(error)")
        }
        renderer.loadContentAsync()
    }

    var body: some Scene {
        #if os(macOS)
        WindowGroup("Forest Render") {
            ContentView(state: state, renderer: renderer)
                .frame(minWidth: 960, minHeight: 600)
        }
        .defaultSize(width: 1600, height: 940)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
        #else
        WindowGroup {
            ContentView(state: state, renderer: renderer)
                .statusBarHidden(true)
                .persistentSystemOverlays(.hidden)
        }
        #endif
    }
}

#if os(macOS)
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        // Debug hook: FOREST_SWITCH=<preset> switches forest after 12 s (tests live regeneration).
        if let s = ProcessInfo.processInfo.environment["FOREST_SWITCH"], let idx = Int(s), let st = ForestApp.sharedState {
            DispatchQueue.main.asyncAfter(deadline: .now() + 12) { st.applyPreset(idx) }
        }
        // Debug hook: FOREST_TESTSHOT=1 presses the screenshot key after 14 s.
        if ProcessInfo.processInfo.environment["FOREST_TESTSHOT"] != nil, let st = ForestApp.sharedState {
            DispatchQueue.main.asyncAfter(deadline: .now() + 14) { st.screenshotRequested = true }
        }
        // Debug hook: FOREST_WINDOWSHOT=/path.png captures this app's own window after a delay.
        if let path = ProcessInfo.processInfo.environment["FOREST_WINDOWSHOT"] {
            let delay = Double(ProcessInfo.processInfo.environment["FOREST_WINDOWSHOT_DELAY"] ?? "14") ?? 14
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                guard let win = NSApp.windows.first(where: { $0.isVisible }) else { print("no window"); return }
                let id = CGWindowID(win.windowNumber)
                if let img = CGWindowListCreateImage(.null, .optionIncludingWindow, id, [.boundsIgnoreFraming, .bestResolution]) {
                    let rep = NSBitmapImageRep(cgImage: img)
                    try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
                    print("[ForestRender] window captured to \(path)")
                } else {
                    print("[ForestRender] window capture failed")
                }
            }
        }
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
#endif

// MARK: - Views

struct ContentView: View {
    @ObservedObject var state: AppState
    let renderer: Renderer

    var body: some View {
        ZStack(alignment: .topLeading) {
            MetalView(renderer: renderer, state: state)
                .ignoresSafeArea()
            #if os(iOS)
            TouchControls(renderer: renderer, state: state)
                .ignoresSafeArea()
            #endif
            if !state.hideUI {
                VStack(alignment: .leading, spacing: 8) {
                    StatsOverlay(state: state)
                    #if os(iOS)
                    if state.showHelp { HelpOverlay() }
                    Spacer()
                    #else
                    Spacer()
                    if state.showHelp { HelpOverlay() }
                    #endif
                }
                .padding(14)
                .allowsHitTesting(false)
            }
            if let toast = state.toast {
                VStack {
                    Spacer()
                    Label(toast, systemImage: "camera.fill")
                        .font(.system(size: 13, weight: .medium))
                        .padding(.horizontal, 14).padding(.vertical, 9)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(.bottom, 40)
                }
                .frame(maxWidth: .infinity)
                .allowsHitTesting(false)
                .transition(.opacity)
            }
            if state.loading {
                VStack(spacing: 12) {
                    ProgressView().controlSize(.large)
                    Text(state.loadingText).font(.headline)
                    Text("Generating terrain, trees, bark & leaf textures procedurally").font(.caption).foregroundStyle(.secondary)
                }
                .padding(28)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            HStack(spacing: 0) {
                Spacer()
                if state.showPanel && !state.hideUI {
                    SettingsPanel(state: state)
                        .frame(width: panelWidth)
                        .transition(.move(edge: .trailing))
                }
            }
            if !state.hideUI {
                VStack {
                    HStack(spacing: 8) {
                        Spacer()
                        #if os(iOS)
                        ToolButton(systemImage: "questionmark") { withAnimation { state.showHelp.toggle() } }
                        ToolButton(systemImage: "eye.slash") { state.hideUI = true }
                        #endif
                        ToolButton(systemImage: "camera") { state.screenshotRequested = true }
                            .help("Save a screenshot without UI (F12 or ⌘S)")
                        ToolButton(systemImage: state.showPanel ? "sidebar.trailing" : "slider.horizontal.3") {
                            withAnimation(.easeInOut(duration: 0.2)) { state.showPanel.toggle() }
                        }
                    }
                    #if os(iOS)
                    .padding(.trailing, state.showPanel ? panelWidth + 12 : 20)
                    #else
                    .padding(.trailing, state.showPanel ? panelWidth + 12 : 12)
                    #endif
                    .padding(.top, 12)
                    Spacer()
                }
            }
        }
        .background(Color.black)
        #if os(iOS)
        .onAppear {
            DispatchQueue.main.asyncAfter(deadline: .now() + 12) { withAnimation { state.showHelp = false } }
            // Debug hook: FOREST_IOS_DEMO=1 opens the settings panel and switches to fly mode after 14 s.
            if ProcessInfo.processInfo.environment["FOREST_IOS_DEMO"] != nil {
                DispatchQueue.main.asyncAfter(deadline: .now() + 14) {
                    state.walkMode = false; state.showPanel = true
                }
            }
        }
        #endif
    }
}

struct ToolButton: View {
    let systemImage: String
    let action: () -> Void

    var body: some View {
        #if os(iOS)
        Image(systemName: systemImage)
            .font(.system(size: 17, weight: .medium))
            .frame(width: 40, height: 40)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 9))
            .pressAction(action)
        #else
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 15, weight: .medium))
                .padding(8)
        }
        .buttonStyle(.plain)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
        #endif
    }
}

struct StatsOverlay: View {
    @ObservedObject var state: AppState
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 10) {
                Text(String(format: "%.0f FPS", state.fps)).font(.system(size: 15, weight: .semibold, design: .monospaced))
                Text(String(format: "GPU %.1f ms", state.gpuMs)).font(.system(size: 12, design: .monospaced))
                Text(String(format: "CPU %.1f ms", state.cpuMs)).font(.system(size: 12, design: .monospaced))
            }
            Text("\(state.internalSize) · \(String(format: "%.1fM", Double(state.triangles) / 1e6)) tris · \(state.instancesDrawn) objects · \(state.quality.rawValue)")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10).padding(.vertical, 7)
        .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
        .foregroundStyle(.white)
    }
}

struct HelpOverlay: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            #if os(iOS)
            Text("Left stick — move   ·   drag anywhere — look   ·   RUN / WALK·FLY buttons   ·   ↑ ↓ — up/down when flying")
            Text("Game controller: sticks move/look · R1/L1 up/down · R2 sprint · Y walk/fly · X screenshot · Menu settings")
            Text("Camera button — screenshot to Photos   ·   sliders button — settings")
            #else
            Text("Drag mouse — look around   ·   W A S D — move   ·   Shift — sprint   ·   Option — slow")
            Text("G — toggle walk / fly   ·   Q/E or C/Space — down/up (fly)   ·   Scroll — move speed")
            Text("Tab — settings panel   ·   P — pause time   ·   H — hide this help   ·   U — hide all UI")
            Text("F12 or ⌘S — save screenshot (no UI) to Pictures/Forest Render   ·   Game controllers supported")
            #endif
        }
        .font(.system(size: 11))
        .padding(.horizontal, 10).padding(.vertical, 7)
        .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
        .foregroundStyle(.white.opacity(0.9))
    }
}

struct SliderRow: View {
    let title: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    var format: String = "%.2f"

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: format, value)).monospacedDigit().foregroundStyle(.secondary)
            }
            .font(.system(size: 11))
            Slider(value: $value, in: range).controlSize(.mini)
        }
    }
}

struct Section<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon).font(.system(size: 12, weight: .semibold)).foregroundStyle(.secondary)
            content
        }
        .padding(12)
        .background(.background.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
    }
}

struct SettingsPanel: View {
    @ObservedObject var state: AppState

    private func gb<T>(_ kp: WritableKeyPath<GraphicsSettings, T>) -> Binding<T> {
        Binding(get: { state.graphics[keyPath: kp] }, set: { v in state.setGraphics { $0[keyPath: kp] = v } })
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("Forest Render").font(.system(size: 18, weight: .bold))
                    .padding(.top, 48)

                Section(title: "Scene", icon: "tree") {
                    Picker("Forest", selection: Binding(get: { state.presetIndex }, set: { state.applyPreset($0) })) {
                        ForEach(ForestPreset.all.indices, id: \.self) { i in Text(ForestPreset.all[i].name).tag(i) }
                    }
                    .labelsHidden()
                    .disabled(state.loading)
                    Text(ForestPreset.all[state.presetIndex].subtitle).font(.caption).foregroundStyle(.secondary)
                    SliderRow(title: state.sunElevation < 0 ? "Sun elevation (below horizon)" : "Sun elevation",
                              value: $state.sunElevation, range: -90...90, format: "%.1f°")
                    SliderRow(title: "Stars", value: $state.stars, range: 0...1)
                        .disabled(state.sunElevation >= 0)
                        .opacity(state.sunElevation >= 0 ? 0.45 : 1)
                    if state.sunElevation >= 0 {
                        Text("Lower the sun below the horizon to see the stars.").font(.caption2).foregroundStyle(.secondary)
                    }
                    SliderRow(title: "Moon phase — \(state.moonPhaseName)", value: $state.moonPhase, range: 0...1, format: "%.2f")
                    SliderRow(title: "Moonlight on the ground", value: $state.moonlight, range: 0...1)
                        .disabled(state.sunElevation >= 0)
                        .opacity(state.sunElevation >= 0 ? 0.45 : 1)
                    SliderRow(title: "Moon height", value: $state.moonElevation, range: 2...85, format: "%.0f°")
                    Text(state.sunElevation >= 0 ? "Moonlight takes over once the sun is below the horizon." :
                         "Moonlight scales with the phase — a new moon gives no light.")
                        .font(.caption2).foregroundStyle(.secondary)
                    SliderRow(title: "Sun direction", value: $state.sunAzimuth, range: 0...360, format: "%.0f°")
                    SliderRow(title: "Fog density", value: $state.fogDensity, range: 0...0.045, format: "%.4f")
                    SliderRow(title: "Atmospheric haze", value: $state.turbidity, range: 1...8, format: "%.1f")
                    SliderRow(title: "Cloud cover", value: $state.cloudCoverage, range: 0...1)
                    SliderRow(title: "Wind", value: $state.windStrength, range: 0...1.5)
                    SliderRow(title: "Autumn", value: $state.autumn, range: 0...1)
                    Toggle("Pause time (P)", isOn: $state.timePaused).font(.system(size: 11))
                }

                Section(title: "Weather", icon: "cloud.rain") {
                    SliderRow(title: "Rain", value: $state.rain, range: 0...1)
                    Toggle("Thunderstorm (lightning)", isOn: $state.thunder).font(.system(size: 11))
                    SliderRow(title: "Snow", value: $state.snow, range: 0...1)
                    Toggle("Falling snow", isOn: $state.snowFalling).font(.system(size: 11))
                        .disabled(state.snow <= 0.001)
                    HStack {
                        Text("Surface wetness").font(.system(size: 11))
                        Spacer()
                        Text("\(Int(state.wetness * 100))%").font(.system(size: 11)).monospacedDigit().foregroundStyle(.secondary)
                    }
                    Text("Ground, bark and leaves wet up while it rains (puddles form in flat spots) and slowly dry afterwards.")
                        .font(.caption2).foregroundStyle(.secondary)
                }

                Section(title: "Graphics", icon: "cpu") {
                    Picker("Quality", selection: $state.quality) {
                        ForEach(QualityLevel.allCases) { q in Text(q.rawValue).tag(q) }
                    }
                    .font(.system(size: 11))
                    SliderRow(title: "Render scale (2× = full Retina res)", value: gb(\.renderScale), range: 0.4...3.0, format: "%.2f×")
                    Picker("Shadow map", selection: gb(\.shadowResolution)) {
                        Text("1024²").tag(1024); Text("2048²").tag(2048); Text("4096²").tag(4096); Text("8192²").tag(8192)
                    }
                    .font(.system(size: 11))
                    SliderRow(title: "Shadow distance", value: gb(\.shadowDistance), range: 60...600, format: "%.0f m")
                    Picker("Soft-shadow samples", selection: gb(\.pcfSamples)) {
                        ForEach([8, 16, 24, 32, 48, 64, 96], id: \.self) { Text("\($0)").tag($0) }
                    }
                    .font(.system(size: 11))
                    Toggle("Contact-hardening soft shadows (PCSS)", isOn: gb(\.pcss)).font(.system(size: 11))
                    SliderRow(title: "Sun size / penumbra", value: gb(\.sunSoftness), range: 0.25...5, format: "%.2f×")
                    Toggle("Screen-space contact shadows", isOn: gb(\.contactShadows)).font(.system(size: 11))
                    Picker("Volumetric light", selection: gb(\.volumetricQuality)) {
                        Text("Off").tag(0); Text("Low").tag(1); Text("Medium").tag(2); Text("High").tag(3); Text("Extreme").tag(4)
                    }
                    .font(.system(size: 11))
                    Toggle("Volumetric fog noise", isOn: gb(\.fogNoise)).font(.system(size: 11))
                    Picker("Ambient occlusion (GTAO)", selection: gb(\.aoQuality)) {
                        Text("Off").tag(0); Text("Low").tag(1); Text("High").tag(2); Text("Ultra").tag(3)
                    }
                    .font(.system(size: 11))
                    SliderRow(title: "Grass density", value: gb(\.grassDensity), range: 0...2.5)
                    SliderRow(title: "Grass distance", value: gb(\.grassDistance), range: 15...160, format: "%.0f m")
                    SliderRow(title: "Tree detail (LOD bias)", value: gb(\.lodBias), range: 0.4...3.5)
                    SliderRow(title: "Draw distance", value: gb(\.drawDistance), range: 250...1700, format: "%.0f m")
                    Picker("Anisotropic filtering", selection: gb(\.anisotropy)) {
                        Text("1×").tag(1); Text("4×").tag(4); Text("8×").tag(8); Text("16×").tag(16)
                    }
                    .font(.system(size: 11))
                    Toggle("Temporal anti-aliasing", isOn: gb(\.taa)).font(.system(size: 11))
                    Toggle("Bloom", isOn: gb(\.bloom)).font(.system(size: 11))
                    Toggle("Clouds & cloud shadows", isOn: gb(\.clouds)).font(.system(size: 11))
                    Picker("Water reflections", selection: gb(\.waterReflections)) {
                        Text("Sky only").tag(0); Text("Planar (half res)").tag(1); Text("Planar (full res)").tag(2)
                    }
                    .font(.system(size: 11))
                }

                Section(title: "Camera & Image", icon: "camera.aperture") {
                    Toggle("Walk mode (G)", isOn: $state.walkMode).font(.system(size: 11))
                    SliderRow(title: "Field of view", value: $state.fov, range: 30...100, format: "%.0f°")
                    SliderRow(title: "Move speed", value: $state.moveSpeed, range: 0.5...60, format: "%.1f m/s")
                    Toggle("Auto exposure (eye adaptation)", isOn: $state.autoExposure).font(.system(size: 11))
                    SliderRow(title: "Exposure compensation", value: $state.exposureEV, range: -4...4, format: "%+.1f EV")
                    Picker("Tone mapping", selection: $state.tonemapper) {
                        Text("AgX (filmic)").tag(0); Text("ACES").tag(1)
                    }
                    .font(.system(size: 11))
                    SliderRow(title: "Bloom strength", value: $state.bloomStrength, range: 0...0.25, format: "%.3f")
                    SliderRow(title: "Saturation", value: $state.saturation, range: 0...1.6)
                    SliderRow(title: "Contrast", value: $state.contrast, range: 0.7...1.4)
                    SliderRow(title: "Vignette", value: $state.vignette, range: 0...1)
                    SliderRow(title: "Film grain", value: $state.filmGrain, range: 0...1)
                    SliderRow(title: "Sharpen", value: $state.sharpen, range: 0...1)
                    Picker("Debug view", selection: $state.debugView) {
                        Text("Final image").tag(0); Text("Albedo").tag(1); Text("Normals").tag(2); Text("Ambient occlusion").tag(3); Text("Sun shadows").tag(4)
                    }
                    .font(.system(size: 11))
                }
                Text("Tip: 'Insane' plus render scale 1.5–2× is very GPU heavy — great for screenshots.")
                    .font(.caption2).foregroundStyle(.secondary)
                    .padding(.bottom, 20)
            }
            .padding(.horizontal, 14)
        }
        .background(.ultraThinMaterial)
    }
}
