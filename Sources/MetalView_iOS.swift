#if os(iOS)
import MetalKit
import SwiftUI
import UIKit

/// Metal view that turns drags on empty screen areas into camera look (controls drawn above it take their own touches).
final class TouchMTKView: MTKView {
    weak var renderer: Renderer?

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        for t in touches {
            let p = t.location(in: self), q = t.previousLocation(in: self)
            renderer?.input.mouseDelta += float2(Float(p.x - q.x), Float(p.y - q.y)) * 1.7
        }
    }
}

struct MetalView: UIViewRepresentable {
    let renderer: Renderer
    let state: AppState

    func makeUIView(context: Context) -> MTKView {
        let v = TouchMTKView(frame: .zero, device: renderer.device)
        v.renderer = renderer
        v.colorPixelFormat = renderer.outputFormat
        v.depthStencilPixelFormat = .invalid
        v.framebufferOnly = true
        v.preferredFramesPerSecond = 60
        v.enableSetNeedsDisplay = false
        v.isPaused = false
        v.isMultipleTouchEnabled = true
        v.delegate = renderer
        return v
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}

// MARK: - Touch controls

/// Fires on touch-down (snappy, game-style), with a short debounce so one press = one action.
struct PressAction: ViewModifier {
    let action: () -> Void
    @State private var down = false
    @State private var lastFire = Date.distantPast

    func body(content: Content) -> some View {
        content
            .contentShape(Rectangle())
            .opacity(down ? 0.55 : 1)
            .scaleEffect(down ? 0.94 : 1)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { v in
                        if v.time.timeIntervalSince(lastFire) > 0.35 && abs(v.translation.width) < 4 && abs(v.translation.height) < 4 {
                            lastFire = v.time
                            action()
                            down = true
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { down = false }
                        }
                    }
                    .onEnded { _ in down = false }
            )
    }
}

extension View {
    func pressAction(_ action: @escaping () -> Void) -> some View { modifier(PressAction(action: action)) }
}

/// Virtual thumbstick. Reports x = right, y = forward in [-1, 1].
struct Joystick: View {
    var radius: CGFloat = 64
    var onChange: (float2) -> Void
    @State private var knob: CGSize = .zero

    var body: some View {
        ZStack {
            Circle()
                .fill(.ultraThinMaterial)
                .overlay(Circle().stroke(.white.opacity(0.35), lineWidth: 1.5))
                .frame(width: radius * 2, height: radius * 2)
            Circle()
                .fill(.white.opacity(0.85))
                .frame(width: radius * 0.8, height: radius * 0.8)
                .shadow(radius: 4)
                .offset(knob)
        }
        .contentShape(Circle().inset(by: -30))
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { v in
                    var t = v.translation
                    let len = sqrt(t.width * t.width + t.height * t.height)
                    if len > radius { t = CGSize(width: t.width / len * radius, height: t.height / len * radius) }
                    knob = t
                    onChange(float2(Float(t.width / radius), Float(-t.height / radius)))
                }
                .onEnded { _ in
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.6)) { knob = .zero }
                    onChange(.zero)
                }
        )
    }
}

/// Round button that reports press / release (for held actions like flying up).
struct HoldButton: View {
    let systemImage: String
    var onPress: (Bool) -> Void
    @State private var down = false

    var body: some View {
        Image(systemName: systemImage)
            .font(.system(size: 22, weight: .semibold))
            .frame(width: 58, height: 58)
            .background(down ? AnyShapeStyle(.white.opacity(0.45)) : AnyShapeStyle(.ultraThinMaterial), in: Circle())
            .overlay(Circle().stroke(.white.opacity(0.35), lineWidth: 1.2))
            .foregroundStyle(.white)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in if !down { down = true; onPress(true) } }
                    .onEnded { _ in down = false; onPress(false) }
            )
    }
}

struct ToggleCircle: View {
    let systemImage: String
    let label: String
    let isOn: Bool
    let action: () -> Void

    var body: some View {
        VStack(spacing: 2) {
            Image(systemName: systemImage).font(.system(size: 19, weight: .semibold))
            Text(label).font(.system(size: 9, weight: .semibold))
        }
        .frame(width: 58, height: 58)
        .background(isOn ? AnyShapeStyle(.white.opacity(0.4)) : AnyShapeStyle(.ultraThinMaterial), in: Circle())
        .overlay(Circle().stroke(.white.opacity(0.35), lineWidth: 1.2))
        .foregroundStyle(.white)
        .pressAction(action)
    }
}

/// Full-screen touch layer: drag anywhere to look, joystick bottom-left, action buttons bottom-right.
struct TouchControls: View {
    let renderer: Renderer
    @ObservedObject var state: AppState
    @State private var sprint = false

    var body: some View {
        // (looking around is handled by TouchMTKView underneath: drag on any empty area)
        ZStack {
            if !state.hideUI {
                VStack {
                    Spacer()
                    HStack(alignment: .bottom) {
                        Joystick { renderer.input.touchMove = $0 }
                            .padding(.leading, 44)
                            .padding(.bottom, 34)
                        Spacer()
                        HStack(alignment: .bottom, spacing: 14) {
                            if !state.walkMode {
                                VStack(spacing: 14) {
                                    HoldButton(systemImage: "arrow.up") { renderer.input.touchVertical = $0 ? 1 : 0 }
                                    HoldButton(systemImage: "arrow.down") { renderer.input.touchVertical = $0 ? -1 : 0 }
                                }
                            }
                            VStack(spacing: 14) {
                                ToggleCircle(systemImage: "hare.fill", label: "RUN", isOn: sprint) {
                                    sprint.toggle()
                                    renderer.input.touchSprint = sprint
                                }
                                ToggleCircle(systemImage: state.walkMode ? "figure.walk" : "airplane",
                                             label: state.walkMode ? "WALK" : "FLY", isOn: !state.walkMode) {
                                    state.walkMode.toggle()
                                    renderer.input.touchVertical = 0
                                }
                            }
                        }
                        .padding(.trailing, 40)
                        .padding(.bottom, 30)
                    }
                }
            } else {
                // bring the UI back
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        Image(systemName: "eye").font(.system(size: 15)).padding(10)
                            .background(.black.opacity(0.25), in: Circle()).foregroundStyle(.white.opacity(0.6))
                            .pressAction { state.hideUI = false }
                            .padding(20)
                    }
                }
            }
        }
    }
}
#endif
