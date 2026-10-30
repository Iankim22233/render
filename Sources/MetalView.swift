#if os(macOS)
import AppKit
import MetalKit
import SwiftUI

final class ForestMTKView: MTKView {
    weak var renderer: Renderer?
    var state: AppState?
    private var dragging = false

    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)
        window?.acceptsMouseMovedEvents = true
    }

    override func keyDown(with event: NSEvent) {
        guard let r = renderer, let st = state else { return }
        if !event.isARepeat {
            // Screenshot without any UI: F12 or ⌘S
            if event.keyCode == Key.f12 || (event.keyCode == Key.s && event.modifierFlags.contains(.command)) {
                st.screenshotRequested = true
                return
            }
            switch event.keyCode {
            case Key.u: st.hideUI.toggle(); return
            case Key.tab: st.showPanel.toggle(); return
            case Key.h: st.showHelp.toggle(); return
            case Key.g: st.walkMode.toggle(); return
            case Key.p: st.timePaused.toggle(); return
            case Key.escape: releaseMouse(); return
            default: break
            }
        }
        r.input.keys.insert(event.keyCode)
    }

    override func keyUp(with event: NSEvent) {
        renderer?.input.keys.remove(event.keyCode)
    }

    override func flagsChanged(with event: NSEvent) {
        renderer?.input.shift = event.modifierFlags.contains(.shift)
        renderer?.input.option = event.modifierFlags.contains(.option)
    }

    override func resignFirstResponder() -> Bool {
        renderer?.input.clear()
        return super.resignFirstResponder()
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        dragging = true
    }

    override func mouseUp(with event: NSEvent) { dragging = false }

    override func mouseDragged(with event: NSEvent) {
        renderer?.input.mouseDelta += float2(Float(event.deltaX), Float(event.deltaY))
    }

    override func rightMouseDragged(with event: NSEvent) {
        renderer?.input.mouseDelta += float2(Float(event.deltaX), Float(event.deltaY))
    }

    override func scrollWheel(with event: NSEvent) {
        guard let st = state else { return }
        let f = Float(1 + event.scrollingDeltaY * (event.hasPreciseScrollingDeltas ? 0.004 : 0.05))
        st.moveSpeed = min(max(st.moveSpeed * f, 0.5), 120)
    }

    private func releaseMouse() {
        dragging = false
    }
}

struct MetalView: NSViewRepresentable {
    let renderer: Renderer
    let state: AppState

    func makeNSView(context: Context) -> ForestMTKView {
        let v = ForestMTKView(frame: .zero, device: renderer.device)
        v.colorPixelFormat = renderer.outputFormat
        v.depthStencilPixelFormat = .invalid
        v.framebufferOnly = true
        v.preferredFramesPerSecond = 120
        v.enableSetNeedsDisplay = false
        v.isPaused = false
        v.delegate = renderer
        v.renderer = renderer
        v.state = state
        return v
    }

    func updateNSView(_ nsView: ForestMTKView, context: Context) {}
}

#endif
