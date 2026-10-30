import Foundation
import GameController
import simd

/// Game controller support (Xbox, PlayStation, MFi, Switch Pro…) on macOS and iOS,
/// plus hardware-keyboard WASD on iPad via GCKeyboard.
///
/// Left stick: move · Right stick: look · R1/L1: up/down (fly) · R2: sprint
/// Y/△: walk/fly · X/□: screenshot · B/○: hide UI · Menu/Options: settings panel · View/Share: help
final class GamepadInput {
    private var previous: [String: Bool] = [:]

    private func deadzone(_ v: float2, _ dz: Float = 0.14) -> float2 {
        let l = simd_length(v)
        if l < dz { return .zero }
        return v / l * min((l - dz) / (1 - dz), 1)
    }

    private func pressed(_ key: String, _ isDown: Bool, action: () -> Void) {
        if isDown && !(previous[key] ?? false) { action() }
        previous[key] = isDown
    }

    func poll(input: InputState, state: AppState) {
        var move = float2(0, 0), look = float2(0, 0)
        var vertical: Float = 0, sprint: Float = 0
        if let pad = GCController.current?.extendedGamepad ?? GCController.controllers().first?.extendedGamepad {
            move = deadzone(float2(pad.leftThumbstick.xAxis.value, pad.leftThumbstick.yAxis.value))
            look = deadzone(float2(pad.rightThumbstick.xAxis.value, pad.rightThumbstick.yAxis.value))
            vertical = pad.rightShoulder.value - pad.leftShoulder.value
            sprint = pad.rightTrigger.value
            pressed("y", pad.buttonY.isPressed) { state.walkMode.toggle() }
            pressed("x", pad.buttonX.isPressed) { state.screenshotRequested = true }
            pressed("b", pad.buttonB.isPressed) { state.hideUI.toggle() }
            pressed("menu", pad.buttonMenu.isPressed) { state.showPanel.toggle() }
            if let opt = pad.buttonOptions { pressed("opt", opt.isPressed) { state.showHelp.toggle() } }
        }
        #if os(iOS)
        if let kb = GCKeyboard.coalesced?.keyboardInput {
            func down(_ k: GCKeyCode) -> Bool { kb.button(forKeyCode: k)?.isPressed ?? false }
            var m = float2(0, 0)
            if down(.keyW) { m.y += 1 }
            if down(.keyS) { m.y -= 1 }
            if down(.keyD) { m.x += 1 }
            if down(.keyA) { m.x -= 1 }
            if down(.upArrow) { look.y += 1 }
            if down(.downArrow) { look.y -= 1 }
            if down(.rightArrow) { look.x += 1 }
            if down(.leftArrow) { look.x -= 1 }
            if down(.keyE) || down(.spacebar) { vertical += 1 }
            if down(.keyQ) || down(.keyC) { vertical -= 1 }
            if down(.leftShift) || down(.rightShift) { sprint = 1 }
            move += m
            pressed("kg", down(.keyG)) { state.walkMode.toggle() }
            pressed("ku", down(.keyU)) { state.hideUI.toggle() }
            pressed("kp", down(.keyP)) { state.timePaused.toggle() }
            pressed("ktab", down(.tab)) { state.showPanel.toggle() }
        }
        #endif
        input.padMove = move
        input.padLook = look
        input.padVertical = max(-1, min(1, vertical))
        input.padSprint = sprint
    }
}
