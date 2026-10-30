import Foundation
import simd

final class InputState {
    var keys = Set<UInt16>()
    var shift = false
    var option = false
    var mouseDelta = float2(0, 0)
    var scrollDelta: Float = 0
    // Analog input: on-screen joystick / buttons (touch) and game controllers (pad). x = right, y = forward/up.
    var touchMove = float2(0, 0)
    var touchVertical: Float = 0
    var touchSprint = false
    var padMove = float2(0, 0)
    var padLook = float2(0, 0)
    var padVertical: Float = 0
    var padSprint: Float = 0

    func clear() { keys.removeAll(); shift = false; option = false; mouseDelta = .zero }
}

enum Key {
    static let a: UInt16 = 0, s: UInt16 = 1, d: UInt16 = 2, f: UInt16 = 3, h: UInt16 = 4, g: UInt16 = 5
    static let c: UInt16 = 8, q: UInt16 = 12, w: UInt16 = 13, e: UInt16 = 14, r: UInt16 = 15, p: UInt16 = 35
    static let tab: UInt16 = 48, space: UInt16 = 49, escape: UInt16 = 53, u: UInt16 = 32, f12: UInt16 = 111
    static let left: UInt16 = 123, right: UInt16 = 124, down: UInt16 = 125, up: UInt16 = 126
}

final class Camera {
    var position = float3(0, 2, 0)
    var yaw: Float = 0
    var pitch: Float = 0
    var velocity = float3(0, 0, 0)

    var forward: float3 { float3(cos(pitch) * sin(yaw), sin(pitch), -cos(pitch) * cos(yaw)) }
    var flatForward: float3 { float3(sin(yaw), 0, -cos(yaw)) }
    var right: float3 { float3(cos(yaw), 0, sin(yaw)) }
    var up: float3 { simd_cross(right, forward) }
    var viewMatrix: float4x4 { float4x4.lookAt(eye: position, center: position + forward, up: float3(0, 1, 0)) }

    func update(dt: Float, input: InputState, speed baseSpeed: Float, walk: Bool, groundHeight: (Float, Float) -> Float, bounds: Float) {
        let sens: Float = 0.0022
        yaw += input.mouseDelta.x * sens
        pitch -= input.mouseDelta.y * sens
        input.mouseDelta = .zero
        let lookSpeed: Float = 1.6 * dt
        if input.keys.contains(Key.left) { yaw -= lookSpeed }
        if input.keys.contains(Key.right) { yaw += lookSpeed }
        if input.keys.contains(Key.up) { pitch += lookSpeed }
        if input.keys.contains(Key.down) { pitch -= lookSpeed }
        // right stick: faster near full deflection for precise small adjustments
        let look = input.padLook * simd_length(input.padLook)
        yaw += look.x * 2.6 * dt
        pitch += look.y * 1.9 * dt
        pitch = clampf(pitch, -1.5, 1.5)

        var move = float3(0, 0, 0)
        let fwd = walk ? flatForward : forward
        if input.keys.contains(Key.w) { move += fwd }
        if input.keys.contains(Key.s) { move -= fwd }
        if input.keys.contains(Key.d) { move += right }
        if input.keys.contains(Key.a) { move -= right }
        if !walk {
            if input.keys.contains(Key.e) || input.keys.contains(Key.space) { move.y += 1 }
            if input.keys.contains(Key.q) || input.keys.contains(Key.c) { move.y -= 1 }
            move.y += input.touchVertical + input.padVertical
        }
        let analog = input.touchMove + input.padMove
        move += fwd * analog.y + right * analog.x
        if simd_length(move) > 1 { move = simd_normalize(move) }
        var speed = baseSpeed
        if input.shift || input.touchSprint { speed *= 4 }
        speed *= 1 + 3 * input.padSprint
        if input.option { speed *= 0.25 }
        let k = 1 - exp(-dt * 9)
        velocity += (move * speed - velocity) * k
        position += velocity * dt
        position.x = clampf(position.x, -bounds, bounds)
        position.z = clampf(position.z, -bounds, bounds)
        let g = groundHeight(position.x, position.z)
        if walk {
            let target = g + 1.7
            position.y += (target - position.y) * (1 - exp(-dt * 14))
        } else {
            position.y = max(position.y, g + 1.3)
        }
    }
}

/// CPU single-scattering atmosphere bits: direct sun transmittance (matches the GPU sky model).
enum Atmosphere {
    static let earthR: Float = 6_360_000
    static let atmoR: Float = 6_460_000
    static let rayleigh = float3(5.802e-6, 13.558e-6, 33.1e-6)
    static let mieExt: Float = 4.4e-6
    static let ozone = float3(0.650e-6, 1.881e-6, 0.085e-6)

    static func sunTransmittance(elevationDeg: Float, turbidity: Float) -> float3 {
        let el = elevationDeg * .pi / 180
        let ro = float3(0, earthR + 300, 0)
        let rd = float3(cos(el), sin(el), 0)
        // ground hit?
        let b = simd_dot(ro, rd)
        let cE = simd_dot(ro, ro) - earthR * earthR
        let discE = b * b - cE
        if discE > 0 && (-b - sqrt(discE)) > 0 {
            return float3(0, 0, 0)
        }
        let cA = simd_dot(ro, ro) - atmoR * atmoR
        let tMax = -b + sqrt(max(b * b - cA, 0))
        let steps = 48
        var od = float3(0, 0, 0)
        let dt = tMax / Float(steps)
        for i in 0..<steps {
            let p = ro + rd * (dt * (Float(i) + 0.5))
            let h = simd_length(p) - earthR
            let rD = exp(-h / 8000)
            let mD = exp(-h / 1200)
            let oD = max(0, 1 - abs(h - 25000) / 15000)
            od += (rayleigh * rD + float3(repeating: mieExt * turbidity * mD) + ozone * oD) * dt
        }
        return float3(exp(-od.x), exp(-od.y), exp(-od.z))
    }
}
