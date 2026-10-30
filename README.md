# Forest Render

A native macOS (Swift + Metal) real-time forest explorer with a custom deferred renderer, built for
realistic sun light, soft shadows and volumetric god rays. Everything — terrain, trees, bark, leaves,
forest-floor textures — is generated procedurally at launch (no asset downloads).

## Build & run

```bash
./build.sh            # builds ForestRender.app (needs Xcode command-line tools)
open ForestRender.app
```

Requires macOS 14+ on Apple silicon. Shaders are compiled at runtime from `Shaders/*.metal`
(copied into the app bundle), so no Metal toolchain download is needed.

## iPhone / iPad (IPA)

```bash
./build_ios.sh        # -> ForestRender.ipa (unsigned; sign + install with iloader / Sideloadly / AltStore)
./build_ios.sh sim    # -> build/ios-sim/… for the iOS Simulator
```

Touch controls: **left joystick** to move, **drag anywhere** to look, **RUN** and **WALK/FLY** buttons,
↑/↓ buttons while flying, toolbar for help / hide UI / screenshot (saved to Photos) / settings.
Game controllers (Xbox, PlayStation, MFi) work on iOS and macOS: left stick move, right stick look,
R1/L1 up/down, R2 sprint, Y walk/fly, X screenshot, B hide UI, Menu settings. iPad hardware keyboards
use the same WASD keys as the Mac. Defaults to the Medium preset; 8K shadow maps are capped at 4K on iOS.

## Controls

| Input | Action |
|---|---|
| Drag mouse (left or right) / arrow keys | Look around |
| W A S D | Move |
| Shift / Option | Sprint ×4 / slow ×0.25 |
| G | Toggle walk (ground-following) / fly |
| Q/E or C/Space | Down / up (fly mode) |
| Scroll | Change move speed |
| Tab | Show/hide settings panel |
| P | Pause time (wind, clouds) |
| H | Hide help |
| U | Hide all UI (clean view) |
| **F12** or **⌘S** | Save a UI-free screenshot (full window resolution) to `~/Pictures/Forest Render/` |

## Forests

1. **Nordic Spruce — Dawn Mist**: dense spruce taiga, low sun, mist
2. **Golden Birch Grove**: autumn birches, warm late-afternoon light
3. **Ancient Redwoods**: 50–75 m giants, ferns, coastal fog, strong light shafts
4. **Autumn Oak Woodland**: oaks and beeches turning copper and gold
5. **Summer Pine Heath — Midday**: open Scots pine forest under a high sun
6. **Misty Beech Cathedral**: tall beeches in thick morning fog
7. **Mirror Lake — Alpine Sunset**: still mountain lake reflecting a golden spruce shore
8. **Rainy Birch Lake**: steady rain, wet ground and puddles
9. **Misty Pine Lake — Morning**: fog drifting over a calm lake at sunrise
10. **Thunderstorm Taiga Lake**: heavy rain, wind and lightning
11. **Winter Taiga — Fresh Snow**: snow-laden spruces in low winter sun
12. **Starry Night Lake**: moonless night, the Milky Way mirrored in a still lake

Rain, thunderstorms and **snow** can be switched on in **any** world from the Weather section.
The sun can go all the way below the horizon (−90°…90°): twilight comes from the sky model, and once
it's dark the **Stars** slider controls the star field and Milky Way. The **moon** has a phase slider
(new → full → new), a height slider, and a **Moonlight** slider for how bright the ground is at night. At night the moon
becomes the key light: moon shadows, moonbeams through the trees and a moon-glitter path on the lakes.

Sun elevation/direction, fog, haze, clouds, wind and autumn colour are live sliders.

## Rendering features

- Deferred G-buffer renderer, reversed-Z infinite projection
- **Cascaded shadow maps** (4 cascades, up to 8192²) with texel snapping, staggered far-cascade updates
  and per-cascade resolution
- **PCSS contact-hardening soft shadows** using the physical sun angular size (dappled light through
  leaves), plus screen-space contact shadows and moving cloud shadows
- **Volumetric lighting**: ray-marched, shadow-mapped in-scattering in height fog with 3D noise and
  Henyey-Greenstein phase → real god rays through the canopy
- Physically based **atmosphere** (Rayleigh/Mie/ozone single scattering + multi-scatter approx.),
  sun disk, procedural cloud layer; sky ambient via L1 spherical harmonics
- Canopy sky-occlusion map (dark forest floors, bright clearings), **GTAO** ambient occlusion with
  multi-bounce
- GGX specular, foliage **translucency** (glowing backlit leaves), moss on rocks/logs/trunk bases
- Procedural trees (spruce, Scots pine, oak, birch, beech, redwood) with 3 mesh LODs + dithered
  cross-fades + **impostor billboards** for distant trees; ferns, shrubs, rocks, fallen logs
- GPU-generated grass (up to ~1M wind-animated blades), hex-tiled height-blended ground materials
- Foliage depth prepass (one shading pass per pixel on Apple TBDR GPUs)
- **Lakes**: carved basins with irregular shores, planar reflections (mirrored G-buffer + lighting pass),
  refraction, wavelength-dependent absorption, underwater caustics, wind waves, sun glint, reeds in the shallows
- **Rain**: up to ~160K motion-blurred streaks (occluded by canopy, depth-tested), ripples on lakes and
  puddles, surfaces that wet up over time (darker, glossier) and dry out after, overcast sky, lightning flashes
- **Snow**: falling flakes with flutter and wind drift; accumulation on up-facing ground, rocks, logs, branches and
  conifer crowns (less under dense canopy), snow on the windward side of trunks, buried grass, glinting snow crystals
- **Night**: airglow, procedural twinkling star field + Milky Way (reflected in lakes), night-aware eye adaptation and
  low-light (scotopic) colour shift
- **TAA** (Halton jitter, Catmull-Rom history, variance clipping), bloom, auto exposure,
  AgX / ACES tone mapping, sharpening, vignette, film grain

## Quality presets

`Low → Medium → High (default) → Ultra → Insane`. Render scale is relative to the window's point size
(2.0 = native Retina pixels). *Insane* (8K shadows, full-res 128-step volumetrics, 1.6× scale) is
meant for screenshots or very fast GPUs.

## Headless snapshot (for testing / wallpapers)

```bash
ForestRender.app/Contents/MacOS/ForestRender --snapshot out.png --scene 2 --quality ultra \
  --width 2560 --height 1440 --frames 60
```

Other flags: `--rain 0-1`, `--snow 0-1`, `--stars 0-1`, `--moon-phase 0-1`, `--moonlight 0-1`, `--moon-elev`, `--fov`, `--sun-elev`, `--sun-az`, `--fog`, `--yaw-offset`, `--pitch`, `--pos x,y,z`, `--debug 1-4`.

## Source layout

- `Sources/Renderer.swift`: frame orchestration, culling, cascades, impostors
- `Sources/World.swift`: terrain, placement, canopy/grass maps
- `Sources/TreeGen.swift`: procedural tree/plant/rock meshes and LODs
- `Sources/TextureGen.swift`: procedural bark, leaf and ground textures
- `Shaders/`: Common, Geometry (terrain/trees/grass/impostors), Lighting (sky, shadows, GTAO,
  volumetrics, deferred lighting), Post (TAA, bloom, exposure, tone mapping)
