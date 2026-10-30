#!/bin/zsh
# Builds ForestRender.app (Swift + Metal, no Xcode project needed).
# Usage: ./build.sh [debug]
set -euo pipefail
cd "$(dirname "$0")"

APP="ForestRender.app"
CONFIG="${1:-release}"
OPT=(-O -wmo)
[[ "$CONFIG" == "debug" ]] && OPT=(-Onone -g)

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/Shaders"

xcrun swiftc "${OPT[@]}" \
  -swift-version 5 \
  -target arm64-apple-macos14.0 \
  -sdk "$(xcrun --show-sdk-path --sdk macosx)" \
  -framework Metal -framework MetalKit -framework AppKit -framework SwiftUI -framework QuartzCore \
  -o "$APP/Contents/MacOS/ForestRender" \
  Sources/*.swift

cp Shaders/*.metal "$APP/Contents/Resources/Shaders/"

# App icon: Assets/AppIcon.png (1024×1024) -> AppIcon.icns
if [[ -f Assets/AppIcon.png ]]; then
  ICONSET="$(mktemp -d)/AppIcon.iconset"
  mkdir -p "$ICONSET"
  for s in 16 32 128 256 512; do
    sips -z $s $s Assets/AppIcon.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    sips -z $((s * 2)) $((s * 2)) Assets/AppIcon.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
  rm -rf "$(dirname "$ICONSET")"
fi

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Forest Render</string>
  <key>CFBundleDisplayName</key><string>Forest Render</string>
  <key>CFBundleIdentifier</key><string>com.local.forestrender</string>
  <key>CFBundleExecutable</key><string>ForestRender</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST

codesign --force --sign - "$APP" >/dev/null 2>&1 || true
# refresh Finder/Dock icon caches for the rebuilt bundle
touch "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" >/dev/null 2>&1 || true
echo "Built $APP"
