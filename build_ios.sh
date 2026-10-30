#!/bin/zsh
# Builds Forest Render for iOS / iPadOS (no Xcode project needed).
#   ./build_ios.sh          -> ForestRender.ipa (device, unsigned: sign/install with iloader, Sideloadly, AltStore…)
#   ./build_ios.sh sim      -> build/ios-sim/ForestRender.app for the iOS Simulator
# Optional signing for a direct device install: SIGN_IDENTITY="Apple Development: …" PROFILE=path.mobileprovision ./build_ios.sh
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-device}"
if [[ "$MODE" == "sim" ]]; then
  SDK=iphonesimulator; TARGET=arm64-apple-ios17.0-simulator; PLATFORM=iPhoneSimulator; OUT=build/ios-sim
else
  SDK=iphoneos; TARGET=arm64-apple-ios17.0; PLATFORM=iPhoneOS; OUT=build/ios
fi
APP="$OUT/Payload/ForestRender.app"
rm -rf "$OUT"
mkdir -p "$APP/Shaders"

echo "Compiling Swift for $SDK…"
xcrun --sdk $SDK swiftc -O -wmo -swift-version 5 \
  -target $TARGET \
  -sdk "$(xcrun --sdk $SDK --show-sdk-path)" \
  -framework Metal -framework MetalKit -framework UIKit -framework SwiftUI -framework GameController -framework QuartzCore \
  -o "$APP/ForestRender" \
  Sources/*.swift

cp Shaders/*.metal "$APP/Shaders/"

echo "Compiling app icon…"
xcrun actool Assets/iOS.xcassets --compile "$APP" --platform $SDK --minimum-deployment-target 17.0 \
  --app-icon AppIcon --target-device iphone --target-device ipad \
  --output-partial-info-plist "$OUT/icon-partial.plist" >/dev/null

cat > "$APP/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Forest Render</string>
  <key>CFBundleDisplayName</key><string>Forest Render</string>
  <key>CFBundleIdentifier</key><string>com.local.forestrender</string>
  <key>CFBundleExecutable</key><string>ForestRender</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleSupportedPlatforms</key><array><string>$PLATFORM</string></array>
  <key>DTPlatformName</key><string>$SDK</string>
  <key>MinimumOSVersion</key><string>17.0</string>
  <key>LSRequiresIPhoneOS</key><true/>
  <key>UIDeviceFamily</key><array><integer>1</integer><integer>2</integer></array>
  <key>UIRequiredDeviceCapabilities</key><array><string>arm64</string><string>metal</string></array>
  <key>UILaunchScreen</key><dict><key>UIColorName</key><string></string></dict>
  <key>UIRequiresFullScreen</key><true/>
  <key>UIStatusBarHidden</key><true/>
  <key>UIViewControllerBasedStatusBarAppearance</key><false/>
  <key>UISupportedInterfaceOrientations</key>
  <array><string>UIInterfaceOrientationLandscapeLeft</string><string>UIInterfaceOrientationLandscapeRight</string><string>UIInterfaceOrientationPortrait</string></array>
  <key>UISupportedInterfaceOrientations~ipad</key>
  <array><string>UIInterfaceOrientationLandscapeLeft</string><string>UIInterfaceOrientationLandscapeRight</string><string>UIInterfaceOrientationPortrait</string><string>UIInterfaceOrientationPortraitUpsideDown</string></array>
  <key>UIApplicationSupportsIndirectInputEvents</key><true/>
  <key>GCSupportsControllerUserInteraction</key><true/>
  <key>GCSupportedGameControllers</key>
  <array><dict><key>ProfileName</key><string>ExtendedGamepad</string></dict></array>
  <key>NSPhotoLibraryAddUsageDescription</key><string>Forest Render saves your screenshots to Photos.</string>
</dict>
</plist>
PLIST
/usr/libexec/PlistBuddy -c "Merge $OUT/icon-partial.plist" "$APP/Info.plist" >/dev/null
printf 'APPL????' > "$APP/PkgInfo"

if [[ "$MODE" == "sim" ]]; then
  codesign --force --sign - "$APP" >/dev/null
  echo "Built $APP (simulator)"
  exit 0
fi

if [[ -n "${SIGN_IDENTITY:-}" && -n "${PROFILE:-}" ]]; then
  cp "$PROFILE" "$APP/embedded.mobileprovision"
  security cms -D -i "$PROFILE" > "$OUT/profile.plist"
  /usr/libexec/PlistBuddy -x -c "Print :Entitlements" "$OUT/profile.plist" > "$OUT/entitlements.plist"
  codesign --force --sign "$SIGN_IDENTITY" --entitlements "$OUT/entitlements.plist" "$APP"
  echo "Signed with $SIGN_IDENTITY"
fi

(cd "$OUT" && rm -f ../../ForestRender.ipa && zip -qry ../../ForestRender.ipa Payload)
echo "Built ForestRender.ipa ($(du -h ForestRender.ipa | cut -f1))"
