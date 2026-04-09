#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLENDER_BIN="${BLENDER_BIN:-$ROOT/third_party/blender/blender-4.2.2-linux-x64/blender}"
ADDON_ROOT="${ADDON_ROOT:-$ROOT/third_party/PantoMatrix/blender_assets/unpacked/smplx_addon}"
FFMPEG_BIN="${FFMPEG_BIN:-}"
SMOKE_SCRIPT="$ROOT/tools/blender/smoke_render.py"

if [[ -z "$FFMPEG_BIN" ]]; then
  if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_BIN="$(command -v ffmpeg)"
  else
    for candidate in \
      "/home/vanto/anaconda3/envs/pantomatrix-emage39/bin/ffmpeg" \
      "/home/vanto/anaconda3/envs/galatea/bin/ffmpeg" \
      "/home/vanto/anaconda3/envs/convofusion-demo/bin/ffmpeg" \
      "/home/vanto/anaconda3/envs/a2p-blackwell/bin/ffmpeg"; do
      if [[ -x "$candidate" ]]; then
        FFMPEG_BIN="$candidate"
        break
      fi
    done
  fi
fi

echo "[check] blender: $BLENDER_BIN"
"$BLENDER_BIN" -b --version | sed -n '1,2p'

echo "[check] ffmpeg"
"$FFMPEG_BIN" -version | sed -n '1,2p'

echo "[check] addon root: $ADDON_ROOT"
test -d "$ADDON_ROOT/smplx_blender_addon"

echo "[check] smoke render"
PANTOMATRIX_ROOT="$ROOT" "$BLENDER_BIN" -b --python "$SMOKE_SCRIPT" >/tmp/blender_smplx_bootstrap.log 2>&1
test -f "$ROOT/data/blender_smoke/smoke.png"

echo "[ok] Blender SMPL-X render prerequisites look ready"
