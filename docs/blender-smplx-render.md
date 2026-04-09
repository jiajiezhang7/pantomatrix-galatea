# Blender SMPL-X Render Pipeline

Date: 2026-04-09

## Summary

This workspace now supports a fully automated fallback rendering path:

`EMAGE .npz -> Blender + official SMPL-X addon -> MP4 with synced audio`

This path does **not** use:

- `BEAT_Avatars`
- `HumGen`
- any `SMPL-X -> HG rig` retarget

The rendered character is the official SMPL-X body imported by the Blender addon.

## Current Entrypoints

- Bootstrap / environment check:
  - `tools/bootstrap_blender_smplx_render.sh`
- Batch render wrapper:
  - `tools/render_emage_smplx_video.py`
- Blender-internal scene script:
  - `tools/blender/render_emage_smplx_scene.py`

## Default Inputs

- Motion:
  - `data/emage_outputs/*.npz`
- Audio:
  - `data/audio/*.wav`

The wrapper matches:

- `<clip>_output.npz` -> `<clip>.wav`

Example:

- `audio-demo-female_output.npz` -> `audio-demo-female.wav`

## Default Outputs

For each clip:

- final MP4:
  - `data/blender_renders_smplx/<clip>/<clip>.mp4`
- preview frame:
  - `data/blender_renders_smplx/<clip>/preview.png`
- intermediate silent video:
  - `data/blender_renders_smplx/<clip>/work/silent.mp4`
- render log:
  - `demo_logs/blender_smplx/<clip>.log`

## Environment And Asset Assumptions

### Blender binaries

- preferred:
  - `third_party/blender/blender-4.2.2-linux-x64/blender`
- legacy fallback kept locally:
  - `third_party/blender3/blender-3.0.1-linux-x64/blender`

### SMPL-X addon root

- `third_party/PantoMatrix/blender_assets/unpacked/smplx_addon/`

### ffmpeg

The wrapper first tries `ffmpeg` from `PATH`.

If not found, it falls back to known local conda binaries, including:

- `/home/vanto/anaconda3/envs/pantomatrix-emage39/bin/ffmpeg`
- `/home/vanto/anaconda3/envs/galatea/bin/ffmpeg`

## Render Defaults

- engine preference:
  - `Cycles`
- device preference:
  - auto-detect GPU
- observed current device on this server:
  - `OPTIX`
- resolution:
  - `1280x720`
- framerate:
  - follows `.npz` / imported scene fps, currently `30`
- scene style:
  - simple gray studio backdrop
  - ground plane
  - three-light setup
  - single front three-quarter camera
  - neutral non-photoreal SMPL-X material

## Current Validation Result

Validated end-to-end on:

- `data/emage_outputs/audio-demo-female_output.npz`
- `data/emage_outputs/audio-demo-male_output.npz`

Produced:

- `data/blender_renders_smplx/audio-demo-female/audio-demo-female.mp4`
- `data/blender_renders_smplx/audio-demo-male/audio-demo-male.mp4`

Both outputs contain:

- H.264 video
- AAC audio

## Failure And Fallback Behavior

- If `Cycles` render setup fails in the wrapper, it retries with:
  - `BLENDER_EEVEE_NEXT`
- If audio matching fails, the wrapper errors out for that clip and does not guess.
- If Blender render fails, the wrapper writes clip-specific logs and continues to the next clip.
- If `ffmpeg` mux fails, the silent render is preserved in the clip work directory.

## Practical Positioning

This path is the current recommended fully automatic render route for this repository.

It is intentionally a fallback path:

- more automated than `BEAT_Avatars`
- less visually rich than a true high-end HumGen / HumanGenerator character pipeline

That tradeoff is currently accepted because it keeps the pipeline reproducible and headless-safe.
