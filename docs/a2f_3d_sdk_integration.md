# Audio2Face-3D SDK Integration Notes

Date: 2026-04-09

This workspace now supports `Audio2Face-3D SDK` as a local face provider for the hybrid pipeline under the provider name `a2f-3d-sdk`.

## Scope

- `EMAGE` remains the body source.
- `LAM_Audio2Expression` remains available for A/B comparison.
- `A2F-3D SDK` is integrated in raw-fidelity mode:
  - no synthetic eye-motion compensation
  - no repository-level `MouthClose` renormalization
  - provider-native coefficients are preserved for comparison

## Local Toolchain Strategy

This workstation currently uses system `CUDA 13.2`, while the A2F-3D SDK integration path targets a side-by-side toolchain under:

- `$HOME/.local/nvidia/a2f3d-sdk/cuda-12.9`
- `$HOME/.local/nvidia/a2f3d-sdk/tensorrt-10.13`

Bootstrap entrypoints:

- `tools/bootstrap_a2f3d_sdk_toolchain.sh`
- `tools/bootstrap_a2f3d_sdk_env.sh`
- `tools/verify_a2f3d_sdk_env.py`

Dedicated conda env:

- `a2f3d-sdk310`

## Example Bootstrap Flow

```bash
CUDA_RUNFILE=/path/to/cuda_12.9_linux.run \
TENSORRT_TAR=/path/to/TensorRT-10.13.x.x.Linux.x86_64-gnu.cuda-12.9.tar.gz \
bash tools/bootstrap_a2f3d_sdk_toolchain.sh

TOOLCHAIN_ROOT=$HOME/.local/nvidia/a2f3d-sdk \
SDK_ROOT=/home/vanto/pantomatrix_ws/third_party/Audio2Face-3D-SDK \
bash tools/bootstrap_a2f3d_sdk_env.sh
```

## Provider Config

Checked-in example config:

- `configs/a2f3d_sdk.example.json`

The wrapper script:

- `tools/run_a2f3d_sdk_inference.py`

accepts either:

- a machine-local `command` template for live SDK inference
- or `raw_output_json` for replaying a previously generated A2F payload

## Hybrid Usage

Single-provider run:

```bash
python tools/hybrid_pipeline.py \
  --audio data/audio/audio-demo-female.wav \
  --body-npz data/emage_outputs/audio-demo-female_output.npz \
  --output-dir /tmp/hybrid_a2f_demo \
  --face-provider a2f-3d-sdk \
  --face-python /home/vanto/anaconda3/envs/a2f3d-sdk310/bin/python \
  --face-repo /home/vanto/pantomatrix_ws/third_party/Audio2Face-3D-SDK \
  --face-config-file /home/vanto/pantomatrix_ws/configs/a2f3d_sdk.example.json
```

Comparison run:

```bash
python tools/run_hybrid_provider_comparison.py \
  --audio data/audio/audio-demo-female.wav \
  --body-npz data/emage_outputs/audio-demo-female_output.npz \
  --output-root /tmp/hybrid_face_compare \
  --a2f-config-file /home/vanto/pantomatrix_ws/configs/a2f3d_sdk.example.json
```

## Output Notes

Each hybrid run now writes:

- `face/arkit_blendshapes.json`
- `face/provider_metadata.json`

For `a2f-3d-sdk`, `provider_metadata.json` records:

- `provider = "a2f-3d-sdk"`
- `normalization_policy = "raw-a2f"`
- provider caveats about eye behavior and `MouthClose`

## Render Path

The showcase renderer is now provider-aware:

- `lam` defaults to browser/WebGL rendering when available
- `a2f-3d-sdk` defaults to coefficient-to-2D rendering for stable `mp4` output
