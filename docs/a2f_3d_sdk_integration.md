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
- `configs/a2f_profiles/raw-regression.json`
- `configs/a2f_profiles/tuned-regression-v1.json`
- `configs/a2f_profiles/tuned-regression-v2.json`
- `configs/a2f_profiles/diffusion-candidate.json`

The wrapper script:

- `tools/run_a2f3d_sdk_inference.py`

accepts either:

- a machine-local `command` template for live SDK inference
- or `raw_output_json` for replaying a previously generated A2F payload

For profile-driven local tuning, the wrapper now also supports:

- `base_profile_dir` plus `profile_overrides` to stage a modified SDK sample profile at run time
- `profile_name`, `model_identity`, `tuning_summary`, and `quality_study_group` metadata
- diffusion-specific knobs such as `identity_index` and `constant_noise`

## Hybrid Usage

Single-provider run:

```bash
python tools/hybrid_pipeline.py \
  --audio output/migrated_workspace_artifacts/data/audio/audio-demo-female.wav \
  --body-npz output/migrated_workspace_artifacts/data/emage_outputs/audio-demo-female_output.npz \
  --output-dir output/hybrid_a2f_demo \
  --face-provider a2f-3d-sdk \
  --face-python /home/vanto/anaconda3/envs/a2f3d-sdk310/bin/python \
  --face-repo /home/vanto/pantomatrix_ws/third_party/Audio2Face-3D-SDK \
  --face-config-file /home/vanto/pantomatrix_ws/configs/a2f3d_sdk.example.json
```

Comparison run:

```bash
python tools/run_hybrid_provider_comparison.py \
  --audio output/migrated_workspace_artifacts/data/audio/audio-demo-female.wav \
  --body-npz output/migrated_workspace_artifacts/data/emage_outputs/audio-demo-female_output.npz \
  --output-root output/hybrid_face_compare \
  --a2f-config-file /home/vanto/pantomatrix_ws/configs/a2f3d_sdk.example.json
```

Quality-study run:

```bash
python tools/run_a2f_quality_study.py \
  --audio output/migrated_workspace_artifacts/data/audio/audio-demo-female.wav \
  --body-npz output/migrated_workspace_artifacts/data/emage_outputs/audio-demo-female_output.npz \
  --output-root output/a2f_quality_study
```

## Output Notes

Each hybrid run now writes:

- `face/arkit_blendshapes.json`
- `face/provider_metadata.json`
- `face/profile_config_snapshot.json` for profile-driven A2F runs
- `metrics/face_quality_report.json`

For `a2f-3d-sdk`, `provider_metadata.json` records:

- `provider = "a2f-3d-sdk"`
- `normalization_policy = "raw-a2f"`
- `profile_name`
- `mode`
- `tuning_summary`
- `quality_study_group`
- provider caveats about eye behavior and `MouthClose`

## Render Path

The showcase renderer is now provider-aware:

- `lam` defaults to browser/WebGL rendering
- `a2f-3d-sdk` now also defaults to browser/WebGL rendering through the same 3D avatar lane
- `coeff2d` remains available as a manual fallback via `--render-mode coeff2d`
- the verified headless-server path uses the `lam-a2e310` env plus `Xvfb` from that env
