# LAM Audio2Expression Integration Notes

Date: 2026-04-09

This note records the branch-local, machine-verified integration path for `LAM_Audio2Expression` on `/home/vanto/pantomatrix_ws_hybrid`.

## Source

- Official repo: `https://github.com/aigc3d/LAM_Audio2Expression`
- Verified repo commit: `02a703c3ea7d8e360eb43098eca85ee98a083529`
- Local source checkout:
  - `third_party/LAM_Audio2Expression`

## Environment

- Conda env:
  - `lam-a2e310`
- Python:
  - `3.10.20`

### Important Machine-Specific Note

The official install script uses `torch 2.1.2 + cu121`. On this workstation's `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`, that install reports `torch.cuda.is_available() == True` but fails real CUDA kernels with:

`CUDA error: no kernel image is available for execution on the device`

To make the official LAM inference runnable on this machine, the env was upgraded after the official install step to:

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torchaudio 2.11.0+cu128`

The helper script for this branch is:

- [`tools/bootstrap_lam_a2e_env.sh`](../tools/bootstrap_lam_a2e_env.sh)

## Assets And Weights

Download helper for this branch:

- [`tools/download_lam_a2e_assets.py`](../tools/download_lam_a2e_assets.py)

Observed behavior on this machine:

- direct Hugging Face access to `3DAIGC/LAM_audio2exp` failed due network reachability
- the official OSS fallback URLs from the README succeeded

Key local paths after download:

- `third_party/LAM_Audio2Expression/pretrained_models/lam_audio2exp_streaming.tar`
- `third_party/LAM_Audio2Expression/assets/sample_audio/BarackObama_english.wav`

## Real Commands Verified

### Official sample-audio smoke

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate lam-a2e310
cd third_party/LAM_Audio2Expression
python inference.py \
  --config-file configs/lam_audio2exp_config_streaming.py \
  --options \
    save_path=/tmp/lam_official_smoke \
    weight=pretrained_models/lam_audio2exp_streaming.tar \
    audio_input=./assets/sample_audio/BarackObama_english.wav \
    save_json_path=/tmp/lam_official_smoke/bsData.json \
    ex_vol=False
```

### Workspace audio smoke

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate lam-a2e310
cd third_party/LAM_Audio2Expression
python inference.py \
  --config-file configs/lam_audio2exp_config_streaming.py \
  --options \
    save_path=/tmp/lam_workspace_audio \
    weight=pretrained_models/lam_audio2exp_streaming.tar \
    audio_input=/home/vanto/pantomatrix_ws_hybrid/data/audio/audio-demo-female.wav \
    save_json_path=/tmp/lam_workspace_audio/audio-demo-female.json \
    ex_vol=False
```

### Why `ex_vol=False` is required here

- The config defaults `ex_vol=True`
- The repo's `requirements.txt` comments out `spleeter`
- Without overriding `ex_vol=False`, inference can try to take a vocal-extraction branch that is not provisioned by the default install

## Real Output Format

Official LAM JSON is not the same as the branch's hybrid face contract.

Official output shape:

- top-level keys:
  - `names`
  - `metadata`
  - `frames`
- `metadata` includes:
  - `fps`
  - `frame_count`
  - `blendshape_names`
- each frame includes:
  - `weights` (length 52)
  - `time`
  - `rotation`

Machine-verified eye-related blendshape names observed in the real workspace run:

- `eyeBlinkLeft`
- `eyeBlinkRight`
- `eyeLookDownLeft`
- `eyeLookDownRight`
- `eyeLookInLeft`
- `eyeLookInRight`
- `eyeLookOutLeft`
- `eyeLookOutRight`
- `eyeLookUpLeft`
- `eyeLookUpRight`
- `eyeSquintLeft`
- `eyeSquintRight`
- `eyeWideLeft`
- `eyeWideRight`

Branch hybrid face contract:

- top-level:
  - `fps`
  - `frame_count`
  - `frames`
- each frame:
  - `frame`
  - `time_sec`
  - `blendshapes` as a name -> float map

The conversion from official LAM JSON to branch hybrid face payload now happens inside:

- [`tools/hybrid_pipeline.py`](../tools/hybrid_pipeline.py)

## Hybrid Integration Status

The `lam` face provider now supports two modes:

- `precomputed`: consume an already prepared face JSON via `--face-input-json`
- `official-run`: invoke official LAM inference when `--face-provider lam` and `--face-repo` are supplied

Machine-verified hybrid command:

```bash
python tools/hybrid_pipeline.py \
  --audio data/audio/audio-demo-female.wav \
  --output-dir /tmp/hybrid_lam_official_run \
  --body-npz data/emage_outputs/audio-demo-female_output.npz \
  --face-provider lam \
  --face-repo /home/vanto/pantomatrix_ws_hybrid/third_party/LAM_Audio2Expression \
  --face-python /home/vanto/anaconda3/envs/lam-a2e310/bin/python
```

Verified result:

- `manifest.components.face_provider.status == "ready"`
- `manifest.sources.face_provider == "lam"`
- `face/arkit_blendshapes.json` written by the adapter from a real official LAM run
- `face/arkit_blendshapes.json` preserves LAM eye-related blendshapes inside the same face payload; there is no separate eye lane in the current hybrid branch

## Official WebGL Avatar On This Server

For visual delivery, this machine cannot rely on an interactive browser window because the normal workflow is SSH-only.

What was verified locally on 2026-04-09:

- headless Firefox did not expose a working WebGL path for the official avatar viewer
- `Xvfb` plus non-headless Firefox did expose a working WebGL renderer
- the packaging step had to preserve the sample asset zip structure and use system `zip` when available

Branch helper:

- [`tools/render_hybrid_showcase_video.py`](../tools/render_hybrid_showcase_video.py)

That helper now:

- launches the local preview app
- starts an `Xvfb` display
- drives Firefox through Selenium
- captures rendered official-avatar frames
- encodes a final showcase `mp4`

Verified repo-local output artifact:

- `demo_artifacts/hybrid/hybrid_showcase_female_webgl.mp4`
