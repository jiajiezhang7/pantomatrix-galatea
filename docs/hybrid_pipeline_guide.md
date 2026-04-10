# Hybrid Pipeline Guide

Date: 2026-04-09

This document is the concise execution contract for branch `hybrid/lam-emage-pipeline`. It complements [`docs/hybrid_plan.md`](hybrid_plan.md) and the paused retarget handoff in [`docs/retarget-handoff-2026-04-09.md`](retarget-handoff-2026-04-09.md).

## Scope

`R1` targets one hybrid output package built from:

- `EMAGE` body motion as the stable baseline
- a selectable face provider under `--face-provider`

Current provider posture:

- preferred local provider: `a2f-3d-sdk`
- comparison provider: `lam`

`EMAGE -> BEAT_Avatars` automation stays paused on this branch. The hybrid pipeline is intended to produce Viser-verifiable body motion plus Unreal-ready face data without making Unreal project execution a repository blocker.

The face lane is intentionally isolated behind a provider boundary. Today that boundary is backed by `LAM_Audio2Expression`, but the orchestrator should be able to swap in a future provider such as `NVIDIA Audio2Face-3D` without changing the body adapter, alignment layer, manifest schema, or result folder contract.

## Usage Contract

The target branch-level workflow is:

1. Run `EMAGE` body inference from the shared input wav.
2. Run the selected face provider on the same wav to produce face coefficients.
   Current implementations:
   - `a2f-3d-sdk`
   - `lam`
   Prefer the neutral `--face-*` options when wiring a provider into the pipeline; `--lam-*` is only a compatibility alias for the LAM provider.
3. Align body and face outputs to one canonical timeline at `30.0` fps.
4. Verify body playback in Viser and export the face stream in an Unreal-ready JSON shape.

Current local component entry points that remain relevant on this branch:

- [`tools/run_emage_audio_inference.py`](../tools/run_emage_audio_inference.py) for EMAGE body generation
- [`tools/hybrid_pipeline.py`](../tools/hybrid_pipeline.py) for orchestration, provider selection, alignment, and manifest writing
- [`tools/viser_emage_player.py`](../tools/viser_emage_player.py) for body-only playback validation
- [`tools/bootstrap_lam_a2e_env.sh`](../tools/bootstrap_lam_a2e_env.sh) for the machine-tested LAM environment bootstrap
- [`tools/bootstrap_a2f3d_sdk_toolchain.sh`](../tools/bootstrap_a2f3d_sdk_toolchain.sh) for side-by-side CUDA/TensorRT provisioning
- [`tools/bootstrap_a2f3d_sdk_env.sh`](../tools/bootstrap_a2f3d_sdk_env.sh) for the dedicated A2F conda env
- [`tools/download_lam_a2e_assets.py`](../tools/download_lam_a2e_assets.py) for LAM asset and weight download
- [`tools/run_hybrid_provider_comparison.py`](../tools/run_hybrid_provider_comparison.py) for same-audio provider A/B runs

## Viser Playback Debug Note

When validating hybrid body motion in Viser, treat visible "flash" or "missing-frame" artifacts as a player-level symptom first, not automatic evidence that hybrid body data is corrupted.

On this branch we verified the following:

- the hybrid body artifact written to `body/body.npz` preserved the same body-side `poses`, `trans`, and `mocap_frame_rate` contract as the source EMAGE `.npz`
- a prior playback glitch came from the viewer implementation, not from the hybrid alignment layer

Two player-side causes were identified and fixed:

1. The preview instance had been started from a short 120-frame clip and the player auto-looped by default, so the jump from the last frame back to frame 0 looked like an intermittent dropped frame.
2. The Viser player rebuilt the body mesh every frame via `remove()` + `add_mesh_simple()`, which introduced visible flicker. The player now updates mesh vertices in place instead.

Practical debugging rule:

- if a hybrid body preview appears to "jump" or "flash", first compare `body/body.npz` against the source EMAGE `.npz` and inspect the Viser playback mode before changing body-generation or alignment logic

## Result Folder Contract

Each successful hybrid run should materialize one result directory with this structure:

```text
<run_dir>/
  manifest.json
  body/
    body.npz
  face/
    arkit_blendshapes.json
    provider_metadata.json
  metrics/
    run_metrics.json
  logs/
    *.log
```

### `manifest.json`

Required top-level fields:

- `audio_path`
- `target_fps`
- `duration_sec`
- `frame_count`
- `components`
- `sources`
- `alignment_policy`

Required `components` keys:

- `emage_body`
- `face_provider`

Compatibility note:

- `component_aliases.lam_face = "face_provider"` is preserved so older branch-local consumers can migrate without breaking immediately.

Each component entry should record at least:

- `status`
- `fps`
- `frame_count`
- `payload_path`
- `provider_name`
- `raw_payload_path`
- `normalization_policy`
- `quality_notes`

Recommended `status` values:

- `ready`

### `face/arkit_blendshapes.json`

The face payload should be a per-frame list using this record shape:

```json
{
  "frame": 0,
  "time_sec": 0.0,
  "blendshapes": {
    "jawOpen": 0.0
  }
}
```

The top-level file should also include enough metadata to identify source audio, fps, and the blendshape name set used for the run.

On this branch, eye behavior is carried inside the same face payload. The aligned face file should preserve:

- `names`
- `eye_blendshape_names`

Machine-verified LAM eye blendshapes currently include:

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

Provider-specific caveat:

- `a2f-3d-sdk` runs in raw-fidelity mode and may emit omitted or zeroed eye-look coefficients; the repository does not synthesize them.
- `face/provider_metadata.json` is the source of truth for provider-specific normalization and caveats.

## Environment Matrix

This branch deliberately keeps `EMAGE`, `LAM_Audio2Expression`, and `Audio2Face-3D SDK` in separate environments.

| Component | Role in hybrid branch | Current environment expectation | Evidence |
| --- | --- | --- | --- |
| `EMAGE` / `PantoMatrix` | Body baseline | Upstream quick start uses `bash setup.sh` and activates `/content/py39/bin/activate`, so this branch treats the local EMAGE lane as a Python 3.9-style environment. | [PantoMatrix README](https://github.com/PantoMatrix/PantoMatrix) |
| `LAM_Audio2Expression` | Main face path | Official README says the project currently supports Python 3.10 and provides separate CUDA 12.1 / 11.8 install scripts. | [LAM_Audio2Expression README](https://github.com/aigc3d/LAM_Audio2Expression) |
| `Audio2Face-3D SDK` | Preferred local face path | This branch keeps a dedicated `a2f3d-sdk310` env and a side-by-side `CUDA 12.9 + TensorRT 10.13` prefix under `$HOME/.local/nvidia/a2f3d-sdk` so the workstation can keep its system CUDA 13.2. | [Audio2Face-3D SDK](https://github.com/NVIDIA/Audio2Face-3D-SDK) |
Practical consequence:

- do not mix the EMAGE and LAM Python dependencies into one environment
- do not mix the A2F SDK runtime with the system CUDA 13.2 path; use the side-by-side toolchain root instead
- on this workstation, treat the official `cu121` install as an initial step only; the verified runnable state required a follow-up torch upgrade to `cu128` because the Blackwell GPU could not execute kernels with `torch 2.1.2+cu121`

## LAM Integration Note

For the machine-verified LAM setup, real inference commands, and the official-output-to-hybrid-contract mapping, see:

- [`docs/lam_a2e_integration.md`](lam_a2e_integration.md)

For the A2F SDK toolchain strategy, environment bootstrap, and provider config contract, see:

- [`docs/a2f_3d_sdk_integration.md`](a2f_3d_sdk_integration.md)

## Server-Side WebGL Showcase Export

Because this branch is often operated over SSH on a headless server, the practical face-preview deliverable is an `mp4`, not a browser tab.

Machine-verified conclusion on this workstation:

- headless Firefox alone was not sufficient for the official WebGL avatar renderer
- the official WebGL avatar did render successfully when launched under `Xvfb` with a non-headless Firefox session
- the branch helper [`tools/render_hybrid_showcase_video.py`](../tools/render_hybrid_showcase_video.py) now uses that route to capture browser frames and encode a deliverable `mp4`

Current verified example artifact:

- `demo_artifacts/hybrid/hybrid_showcase_female_webgl.mp4`

Practical debugging rule:

- if the browser preview looks unavailable on a remote server, do not assume the official avatar is unsupported there; first verify whether the session is running under a real X display such as `Xvfb`

## Legacy / Debug Path

[`tools/export_arkit_from_npz.py`](../tools/export_arkit_from_npz.py) remains useful for debugging and comparison only.

Why it is no longer the main hybrid face path:

- it starts from EMAGE FLAME outputs instead of native ARKit predictions
- it reconstructs ARKit with an inverse least-squares approximation
- it is helpful for regression checks, but it is not the branch's intended production face source

Use it when you need a quick reference export from an existing EMAGE `.npz`, not when documenting or validating the hybrid face architecture.
