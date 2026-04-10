# pantomatrix-galatea

Workspace snapshot for running and extending a PantoMatrix / EMAGE motion-generation pipeline.

This repository contains:

- local helper scripts for EMAGE inference, ARKit export, Viser playback, and Blender smoke testing
- documentation for the EMAGE pipeline and the paused high-quality avatar retarget investigation
- curated source snapshots of the upstream `PantoMatrix` and legacy `EMAGE_2024` code used in this workspace
- small example inputs and generated outputs under `data/`

This repository intentionally does **not** include:

- large pretrained model weights
- downloaded Blender avatar assets
- downloaded evaluation caches
- local debugging logs
- Blender binaries

## Hybrid Branch Focus

The `hybrid/lam-emage-pipeline` branch is scoped around a hybrid output contract:

- keep `EMAGE` as the body baseline
- add `Audio2Face-3D SDK` as the preferred local face-provider path under `--face-provider a2f-3d-sdk`
- retain `LAM_Audio2Expression` as the alternate face provider for A/B comparison
- preserve provider-native face semantics instead of forcing one repository-wide normalization policy

Implementation note:

- the hybrid orchestrator now routes face generation through a provider boundary (`--face-provider`) so `LAM_Audio2Expression` can be replaced later by another face module such as `NVIDIA Audio2Face-3D` without rewriting the body / alignment / manifest pipeline.
- the preferred provider-facing CLI is now the neutral `--face-*` option family (`--face-input-json`, `--face-command`, `--face-python`, `--face-repo`); legacy `--lam-*` flags remain as compatibility aliases.
- each run now also writes `face/provider_metadata.json` so downstream tools can distinguish provider-native payloads from aligned hybrid payloads.

The branch-level execution contract, result folder layout, and environment matrix live in:

- [`docs/hybrid_pipeline_guide.md`](docs/hybrid_pipeline_guide.md)
- [`docs/lam_a2e_integration.md`](docs/lam_a2e_integration.md)
- [`docs/a2f_3d_sdk_integration.md`](docs/a2f_3d_sdk_integration.md)

Server-side showcase note:

- when operating this branch over SSH, the official LAM WebGL avatar is now exported as an `mp4` through [`tools/render_hybrid_showcase_video.py`](tools/render_hybrid_showcase_video.py) using `Xvfb + Firefox`, instead of requiring an interactive browser tab
- current example deliverable: `demo_artifacts/hybrid/hybrid_showcase_female_webgl.mp4`

Important note:

- [`tools/export_arkit_from_npz.py`](tools/export_arkit_from_npz.py) is retained only as a legacy/debug helper that approximates ARKit weights from EMAGE FLAME outputs. It is not the primary hybrid face path.
- Viser-side flashing observed during hybrid validation was traced to player behavior, not hybrid body data. See [`docs/hybrid_pipeline_guide.md`](docs/hybrid_pipeline_guide.md) for the playback debugging note.

## Included Example Data

The `data/` directory includes lightweight example artifacts that were produced locally and are small enough to share:

- input audio clips in [`data/audio/`](data/audio/)
- EMAGE output `.npz` files in [`data/emage_outputs/`](data/emage_outputs/)
- upper-body-only `.npz` variants in [`data/emage_outputs_upper_only/`](data/emage_outputs_upper_only/)
- rendered mesh videos in [`data/rendered_videos/`](data/rendered_videos/)

## External Dependencies To Re-Download

The following resources must be downloaded separately after clone:

- `H-Liu1997/emage_audio`
- `H-Liu1997/emage_evaltools`
- `H-Liu1997/BEAT2_Tools`
- Blender binaries
- optional Blender avatar assets

Helper scripts:

- [`tools/bootstrap_emage_env.sh`](tools/bootstrap_emage_env.sh)
- [`tools/bootstrap_lam_a2e_env.sh`](tools/bootstrap_lam_a2e_env.sh)
- [`tools/bootstrap_a2f3d_sdk_toolchain.sh`](tools/bootstrap_a2f3d_sdk_toolchain.sh)
- [`tools/bootstrap_a2f3d_sdk_env.sh`](tools/bootstrap_a2f3d_sdk_env.sh)
- [`tools/download_emage_assets.py`](tools/download_emage_assets.py)
- [`tools/bootstrap_blender_smplx_render.sh`](tools/bootstrap_blender_smplx_render.sh)
- [`tools/render_emage_smplx_video.py`](tools/render_emage_smplx_video.py)
- [`tools/download_lam_a2e_assets.py`](tools/download_lam_a2e_assets.py)
- [`tools/run_a2f3d_sdk_inference.py`](tools/run_a2f3d_sdk_inference.py)
- [`tools/run_hybrid_provider_comparison.py`](tools/run_hybrid_provider_comparison.py)
- asset redownload reference: [`docs/external-assets-redownload.md`](docs/external-assets-redownload.md)

## Upstream Source Snapshots

This repository contains source snapshots from:

- `PantoMatrix/PantoMatrix`
- a legacy `EMAGE_2024` code path based on the older PantoMatrix tree

The exact snapshot commits and local compatibility notes are recorded in:

- [`docs/upstream-provenance.md`](docs/upstream-provenance.md)

## Key Docs

- [`docs/emage_plan.md`](docs/emage_plan.md)
- [`docs/hybrid_plan.md`](docs/hybrid_plan.md)
- [`docs/blender-smplx-render.md`](docs/blender-smplx-render.md)
- [`docs/hybrid_pipeline_guide.md`](docs/hybrid_pipeline_guide.md)
- [`docs/a2f_3d_sdk_integration.md`](docs/a2f_3d_sdk_integration.md)
- [`docs/retarget-handoff-2026-04-09.md`](docs/retarget-handoff-2026-04-09.md)
