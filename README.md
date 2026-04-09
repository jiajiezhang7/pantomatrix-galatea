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
- [`tools/download_emage_assets.py`](tools/download_emage_assets.py)
- [`tools/bootstrap_blender_smplx_render.sh`](tools/bootstrap_blender_smplx_render.sh)
- [`tools/render_emage_smplx_video.py`](tools/render_emage_smplx_video.py)
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
- [`docs/retarget-handoff-2026-04-09.md`](docs/retarget-handoff-2026-04-09.md)
