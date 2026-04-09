# External Assets To Re-Download

This file records the external assets that are intentionally **not** committed into this repository, plus where to obtain them again later.

## Quick Rule

- Repo keeps: code, docs, helper scripts, small example audio / `.npz` / demo videos.
- Repo does **not** keep: large model weights, Blender binaries, downloaded avatar zips, evaluation caches, and full datasets.

## Recommended Download Methods

### 1. Use the helper script for the core EMAGE assets

From the repository root:

```bash
python tools/download_emage_assets.py
```

If Hugging Face is slow or blocked, use a mirror endpoint:

```bash
python tools/download_emage_assets.py --hf-endpoint https://hf-mirror.com
```

This script downloads the core runtime assets into:

- `third_party/PantoMatrix/hf_cache/emage_audio/`
- `third_party/PantoMatrix/emage_evaltools/`
- `third_party/PantoMatrix/beat2_tools/`

### 2. Download Blender-side assets manually

The Blender addon / avatar / render-script zips are not downloaded by the helper script today. Download them manually and place them under:

- `third_party/PantoMatrix/blender_assets/`

## Asset List

### A. Required for EMAGE inference in this repo

#### 1. EMAGE model weights

- Purpose: main audio-to-body+face inference weights
- Target location:
  - `third_party/PantoMatrix/hf_cache/emage_audio/`
- Official source:
  - `https://huggingface.co/H-Liu1997/emage_audio`
- Preferred way:
  - `python tools/download_emage_assets.py`
- Notes:
  - This is the main checkpoint family used by `tools/run_emage_audio_inference.py`.

#### 2. EMAGE evaluation / rendering helper assets

- Purpose: evaluation utilities, mesh rendering helpers, bundled SMPL-X assets used by the PantoMatrix code path
- Target location:
  - `third_party/PantoMatrix/emage_evaltools/`
- Official source:
  - `https://huggingface.co/H-Liu1997/emage_evaltools`
- Preferred way:
  - `python tools/download_emage_assets.py`
- Important contents:
  - `AESKConv_240_100.bin`
  - `mean_vel_smplxflame_30.npy`
  - `smplx_models/smplx/SMPLX_NEUTRAL_2020.npz`

#### 3. BEAT2 tool files for ARKit conversion

- Purpose: convert EMAGE face output into ARKit-style blendshape weights
- Target location:
  - `third_party/PantoMatrix/beat2_tools/`
- Official source:
  - `https://huggingface.co/datasets/H-Liu1997/BEAT2_Tools`
- Files currently needed by this repo:
  - `mat_final.npy`
  - `ARkit_FLAME.zip`
- Preferred way:
  - `python tools/download_emage_assets.py`
- Notes:
  - These are used by `tools/export_arkit_from_npz.py`.

### B. Optional, for Blender rendering / avatar experiments

These are not required for the current EMAGE inference demo, but they are relevant for Blender-side visualization and the paused high-quality avatar pipeline.

#### 4. SMPL-X Blender addon

- Purpose: import / visualize SMPL-X motion in Blender
- Target location:
  - `third_party/PantoMatrix/blender_assets/smplx_blender_addon_20230921.zip`
- Official source:
  - `https://huggingface.co/datasets/H-Liu1997/BEAT2_Tools/blob/main/smplx_blender_addon_20230921.zip`
- Notes:
  - Upstream PantoMatrix README lists this as the official Blender addon download.

#### 5. BEAT avatars package

- Purpose: official Blender-side avatar assets, including the HumGen-based character package discussed during retarget investigation
- Target location:
  - `third_party/PantoMatrix/blender_assets/BEAT_Avatars.zip`
- Official source:
  - `https://huggingface.co/datasets/H-Liu1997/BEAT2_Tools/blob/main/BEAT_Avatars.zip`
- Notes:
  - This asset is large.
  - `EMAGE .npz -> BEAT_Avatars` automatic retarget is still paused and not solved in this repo.

#### 6. Blender render scripts

- Purpose: official render-side helper scripts from the BEAT2 tools package
- Target location:
  - `third_party/PantoMatrix/blender_assets/rendervideo.zip`
- Official source:
  - `https://huggingface.co/datasets/H-Liu1997/BEAT2_Tools/blob/main/rendervideo.zip`

#### 7. Blender binary itself

- Purpose: local or headless Blender execution
- Target location:
  - usually `third_party/blender/` or `third_party/blender3/` in this workspace
- Official source:
  - `https://www.blender.org/download/`
  - release archive: `https://www.blender.org/download/releases/`
- Notes:
  - We intentionally do not version Blender binaries in git.

### C. Optional, for dataset-level work or retraining

These are not needed for the current demo inference flow, but may be needed by teammates who want to reproduce training or dataset preprocessing.

#### 8. BEAT2 dataset

- Purpose: SMPL-X + FLAME dataset used by the newer PantoMatrix / EMAGE code paths
- Official source:
  - `https://huggingface.co/datasets/H-Liu1997/BEAT2`
- Notes:
  - Large dataset, not needed for the current inference-only demo.

#### 9. BEAT dataset

- Purpose: older BVH + ARKit dataset used by legacy pipelines
- Official source:
  - `https://huggingface.co/datasets/H-Liu1997/BEAT`

#### 10. Rendered BEAT videos

- Purpose: rendered reference videos for qualitative inspection
- Official source:
  - `https://huggingface.co/datasets/H-Liu1997/BEAT_Rendered_Videos`

## Licensing / Access Notes

### SMPL-X licensing note

- `SMPLX_NEUTRAL_2020.npz` is currently mirrored inside `H-Liu1997/emage_evaltools`, which is the practical source used by this repo.
- The official SMPL-X license page is:
  - `https://smpl-x.is.tue.mpg.de/modellicense.html`
- If your team needs a stricter compliance route, verify whether you want to rely on the Hugging Face mirror or separately obtain SMPL-X through its official license workflow.

## What The Repo Expects To Exist

After a full re-download, the important paths should look like:

```text
third_party/PantoMatrix/
  hf_cache/emage_audio/
  emage_evaltools/
    smplx_models/smplx/SMPLX_NEUTRAL_2020.npz
  beat2_tools/
    mat_final.npy
    ARkit_FLAME.zip
  blender_assets/
    smplx_blender_addon_20230921.zip
    BEAT_Avatars.zip
    rendervideo.zip
```

## Related Files In This Repo

- `tools/download_emage_assets.py`
- `tools/run_emage_audio_inference.py`
- `tools/export_arkit_from_npz.py`
- `docs/retarget-handoff-2026-04-09.md`
- `docs/upstream-provenance.md`
