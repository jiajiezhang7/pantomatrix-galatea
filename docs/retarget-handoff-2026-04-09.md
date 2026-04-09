# Retarget Handoff Notes

Date: 2026-04-09

## Current Status

The `EMAGE -> Blender` path has been validated for:

- headless Blender installation
- headless still-image rendering
- official `smplx_blender_addon` asset download
- official `BEAT_Avatars.zip` asset download
- official `rendervideo.zip` asset download

The `EMAGE .npz -> official SMPL-X Blender addon` link is supported by the addon itself.

The `EMAGE .npz -> BEAT_Avatars (HumGen) high-quality avatar` link is NOT currently automated and is now paused for manual follow-up work.

## What Was Verified

### 1. Blender works headless on this server

- Installed Blender 4.2.2 locally under:
  - `third_party/blender/blender-4.2.2-linux-x64/blender`
- Headless smoke render succeeded and produced:
  - `data/blender_smoke/smoke.png`

### 2. Official Blender-side assets were downloaded

- `smplx_blender_addon_20230921.zip`
- `BEAT_Avatars.zip`
- `rendervideo.zip`

Location:
- `third_party/PantoMatrix/blender_assets/`

### 3. Official SMPL-X addon directly supports SMPL-X `.npz`

The addon's `SMPLXAddAnimation` operator loads `.npz` files and reads:

- `poses`
- `trans`
- `betas`
- `expressions`
- `mocap_frame_rate`

This matches the EMAGE `.npz` output format.

Relevant local file:
- `third_party/PantoMatrix/blender_assets/unpacked/smplx_addon/smplx_blender_addon/__init__.py`

### 4. `BEAT_Avatars.blend` is not plug-and-play with the SMPL-X addon

Findings:

- Blender 4.2.2 cannot open `BEAT_Avatars.blend` correctly.
- Blender 3.0.1 can open it.
- The scene contains multiple `HG_*` HumanGenerator-style armatures.
- The main `HG_*` rigs share a consistent 105-bone structure.
- That 105-bone structure is NOT identical to the official 55-joint SMPL-X skeleton used by EMAGE.

Representative checks were done on:

- `HG_Cassandra`
- `HG_Diane`
- `HG_Mario`

The primary mismatch areas are:

- root / pelvis / hips
- arm and hand chain structure
- face rig controls

## Why Retarget Is Paused

There is no verified official public automation that performs:

`EMAGE .npz -> BEAT_Avatars (HumGen) high-quality avatar`

The available official pieces are:

- EMAGE `.npz`
- SMPL-X Blender addon
- BEAT avatar `.blend` scene
- sample Blender render project

But no confirmed public end-to-end retarget script was found that bridges them.

Because of this, continuing would require manual retarget design and visual validation rather than safe autonomous execution.

## Bone / Rig Findings

### SMPL-X rig used by EMAGE

Properties:

- 55 joints
- explicit `pelvis`, `left_hip`, `right_hip`
- explicit `jaw`, `left_eye_smplhf`, `right_eye_smplhf`
- direct finger chains from wrist

### HumGen / BEAT avatar rig

Properties:

- 105 bones on the main `HG_*` rigs
- root begins at `spine`
- legs branch from `spine` via `thigh.L` / `thigh.R`
- hands include intermediate `palm.*` bones
- face uses a large set of rig controls plus shape keys

This means the rigs are similar enough for a manual PoC, but not close enough for safe automatic retarget.

## Practical Consequence

The following pipeline is feasible now:

`EMAGE .npz -> SMPL-X addon -> Blender SMPL-X character -> render`

The following pipeline is NOT yet solved:

`EMAGE .npz -> HG/BEAT_Avatars high-quality character -> render`

## Recommended Next Owner

This should be handed to a teammate who is comfortable with:

- Blender rigging
- retargeting workflows
- HumanGenerator / HumGen rigs
- manual visual QA

## Recommended Next Step For That Teammate

1. Pick one avatar only:
   - `HG_Cassandra`
2. Build a body-only retarget PoC first:
   - spine
   - neck/head
   - shoulders/arms
   - legs
3. Ignore face retarget initially.
4. Validate motion quality visually.
5. Only after body works, investigate:
   - fingers
   - jaw / eyes
   - face shape key mapping

## Candidate Tools For Manual PoC

Most practical first choice:

- `Keemap-Blender-Rig-ReTargeting-Addon`

Other possible tools:

- `ReNim`
- `Rokoko Studio Live for Blender`
- `retarget-bvh` / MakeWalk style workflows

These are not yet integrated in this workspace.

## Important Local Paths

### Inputs

- `data/emage_outputs/audio-demo-female_output.npz`
- `data/emage_outputs/audio-demo-male_output.npz`

### Blender binaries

- `third_party/blender/blender-4.2.2-linux-x64/blender`
- `third_party/blender3/blender-3.0.1-linux-x64/blender`

### Downloaded assets

- `third_party/PantoMatrix/blender_assets/smplx_blender_addon_20230921.zip`
- `third_party/PantoMatrix/blender_assets/BEAT_Avatars.zip`
- `third_party/PantoMatrix/blender_assets/rendervideo.zip`

### Logs / evidence

- `data/blender_smoke/smoke.log`
- `demo_logs/blender_301_beat_avatars.log`
- `demo_logs/blender_301_beat_objects.log`
- `demo_logs/blender_data2video_text.log`

## Pause Condition

Retarget work is intentionally paused here.

Do not resume autonomous implementation unless one of the following happens:

- a teammate explicitly accepts manual retarget ownership
- a proven public `SMPL-X -> HumGen` workflow is found
- the plan is changed to use Blender SMPL-X rendering instead of BEAT avatars
