# Upstream Provenance

This workspace includes curated source snapshots from upstream projects plus local glue code and compatibility fixes.

## PantoMatrix Snapshot

- Upstream repository: `https://github.com/PantoMatrix/PantoMatrix`
- Snapshot commit: `c7356f35f8e39e469e510ccd1bf37e44adf8ec0e`
- Imported as source snapshot under:
  - `third_party/PantoMatrix/`

Local policy:

- keep upstream source, configs, and scripts
- exclude downloaded assets and caches

Excluded local-only directories:

- `third_party/PantoMatrix/hf_cache/`
- `third_party/PantoMatrix/emage_evaltools/`
- `third_party/PantoMatrix/beat2_tools/`
- `third_party/PantoMatrix/blender_assets/`

## Legacy EMAGE Snapshot

- Upstream historical commit basis: `6ca70b9541285b124da2eeedcd80f7c5a54eb111`
- Imported as source snapshot under:
  - `third_party/PantoMatrix_legacy/`

Local compatibility fix retained:

- `third_party/PantoMatrix_legacy/scripts/EMAGE_2024/dataloaders/beat_testonly.py`
- `third_party/PantoMatrix_legacy/scripts/EMAGE_2024/dataloaders/utils/other_tools_hf.py`
- `third_party/PantoMatrix_legacy/scripts/EMAGE_2024/utils/other_tools.py`
- `third_party/PantoMatrix_legacy/scripts/EMAGE_2024/utils/other_tools_hf.py`

Reason:

- fixed invalid numpy-vs-list emptiness checks
- added pyarrow serialization fallback for newer Python environments
- removed hardcoded `REPLICATE_API_TOKEN` values and require environment injection instead
- these changes were needed to run the legacy sample demo successfully in this workspace

Excluded local-only directories:

- `third_party/PantoMatrix_legacy/EMAGE/`
- `third_party/PantoMatrix_legacy/camenduru_emage/`

## Local Workspace Code

Authored in this workspace:

- `tools/`
- `tests/`
- `docs/`

These files are intended to be the main collaborative layer for this repository.
