import os
from pathlib import Path

import bpy

repo_root = Path(os.environ.get("PANTOMATRIX_ROOT", Path(__file__).resolve().parents[2]))
output_path = repo_root / "data" / "blender_smoke" / "smoke.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(output_path)
bpy.ops.render.render(write_still=True)
