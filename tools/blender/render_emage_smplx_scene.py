import argparse
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Render one EMAGE .npz clip through the SMPL-X Blender addon.")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--addon-root", required=True)
    parser.add_argument("--preview-path", required=True)
    parser.add_argument("--silent-video-path", required=True)
    parser.add_argument("--engine", default="CYCLES")
    parser.add_argument("--device", choices=("AUTO", "GPU", "CPU"), default="AUTO")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args(argv)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablock_collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.actions,
        bpy.data.images,
    ):
        for block in list(datablock_collection):
            if block.users == 0:
                datablock_collection.remove(block)


def enable_smplx_addon(addon_root: Path) -> None:
    addon_root_str = str(addon_root.resolve())
    if addon_root_str not in sys.path:
        sys.path.insert(0, addon_root_str)
    result = bpy.ops.preferences.addon_enable(module="smplx_blender_addon")
    print(f"[addon] enable result: {result}")
    if "smplx_blender_addon" not in bpy.context.preferences.addons:
        raise RuntimeError("Failed to enable smplx_blender_addon")


def configure_cycles_device(requested: str) -> str:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.cycles.preview_samples = 16

    if requested == "CPU":
        scene.cycles.device = "CPU"
        return "CPU"

    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons["cycles"].preferences
    chosen = None
    for compute_type in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            cycles_prefs.compute_device_type = compute_type
            cycles_prefs.get_devices()
            devices = [device for device in cycles_prefs.devices if device.type != "CPU"]
            if devices:
                for device in cycles_prefs.devices:
                    device.use = device.type != "CPU"
                chosen = compute_type
                break
        except Exception:
            continue

    if requested == "GPU" and chosen is None:
        raise RuntimeError("GPU rendering requested but no supported Cycles GPU device was found")

    if chosen is None:
        scene.cycles.device = "CPU"
        return "CPU"

    scene.cycles.device = "GPU"
    return chosen


def configure_render(engine: str, device: str, width: int, height: int) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.exposure = -0.85
    scene.view_settings.gamma = 1.0

    if engine == "CYCLES":
        chosen_device = configure_cycles_device(device)
        print(f"[render] engine=CYCLES device={chosen_device}")
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        print("[render] engine=BLENDER_EEVEE_NEXT")


def import_animation(npz_path: Path) -> tuple[bpy.types.Object, bpy.types.Object]:
    result = bpy.ops.object.smplx_add_animation(
        filepath=str(npz_path),
        anim_format="SMPL-X",
        rest_position="GROUNDED",
        target_framerate=30,
        keyframe_facial_expression_weights=True,
    )
    print(f"[import] animation result: {result}")
    obj = bpy.context.view_layer.objects.active
    if obj is None or obj.type != "MESH" or obj.parent is None:
        raise RuntimeError("SMPL-X animation import did not leave an active mesh object with armature parent")
    return obj, obj.parent


def ensure_material(mesh_obj: bpy.types.Object) -> None:
    if mesh_obj.data.materials:
        material = mesh_obj.data.materials[0]
    else:
        material = bpy.data.materials.new(name="SMPLXStudioMaterial")
        mesh_obj.data.materials.append(material)

    material.use_nodes = True
    material.node_tree.nodes.clear()
    output = material.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)
    bsdf = material.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Base Color"].default_value = (0.41, 0.32, 0.27, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.84
    bsdf.inputs["Subsurface Weight"].default_value = 0.015
    material.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])


def add_floor() -> None:
    bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "StudioFloor"
    material = bpy.data.materials.new(name="StudioFloorMaterial")
    material.use_nodes = True
    bsdf = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.67, 0.68, 0.7, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    plane.data.materials.append(material)
    if bpy.context.scene.render.engine == "CYCLES":
        plane.is_shadow_catcher = True


def add_lights(target: Vector, height: float) -> None:
    lights = [
        ("KeyLight", "AREA", (2.2, -3.0, height + 1.0), 850),
        ("FillLight", "AREA", (-2.6, -1.6, height + 0.6), 180),
        ("RimLight", "AREA", (-1.2, 3.1, height + 1.5), 520),
    ]
    for name, light_type, location, energy in lights:
        light_data = bpy.data.lights.new(name=name, type=light_type)
        light_data.energy = energy
        if light_type == "AREA":
            light_data.shape = "RECTANGLE"
            light_data.size = 3.0
            light_data.size_y = 2.0
        light_obj = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = location
        direction = target - light_obj.location
        light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(target: Vector, height: float) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new("RenderCamera")
    camera = bpy.data.objects.new("RenderCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (3.2, -3.7, height * 0.74 + 0.9)
    camera.rotation_euler = (1.16, 0.0, 0.71)
    constraint = camera.constraints.new(type="TRACK_TO")
    focus = bpy.data.objects.new("CameraFocus", None)
    focus.location = target
    bpy.context.collection.objects.link(focus)
    constraint.target = focus
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    camera.data.lens = 58
    bpy.context.scene.camera = camera
    return camera


def configure_world() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (0.73, 0.74, 0.76, 1.0)
    background.inputs[1].default_value = 0.18


def compute_body_target(mesh_obj: bpy.types.Object) -> tuple[Vector, float]:
    bbox = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
    min_corner = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    max_corner = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
    center = (min_corner + max_corner) * 0.5
    height = max_corner.z - min_corner.z
    target = Vector((center.x, center.y, min_corner.z + height * 0.58))
    return target, height


def render_preview(preview_path: Path, frame: int) -> None:
    scene = bpy.context.scene
    scene.frame_set(frame)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(preview_path)
    bpy.ops.render.render(write_still=True)


def render_animation(silent_video_path: Path) -> None:
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.audio_codec = "NONE"
    scene.render.filepath = str(silent_video_path)
    bpy.ops.render.render(animation=True)


def main() -> int:
    args = parse_args()
    npz_path = Path(args.npz)
    preview_path = Path(args.preview_path)
    silent_video_path = Path(args.silent_video_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    silent_video_path.parent.mkdir(parents=True, exist_ok=True)

    reset_scene()
    enable_smplx_addon(Path(args.addon_root))
    configure_render(args.engine, args.device, args.width, args.height)
    mesh_obj, _armature = import_animation(npz_path)
    ensure_material(mesh_obj)
    configure_world()
    add_floor()
    target, height = compute_body_target(mesh_obj)
    add_lights(target, height)
    add_camera(target, height)

    scene = bpy.context.scene
    preview_frame = scene.frame_start + (scene.frame_end - scene.frame_start) // 2
    print(f"[scene] frames={scene.frame_start}-{scene.frame_end} preview_frame={preview_frame}")
    render_preview(preview_path, preview_frame)
    if args.preview_only:
        print(f"[done] preview={preview_path}")
        return 0
    render_animation(silent_video_path)
    print(f"[done] preview={preview_path}")
    print(f"[done] silent_video={silent_video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
