from __future__ import annotations

from conftest import load_tool_module


def test_compute_next_frame_wraps_only_when_loop_enabled():
    player = load_tool_module("viser_emage_player_test", "viser_emage_player.py")

    assert player.compute_next_frame(current_frame=3, frame_count=4, loop=True) == 0
    assert player.compute_next_frame(current_frame=3, frame_count=4, loop=False) == 3
    assert player.compute_next_frame(current_frame=1, frame_count=4, loop=False) == 2


def test_upsert_body_mesh_updates_vertices_in_place():
    player = load_tool_module("viser_emage_player_test", "viser_emage_player.py")

    class FakeHandle:
        def __init__(self):
            self.vertices = None
            self.faces = None
            self.remove_called = False

        def remove(self):
            self.remove_called = True

    class FakeScene:
        def __init__(self):
            self.calls = []
            self.handle = FakeHandle()

        def add_mesh_simple(self, name, vertices, faces, color):
            self.calls.append((name, vertices.copy(), faces.copy(), color))
            self.handle.vertices = vertices.copy()
            self.handle.faces = faces.copy()
            return self.handle

    scene = FakeScene()
    mesh_holder = {"handle": None}
    vertices = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    ]
    faces = [[0, 1, 1]]

    player.upsert_body_mesh(scene=scene, mesh_holder=mesh_holder, vertices=vertices, faces=faces, frame_idx=0)
    first_handle = mesh_holder["handle"]
    assert len(scene.calls) == 1

    player.upsert_body_mesh(scene=scene, mesh_holder=mesh_holder, vertices=vertices, faces=faces, frame_idx=1)
    assert mesh_holder["handle"] is first_handle
    assert len(scene.calls) == 1
    assert first_handle.remove_called is False
    assert first_handle.vertices == vertices[1]
