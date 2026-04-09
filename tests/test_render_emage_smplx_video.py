import tempfile
import unittest
from pathlib import Path
from unittest import mock


class RenderEmageSmplxVideoTests(unittest.TestCase):
    def test_normalize_clip_name_strips_known_suffixes(self):
        from tools.render_emage_smplx_video import normalize_clip_name

        self.assertEqual(normalize_clip_name("audio-demo-female_output"), "audio-demo-female")
        self.assertEqual(normalize_clip_name("audio-demo-male_upper_only"), "audio-demo-male")
        self.assertEqual(normalize_clip_name("plain-name"), "plain-name")

    def test_find_matching_audio_uses_normalized_npz_stem(self):
        from tools.render_emage_smplx_video import find_matching_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            npz_path = root / "audio-demo-female_output.npz"
            npz_path.write_bytes(b"npz")
            audio_dir = root / "audio"
            audio_dir.mkdir()
            wav_path = audio_dir / "audio-demo-female.wav"
            wav_path.write_bytes(b"wav")

            self.assertEqual(find_matching_audio(npz_path, audio_dir), wav_path)

    def test_find_matching_audio_raises_for_missing_match(self):
        from tools.render_emage_smplx_video import find_matching_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            npz_path = root / "audio-demo-female_output.npz"
            npz_path.write_bytes(b"npz")
            audio_dir = root / "audio"
            audio_dir.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "No matching audio"):
                find_matching_audio(npz_path, audio_dir)

    def test_build_render_job_uses_expected_output_contract(self):
        from tools.render_emage_smplx_video import build_render_job

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            npz_path = root / "audio-demo-male_output.npz"
            npz_path.write_bytes(b"npz")
            audio_dir = root / "audio"
            audio_dir.mkdir()
            wav_path = audio_dir / "audio-demo-male.wav"
            wav_path.write_bytes(b"wav")
            output_root = root / "renders"
            log_root = root / "logs"

            job = build_render_job(npz_path, audio_dir, output_root, log_root)

            self.assertEqual(job.clip_name, "audio-demo-male")
            self.assertEqual(job.audio_path, wav_path)
            self.assertEqual(job.output_dir, output_root / "audio-demo-male")
            self.assertEqual(job.preview_path, output_root / "audio-demo-male" / "preview.png")
            self.assertEqual(job.video_path, output_root / "audio-demo-male" / "audio-demo-male.mp4")
            self.assertEqual(job.work_dir, output_root / "audio-demo-male" / "work")
            self.assertEqual(job.log_path, log_root / "audio-demo-male.log")

    def test_resolve_binary_prefers_path_lookup(self):
        from tools.render_emage_smplx_video import resolve_binary

        with mock.patch("tools.render_emage_smplx_video.shutil.which", return_value="/usr/bin/ffmpeg"):
            self.assertEqual(resolve_binary("ffmpeg", [Path("/tmp/missing")]), Path("/usr/bin/ffmpeg"))

    def test_resolve_binary_uses_first_existing_fallback(self):
        from tools.render_emage_smplx_video import resolve_binary

        with tempfile.TemporaryDirectory() as tmpdir:
            fallback = Path(tmpdir) / "ffmpeg"
            fallback.write_text("bin")
            with mock.patch("tools.render_emage_smplx_video.shutil.which", return_value=None):
                self.assertEqual(resolve_binary("ffmpeg", [fallback]), fallback)


if __name__ == "__main__":
    unittest.main()
