import contextlib
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


LEGACY_PATH = Path(__file__).resolve().parents[1] / "third_party" / "PantoMatrix_legacy" / "scripts" / "EMAGE_2024"


class DummyTxn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def put(self, *_args, **_kwargs):
        return None


class DummyEnv:
    def begin(self, **_kwargs):
        return DummyTxn()


class DummySerialized:
    def to_buffer(self):
        return b"ok"


def _load_beat_testonly():
    if str(LEGACY_PATH) not in sys.path:
        sys.path.insert(0, str(LEGACY_PATH))

    importlib.import_module("dataloaders")
    sys.modules.pop("dataloaders.beat_testonly", None)

    logger_stub = types.SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)

    torch_module = types.ModuleType("torch")
    torch_utils_module = types.ModuleType("torch.utils")
    torch_utils_data_module = types.ModuleType("torch.utils.data")
    torch_distributed_module = types.ModuleType("torch.distributed")

    class Dataset:
        pass

    torch_utils_data_module.Dataset = Dataset
    torch_utils_module.data = torch_utils_data_module
    torch_distributed_module.get_rank = lambda: 0
    torch_module.utils = torch_utils_module
    torch_module.distributed = torch_distributed_module
    torch_module.no_grad = contextlib.nullcontext

    build_vocab_module = types.ModuleType("dataloaders.build_vocab")
    build_vocab_module.Vocab = type("Vocab", (), {})

    data_tools_module = types.ModuleType("dataloaders.data_tools")
    data_tools_module.joints_list = {}

    utils_package = types.ModuleType("dataloaders.utils")
    utils_package.__path__ = []

    audio_features_module = types.ModuleType("dataloaders.utils.audio_features")
    audio_features_module.Wav2Vec2Model = type("Wav2Vec2Model", (), {})

    rotation_module = types.ModuleType("dataloaders.utils.rotation_conversions")
    other_tools_module = types.ModuleType("dataloaders.utils.other_tools")

    stub_modules = {
        "lmdb": types.ModuleType("lmdb"),
        "textgrid": types.ModuleType("textgrid"),
        "pandas": types.ModuleType("pandas"),
        "torch": torch_module,
        "torch.utils": torch_utils_module,
        "torch.utils.data": torch_utils_data_module,
        "torch.distributed": torch_distributed_module,
        "termcolor": types.SimpleNamespace(colored=lambda text, *_args, **_kwargs: text),
        "loguru": types.SimpleNamespace(logger=logger_stub),
        "pyarrow": types.ModuleType("pyarrow"),
        "librosa": types.ModuleType("librosa"),
        "smplx": types.ModuleType("smplx"),
        "dataloaders.build_vocab": build_vocab_module,
        "dataloaders.data_tools": data_tools_module,
        "dataloaders.utils": utils_package,
        "dataloaders.utils.audio_features": audio_features_module,
        "dataloaders.utils.rotation_conversions": rotation_module,
        "dataloaders.utils.other_tools": other_tools_module,
    }

    with mock.patch.dict(sys.modules, stub_modules, clear=False):
        return importlib.import_module("dataloaders.beat_testonly")


class SampleFromClipTests(unittest.TestCase):
    def test_lmdb_serialization_falls_back_when_pyarrow_serialize_is_unavailable(self):
        beat_testonly = _load_beat_testonly()

        sample = [np.array([1, 2, 3], dtype=np.float32), {"word": 7}]
        buffer = beat_testonly._serialize_sample(sample)
        restored = beat_testonly._deserialize_sample(buffer)

        np.testing.assert_array_equal(restored[0], sample[0])
        self.assertEqual(restored[1], sample[1])

    def test_numpy_audio_array_does_not_trigger_list_comparison_value_error(self):
        beat_testonly = _load_beat_testonly()

        dataset = beat_testonly.CustomDataset.__new__(beat_testonly.CustomDataset)
        dataset.args = types.SimpleNamespace(
            audio_fps=16000,
            pose_fps=30,
            audio_rep="onset+amplitude",
            audio_sr=16000,
            multi_length_training=[1.0],
            stride=20,
            facial_rep=None,
            word_rep=None,
            emo_rep=None,
            sem_rep=None,
            id_rep=None,
        )
        dataset.max_length = 0
        dataset.ori_stride = 20
        dataset.ori_length = 64
        dataset.n_out_samples = 0

        pose_each_file = np.ones((60, 10), dtype=np.float32)
        trans_each_file = np.zeros((60, 3), dtype=np.float32)
        shape_each_file = np.zeros((60, 300), dtype=np.float32)
        audio_each_file = np.zeros((60 * 16000 // 30, 2), dtype=np.float32)

        with mock.patch.object(
            beat_testonly,
            "MotionPreprocessor",
            return_value=types.SimpleNamespace(get=lambda: (np.ones((60, 10), dtype=np.float32), "PASS")),
        ), mock.patch.object(
            beat_testonly,
            "pyarrow",
            types.SimpleNamespace(serialize=lambda _value: DummySerialized()),
        ):
            dataset._sample_from_clip(
                DummyEnv(),
                audio_each_file=audio_each_file,
                pose_each_file=pose_each_file,
                trans_each_file=trans_each_file,
                shape_each_file=shape_each_file,
                facial_each_file=[],
                word_each_file=[],
                vid_each_file=[],
                emo_each_file=[],
                sem_each_file=[],
                disable_filtering=True,
                clean_first_seconds=0,
                clean_final_seconds=0,
                is_test=True,
            )


if __name__ == "__main__":
    unittest.main()
