from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
DATA_ROOT = WORKSPACE_ROOT / "data"
TOOLS_ROOT = WORKSPACE_ROOT / "tools"
EMAGE_OUTPUTS_ROOT = DATA_ROOT / "emage_outputs"
EMAGE_SAMPLE_OUTPUTS = tuple(sorted(EMAGE_OUTPUTS_ROOT.glob("*_output.npz")))


def load_tool_module(module_name: str, relative_tool_path: str):
    tool_path = TOOLS_ROOT / relative_tool_path
    spec = importlib.util.spec_from_file_location(module_name, tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {tool_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
