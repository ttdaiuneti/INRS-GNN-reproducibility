"""Re-export Planetoid loaders from shared (avoids duplicating graph_data.py)."""
import importlib.util
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[1] / "shared" / "graph_data.py"
_spec = importlib.util.spec_from_file_location("_shared_graph_data", _SHARED)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("_")})
