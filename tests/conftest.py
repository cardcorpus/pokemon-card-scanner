import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    # ONNX exports (runtime.py) go to a throwaway cache, not the user's
    import os
    import tempfile

    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="cardcorpus-test-cache-")
