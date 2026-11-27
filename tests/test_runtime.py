"""On the CPU, the models through ONNX Runtime give PyTorch's answers (runtime.py)."""
import numpy as np
import pytest
from PIL import Image

import tiny

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

QUADS = [[[100, 120], [700, 120], [700, 958], [100, 958]], [[900, 140], [1480, 160], [1460, 980], [880, 960]]]


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    return tiny.make(tmp_path_factory.mktemp("weights"))


def load(weights, runtime, cache):
    from cardcorpus_scanner import Scanner

    s = Scanner(weights, verifier=False, runtime="torch")
    if runtime == "onnx":
        s.use_onnx(folder=cache)
    return s


def test_same_answers(weights, tmp_path):
    rng = np.random.default_rng(3)
    photo = Image.fromarray(rng.integers(0, 255, (1100, 1600, 3), dtype=np.uint8))
    torch_s = load(weights, "torch", tmp_path)
    onnx_s = load(weights, "onnx", tmp_path)
    assert onnx_s.runtime == {"detector": "onnx", "matcher": "onnx", "picker": "onnx", "number-reader": "onnx"}
    assert torch_s.runtime == {}
    # the random detector, run both ways, gives the same raw output
    arr = np.asarray(photo)
    want, got = torch_s.detector(arr), onnx_s.detector(arr)
    assert len(want) == len(got)
    for (s1, q1), (s2, q2) in zip(want, got):
        assert s1 == pytest.approx(s2, abs=1e-3)
        # random weights put corners far off the photo: compare relatively
        assert np.allclose(q1, q2, rtol=1e-4, atol=0.5)
    results = []
    for s in (torch_s, onnx_s):
        s.detector = lambda a: [(0.9, q) for q in QUADS]
        results.append(s.scan(photo, evidence=True))
    a, b = results
    assert len(a["cards"]) == len(b["cards"]) == 2
    for x, y in zip(a["cards"], b["cards"]):
        assert [c["print"] for c in x["candidates"]] == [c["print"] for c in y["candidates"]]
        assert [c["sim"] for c in x["candidates"]] == pytest.approx([c["sim"] for c in y["candidates"]], abs=2e-3)
        assert x["number_read"] == y["number_read"]


def test_cached_exports(weights, tmp_path):
    load(weights, "onnx", tmp_path)
    files = sorted(p.name for p in tmp_path.glob("*.onnx"))
    assert len(files) == 4
    # a second load reuses them
    load(weights, "onnx", tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.onnx")) == files


def test_runtime_choice(weights):
    from cardcorpus_scanner import Scanner

    with pytest.raises(ValueError):
        Scanner(weights, verifier=False, runtime="tensorflow")
