"""The scanner still says what examples/ says, with the release's weights.

Needs the weights: CARDCORPUS_WEIGHTS=<folder>, or =hub for the Hugging
Face Hub's copy. Skipped without it."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
WEIGHTS = os.environ.get("CARDCORPUS_WEIGHTS")
pytestmark = pytest.mark.skipif(not WEIGHTS, reason="set CARDCORPUS_WEIGHTS to a weights folder or 'hub'")


@pytest.fixture(scope="module")
def scanner():
    from cardcorpus_scanner import Scanner

    return Scanner.load(None if WEIGHTS == "hub" else WEIGHTS)


@pytest.mark.parametrize("expected", sorted(EXAMPLES.glob("*.json")), ids=lambda p: p.stem)
def test_example(scanner, expected):
    want = json.loads(expected.read_text())
    got = scanner.scan(str(EXAMPLES / want["file"]))
    assert len(got["cards"]) == len(want["cards"])
    for g, w in zip(got["cards"], want["cards"]):
        # boxes within 2 px; sims move by a few thousandths between devices,
        # so only the first pick has to agree
        assert np.abs(np.array(g["quad"]) - np.array(w["quad"])).max() <= 2
        first = lambda c: (c["candidates"][0]["print"], c["candidates"][0]["name"]) if c["candidates"] else None
        assert first(g) == first(w)
