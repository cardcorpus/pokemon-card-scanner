"""The scanner runs end to end on random weights (what it says is noise)."""
import io
import json
import threading
import urllib.request

import numpy as np
import pytest
from PIL import Image

import tiny

QUADS = [[[100, 120], [700, 120], [700, 958], [100, 958]], [[900, 140], [1480, 160], [1460, 980], [880, 960]]]


@pytest.fixture(scope="module")
def scanner(tmp_path_factory):
    from cardcorpus_scanner import Scanner

    weights = tiny.make(tmp_path_factory.mktemp("weights"))
    s = Scanner(weights, verifier=False)
    # a random detector finds nothing: give it the cards' corners
    s.detector = lambda arr: [(0.9, q) for q in QUADS]
    return s


@pytest.fixture(scope="module")
def photo():
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (1100, 1600, 3), dtype=np.uint8))
    return img


def test_scan(scanner, photo):
    result = scanner.scan(photo, evidence=True)
    assert (result["width"], result["height"]) == (1600, 1100)
    assert len(result["cards"]) == 2
    for card in result["cards"]:
        assert len(card["quad"]) == 4
        assert 1 <= len(card["candidates"]) <= 5
        best = card["candidates"][0]
        assert best["print"] is None or best["print"].startswith("international/test-set/")
        assert -1 <= best["sim"] <= 1
        assert 0 <= card["orient_p"] <= 1
        assert card["evidence"] and all("sim" in f for f in card["evidence"])
        assert "edition_stamp" in card and "ecard_reverse" in card
        assert card["quality"] in ("sure", "printing_unclear", "unsure", "likely_wrong", "back")
        assert card["ecard_reading"] in ("bars", "dot_code", None)


def test_language_filter(scanner, photo):
    # fewer Japanese rows than the 20 retrieved: the rest must not fill in
    result = scanner.scan(photo, families=["japanese"])
    for card in result["cards"]:
        assert card["candidates"]
        assert all(c["family"] == "japanese" for c in card["candidates"])
    feat = scanner.embed([photo.crop((100, 120, 700, 958))])[0]
    assert all(c["family"] == "japanese" for c in scanner.rank(feat, limit=10, families=["japanese"]))


def test_web(scanner, photo):
    from cardcorpus_scanner.web import make_server

    server = make_server(scanner, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "<html" in page.lower()
        info = json.loads(urllib.request.urlopen(base + "/api/info").read())
        assert info["verifier"] is False
        buf = io.BytesIO()
        photo.save(buf, "JPEG")
        req = urllib.request.Request(base + "/api/scan?language=international", data=buf.getvalue(), method="POST")
        # lines of JSON: the boxes, each pick, the first picks, each card
        # double-checked, then the answer
        lines = [json.loads(x) for x in urllib.request.urlopen(req).read().splitlines() if x.strip()]
        steps = [x.get("step") or ("first" if x.get("first") else "answer") for x in lines]
        k = steps.index("first")
        assert steps[0] == "boxes" and set(steps[1:k]) == {"pick"}
        first, answer = lines[k], lines[-1]
        assert len(first["cards"]) == len(answer["cards"]) == 2
        assert steps[k + 1:] == ["card", "card", "answer"]
        assert [x["card"] for x in lines[k + 1:-1]] == answer["cards"]
        bad = urllib.request.Request(base + "/api/scan", data=b"not a photo", method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
    finally:
        server.shutdown()


def test_cut_off_flags():
    from cardcorpus_scanner.scanner import cut_off_flags

    whole = [[[100, 100], [700, 100], [700, 940], [100, 940]], [[800, 100], [1400, 100], [1400, 940], [800, 940]]]
    sliver = [[0, 100], [90, 100], [90, 940], [0, 940]]  # at the left edge, a seventh of a card
    big_at_edge = [[0, 1000], [600, 1000], [600, 1840], [0, 1840]]  # a whole card touching the edge
    assert cut_off_flags(whole + [sliver, big_at_edge], 1600, 1840) == [False, False, True, False]
    assert cut_off_flags([], 1600, 1100) == []


def test_cut_off_left_out(scanner, photo):
    sliver = [[0, 300], [70, 300], [70, 900], [0, 900]]
    plain = scanner.detector
    scanner.detector = lambda arr: [(0.9, q) for q in QUADS] + [(0.5, sliver)]
    try:
        result = scanner.scan(photo)
        assert len(result["cards"]) == 2 and not any(c.get("cut_off") for c in result["cards"])
        assert len(result["cut_off"]) == 1 and len(result["cut_off"][0]["quad"]) == 4
        kept = scanner.scan(photo, rules={"cut_off": False})
        assert len(kept["cards"]) == 3 and kept["cut_off"] == []
    finally:
        scanner.detector = plain


def test_left_out(scanner, photo):
    # a box the detector is unsure of (0.1) and a weak one (0.3), both clear
    # of the cards and the edge, with nothing like them in the index
    unsure = [[1500, 300], [1560, 300], [1560, 380], [1500, 380]]
    weak = [[1500, 600], [1560, 600], [1560, 680], [1500, 680]]
    plain = scanner.detector
    scanner.detector = lambda arr: [(0.9, q) for q in QUADS] + [(0.1, unsure), (0.3, weak)]
    try:
        result = scanner.scan(photo)
        assert len(result["cards"]) == 2
        assert sorted(x["reason"] for x in result["left_out"]) == ["junk", "unsure"]
        everything = scanner.scan(photo, rules={"left_out": False})
        assert len(everything["cards"]) == 4 and everything["left_out"] == []
    finally:
        scanner.detector = plain


def test_first_picks(scanner, photo):
    seen = []
    result = scanner.scan(photo, on_first=seen.append)
    assert len(seen) == 1 and seen[0]["first"]
    first = seen[0]
    assert [c["quad"] for c in first["cards"]] == [c["quad"] for c in result["cards"]]
    assert all(c["candidates"] and "print" in c["candidates"][0] for c in first["cards"])


def test_progress(scanner, photo):
    events = []
    result = scanner.scan(photo, on_progress=events.append)
    steps = [e["step"] for e in events]
    assert steps[0] == "boxes" and steps.count("boxes") == 1
    boxes = {b["box"] for b in events[0]["boxes"]}
    picks = [e for e in events if e["step"] == "pick"]
    assert picks and all(e["box"] in boxes and e["candidates"] for e in picks)
    # each card is sent once, in order, just as the answer has it
    cards = [e for e in events if e["step"] == "card"]
    assert [e["index"] for e in cards] == list(range(len(result["cards"])))
    assert [e["card"] for e in cards] == result["cards"]
    assert steps.index("card") > max(i for i, s in enumerate(steps) if s == "pick")
