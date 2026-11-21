"""An index in parts loads as the whole of it, and a reader can load only
some catalogs' parts (SPEC.md, "Index")."""
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import tiny


def split(whole, out, groups):
    """Copy a weights folder with its index cut into parts: groups is
    [(path, [row positions])], each part holding those rows in order."""
    out = Path(out)
    shutil.copytree(whole, out)
    index = out / "index"
    handles = json.loads((index / "rows.json").read_text())
    vectors = np.load(index / "embeddings.npy")
    with gzip.open(index / "cards.jsonl.gz", "rt") as f:
        cards = {r["handle"]: r for r in map(json.loads, f)}
    dupes = json.loads((index / "dupe-groups.json").read_text())
    picker = np.load(index / "picker-embeddings.npy")
    picker_rows = json.loads((index / "picker-rows.json").read_text())
    for name in ("rows.json", "embeddings.npy", "cards.jsonl.gz", "dupe-groups.json", "picker-embeddings.npy",
                 "picker-rows.json"):
        (index / name).unlink()
    parts = []
    for n, (path, at) in enumerate(groups):
        folder = index / path
        folder.mkdir()
        mine = [handles[i] for i in at]
        (folder / "rows.json").write_text(json.dumps(mine))
        np.save(folder / "embeddings.npy", vectors[at])
        with gzip.open(folder / "cards.jsonl.gz", "wt") as f:
            for h in mine:
                f.write(json.dumps(cards[h]) + "\n")
        (folder / "dupe-groups.json").write_text(json.dumps({h: g for h, g in dupes.items() if h in mine}))
        if n == 0:  # the picker's gallery in the first part only
            np.save(folder / "picker-embeddings.npy", picker)
            (folder / "picker-rows.json").write_text(json.dumps(picker_rows))
        parts.append({"path": path, "catalogs": sorted({cards[h]["catalog"] for h in mine}), "rows": len(mine)})
    release = json.loads((index / "release.json").read_text())
    (index / "release.json").write_text(json.dumps({**release, "parts": parts}))
    return out


@pytest.fixture(scope="module")
def whole(tmp_path_factory):
    return tiny.make(tmp_path_factory.mktemp("weights"))


def scanner(weights, **kw):
    from cardcorpus_scanner import Scanner

    return Scanner(weights, verifier=False, number_reader=False, **kw)


def test_parts_load_as_the_whole(whole, tmp_path):
    from cardcorpus_scanner.index import Index

    n = len(json.loads((whole / "index" / "rows.json").read_text()))
    parted = split(whole, tmp_path / "parted", [("a", list(range(0, 10))), ("b", list(range(10, n)))])
    one, two = Index(whole / "index"), Index(parted / "index")
    assert one.handles == two.handles
    assert np.array_equal(one.embeddings, two.embeddings)
    assert one.cards == two.cards and one.groups == two.groups
    assert np.array_equal(one.picker_embeddings, two.picker_embeddings) and one.picker_rows == two.picker_rows
    s1, s2 = scanner(whole), scanner(parted)
    rng = np.random.default_rng(3)
    for _ in range(5):
        feat = rng.standard_normal(512).astype(np.float32)
        feat = s1.torch.from_numpy(feat / np.linalg.norm(feat))
        assert s1.rank(feat, limit=10) == s2.rank(feat, limit=10)


def test_a_reader_loads_only_the_catalogs_it_wants(whole, tmp_path):
    # each row in the part of the catalog the scanner gives it: a second
    # view's is the card's it stands for (tiny's view-1 stands for card 0)
    full = scanner(whole)
    catalog = list(full.row_catalog)
    by = {c: [i for i, x in enumerate(catalog) if x == c] for c in ("international", "japanese")}
    parted = split(whole, tmp_path / "parted", [("international", by["international"]), ("japanese", by["japanese"])])
    s = scanner(parted, catalogs=["japanese"])
    assert len(s.index.handles) == len(by["japanese"])
    assert set(s.row_catalog) == {"japanese"}
    rng = np.random.default_rng(4)
    feat = rng.standard_normal(512).astype(np.float32)
    feat = full.torch.from_numpy(feat / np.linalg.norm(feat))
    assert s.rank(feat, limit=5) == full.rank(feat, limit=5, catalogs=["japanese"])


def test_an_older_index_reads_family_as_catalog(whole, tmp_path):
    from cardcorpus_scanner.index import Index

    old = Path(shutil.copytree(whole, tmp_path / "old"))
    path = old / "index" / "cards.jsonl.gz"
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f]
    with gzip.open(path, "wt") as f:
        for r in rows:
            r["family"] = r.pop("catalog")
            f.write(json.dumps(r) + "\n")
    assert [r.get("catalog") for r in Index(old / "index").cards.values()] == [r["family"] for r in rows]
