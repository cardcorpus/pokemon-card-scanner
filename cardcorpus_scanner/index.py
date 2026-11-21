"""The reference index: what the scanner compares a card's fingerprint with.

Files, in the weights folder's index/:

  embeddings.npy        rows x 512 float16, unit length: one fingerprint per
                        reference picture
  rows.json             row -> the picture's handle
  cards.jsonl.gz        one line per handle: the card it shows (set, card,
                        variant as the CardCorpus dataset addresses them) and
                        what the scanner reads about it
  dupe-groups.json      handle -> its group of near-identical pictures
  picker-embeddings.npy the print picker's gallery, rows x 512 float16
  picker-rows.json      gallery row -> {"print", "group"}
  verifier-pictures.json
                        print -> the picture the print verifier compares it
                        with, where that isn't the card's own
  release.json          which dataset version and models it was built from,
                        and its parts

The files other than release.json come in parts: release.json's `parts`
lists each part's folder (relative to index/), the catalogs its rows are
in and its row count, and the index is its parts joined in that order. A
release.json without `parts` is one part, index/ itself. Load only some
catalogs' parts with Index(folder, catalogs=[...]). Handles, prints,
releases and artwork and picture groups are the same across parts, so a
group can span them.

A handle is the picture's id on the image mirror (32 hex digits, as the
dataset gives it) for a picture the dataset lists; a second view of a card
that the dataset doesn't list has a handle of its own ("view-…").

A "print" is how the scanner names a card: "<set>/<card>", for example
"english/base-set/4" is https://pokemon.cardcorpus.com/v1/sets/english/base-set.json,
card "4".
"""
import gzip
import json
from pathlib import Path

import numpy as np


def print_id(row):
    """A card row's print id, or None."""
    return f"{row['set']}/{row['card']}" if row.get("set") and row.get("card") else None


def parts(release, catalogs=None):
    """The parts release.json lists, those holding one of `catalogs` (all
    when None); one part at "." where it lists none."""
    listed = release.get("parts") or [{"path": "."}]
    if not catalogs:
        return listed
    wanted = set(catalogs)
    return [p for p in listed if "catalogs" not in p or wanted & set(p["catalogs"])]


class Index:
    def __init__(self, folder, catalogs=None):
        folder = Path(folder)
        self.folder = folder
        self.release = json.loads((folder / "release.json").read_text())
        self.handles, embeddings = [], []
        # handle -> card row, in the order the build wrote them (the first
        # card of each picture group with a print stands in for its
        # print-less second views, so the order matters)
        self.cards, self.groups, self.verifier_pictures = {}, {}, {}
        picker_embeddings, self.picker_rows = [], []
        for part in parts(self.release, catalogs):
            at = folder / part["path"]
            handles = json.loads((at / "rows.json").read_text())
            vectors = np.load(at / "embeddings.npy").astype(np.float32)
            if len(handles) != len(vectors):
                raise ValueError(f"{at}: {len(handles)} rows but {len(vectors)} fingerprints")
            self.handles += handles
            embeddings.append(vectors)
            with gzip.open(at / "cards.jsonl.gz", "rt") as f:
                for line in f:
                    row = json.loads(line)
                    row["print"] = row.get("print") or print_id(row)
                    # an index built before the word was `catalog` calls it `family`
                    if "catalog" not in row and "family" in row:
                        row["catalog"] = row.pop("family")
                    self.cards[row["handle"]] = row
            for name, into in (("dupe-groups.json", self.groups), ("verifier-pictures.json", self.verifier_pictures)):
                if (at / name).exists():
                    into.update(json.loads((at / name).read_text()))
            if (at / "picker-embeddings.npy").exists():
                picker_embeddings.append(np.load(at / "picker-embeddings.npy").astype(np.float32))
                self.picker_rows += json.loads((at / "picker-rows.json").read_text())
        self.embeddings = np.concatenate(embeddings) if embeddings else np.zeros((0, 512), np.float32)
        self.picker_embeddings = np.concatenate(picker_embeddings) if picker_embeddings else None
        if self.picker_embeddings is None:
            self.picker_rows = None
