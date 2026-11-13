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
  release.json          which dataset version and models it was built from

A handle is the picture's id on the image mirror (32 hex digits, as the
dataset gives it) for a picture the dataset lists; a second view of a card
that the dataset doesn't list has a handle of its own ("view-…").

A "print" is how the scanner names a card: "<set>/<card>", for example
"international/base-set/4" is https://cardcorpus.com/v1/sets/international/base-set.json,
card "4".
"""
import gzip
import json
from pathlib import Path

import numpy as np


def print_id(row):
    """A card row's print id, or None."""
    return f"{row['set']}/{row['card']}" if row.get("set") and row.get("card") else None


class Index:
    def __init__(self, folder):
        folder = Path(folder)
        self.folder = folder
        self.release = json.loads((folder / "release.json").read_text())
        self.handles = json.loads((folder / "rows.json").read_text())
        self.embeddings = np.load(folder / "embeddings.npy").astype(np.float32)
        if len(self.handles) != len(self.embeddings):
            raise ValueError(f"{folder}: {len(self.handles)} rows but {len(self.embeddings)} fingerprints")
        # handle -> card row, in the order the build wrote them (the first
        # card of each picture group with a print stands in for its
        # print-less second views, so the order matters)
        self.cards = {}
        with gzip.open(folder / "cards.jsonl.gz", "rt") as f:
            for line in f:
                row = json.loads(line)
                row["print"] = row.get("print") or print_id(row)
                self.cards[row["handle"]] = row
        groups = folder / "dupe-groups.json"
        self.groups = json.loads(groups.read_text()) if groups.exists() else {}
        vp = folder / "verifier-pictures.json"
        self.verifier_pictures = json.loads(vp.read_text()) if vp.exists() else {}
        self.picker_embeddings = self.picker_rows = None
        if (folder / "picker-embeddings.npy").exists():
            self.picker_embeddings = np.load(folder / "picker-embeddings.npy").astype(np.float32)
            self.picker_rows = json.loads((folder / "picker-rows.json").read_text())
