"""The corner reader: the set code and language mark printed at a card's foot ("MEW EN", "PAF DE", "M6a F",
"151C"), read by a small CRNN of the number reader's shape, for the language reader (language.py). It reads 90 grey
strips (3 places x 3 heights x 5 offsets x 2 widths) and keeps the text most strips agree on.

A read settles a language only when it carries a language mark (or is a Simplified Chinese code ending in C) and it is
exactly the code printed on the prints in play of one language; where no print in play has a code on record, the mark
alone names the language, unless that language's prints in play have another code on record. A read
without its mark never settles anything: the mark is the smallest thing on the card and is the first to be lost, and
"M4" is then both the Japanese print and a Thai one with its "T" missing. "PROMO" is no set's code. Japanese and
Korean print no mark, so the corner never tells them apart; the classifier and the OCR do.

The model is its own data part (scripts/scanner/corner-reader): weights as safetensors, and corner-reader.json with
the charset, where to cut the strips, `codes` (each set's printed code, the dataset's `printedCode`) and `marks` and
`joined_marks` (each language's mark, the dataset's `codeMark`), written from the dataset by export.py. A
corner-reader.json written before the dataset had them has no marks, and MARKS stands in.
"""
import json
from collections import Counter
from pathlib import Path

from .number_reader import Reader

# The marks a corner-reader.json without its own stands in with (the dataset's `codeMark`s before Latin American
# Spanish's "LA"); a mark after a space, and Simplified Chinese's "C" joined to the code.
MARKS = {"EN": "en", "FR": "fr", "DE": "de", "IT": "it", "ES": "es", "PT": "pt", "F": "zh-Hant", "T": "th", "I": "id"}
JOINED_MARKS = {"C": "zh-Hans"}
IGNORED = {"PROMO", "PR0M0"}
CATALOG_ALIASES = {"international": "english"}


def joined_mark(text, joined=None):
    """The joined mark a one-word read ends in, or None. zh-Hans codes start with C (CS3.5C, CBB3C, CMPC) or hold a
    digit (151C, 30THC); "PMC" and "GMC" are misreads."""
    for mark in (JOINED_MARKS if joined is None else joined):
        if (" " not in text and len(text) >= len(mark) + 2 and text.endswith(mark)
                and (text.startswith(mark) or any(ch.isdigit() for ch in text))):
            return mark
    return None


def marked(text, marks=None, joined=None):
    """Whether a read carries what tells languages apart: a language mark after the code, or a mark joined to it
    (zh-Hans's closing C)."""
    parts = text.split(" ")
    return (len(parts) == 2 and parts[1] in (MARKS if marks is None else marks)) or joined_mark(text, joined) is not None


def mark_language(text, marks=None, joined=None):
    """The language a marked read's mark names."""
    parts = text.split(" ")
    if len(parts) == 2:
        return (MARKS if marks is None else marks).get(parts[1])
    mark = joined_mark(text, joined)
    return (JOINED_MARKS if joined is None else joined).get(mark) if mark else None


def set_key(c):
    s, catalog = c.get("set") or "", CATALOG_ALIASES.get(c.get("catalog") or "", c.get("catalog") or "")
    if "/" in s:
        head, rest = s.split("/", 1)
        return f"{CATALOG_ALIASES.get(head, head)}/{rest}"
    return f"{catalog}/{s}"


class CornerReader:
    """weights: a safetensors file (or a state dict); settings: corner-reader.json (a path or its contents)."""

    def __init__(self, weights, settings, device="cpu"):
        s = settings if isinstance(settings, dict) else json.loads(Path(settings).read_text())
        self.s = s
        self.reader = Reader(weights, {"charset": s["charset"], "strip": s["strip"], "slots": s["slots"],
                                       "denominators": [1]}, [], device)
        self.codes = s["codes"]
        self.marks = s.get("marks") or MARKS
        self.joined_marks = s.get("joined_marks") or JOINED_MARKS
        self.min_votes = s.get("min_votes", 5)

    def strips(self, crop):
        """The strips of a straightened card (PIL), as (90, 1, 48, 384) tensors."""
        W, H = self.s["card"]
        img = crop.convert("L").resize((W, H))
        out = []
        for slot in self.s["slots"]:
            for dy in self.s["dy"]:
                for dx in self.s["dx"]:
                    for width in self.s["widths"]:
                        bw, bh = width * W, slot["h"] * H * self.s["height"]
                        cx, cy = (slot["cx"] + dx) * W, (slot["cy"] + dy) * H
                        out.append(self.reader._to_tensor(img.crop((int(cx - bw / 2), int(cy - bh / 2),
                                                                    int(cx + bw / 2), int(cy + bh / 2)))))
        return out

    def read(self, crop):
        """{"text", "votes"}: the text most strips agree on (two characters or more, with a letter, no "/")."""
        torch = self.reader.torch
        variants = self.strips(crop)
        with torch.no_grad():
            logits = self.reader.model(torch.stack(variants).to(self.reader.device)).float().cpu()
        votes = Counter()
        for j in range(logits.shape[0]):
            text = self.reader._decode(logits[j])[0].strip()
            if len(text) >= 2 and any(ch.isalpha() for ch in text) and "/" not in text:
                votes[text] += 1
        if not votes:
            return {"text": "", "votes": 0}
        text, n = votes.most_common(1)[0]
        return {"text": text, "votes": n}

    def settle(self, read, prints, languages):
        """(language or None, why) from a read and the prints in play (candidates with "set", "catalog",
        "language")."""
        text = read["text"]
        if read["votes"] < self.min_votes:
            return None, "corner: unsure"
        if text in IGNORED:
            return None, "corner: promo"
        marks, joined = getattr(self, "marks", MARKS), getattr(self, "joined_marks", JOINED_MARKS)
        if not marked(text, marks, joined):
            return None, f"corner: {text} has no mark"
        found = {c.get("language") for c in prints
                 if c.get("language") in languages and self.codes.get(set_key(c)) == text}
        if len(found) == 1:
            return found.pop(), f"corner {text}"
        if found:
            return None, f"corner: {text} matches {len(found)} languages"
        # no print in play has this code on record (the dataset hasn't settled the set's printed code yet): the mark
        # alone names the language, but only when none of that language's prints in play has another code on record
        language = mark_language(text, marks, joined)
        own = [c for c in prints if c.get("language") == language]
        if language in languages and own and all(set_key(c) not in self.codes for c in own):
            return language, f"corner mark {text}"
        return None, f"corner: {text} matches no print in play"

    def language(self, crop, prints, languages):
        read = self.read(crop)
        found, why = self.settle(read, prints, languages)
        return found, why, read
