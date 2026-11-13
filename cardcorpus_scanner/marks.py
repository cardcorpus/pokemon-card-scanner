"""Stages 8 and 9: two marks that tell copies of one print apart.

Neither the matcher nor the print verifier can tell two copies that share
one picture apart. These two checks look for what differs instead. Each
returns a raw score; which copy that makes a card depends on what copies
the card has, which the dataset says (the variants in its set file).

**1st Edition stamp** (`edition_stamp`). Wizards of the Coast's English sets
(Base Set to Neo Destiny) put the stamp in one of three places: below the
picture on the left on a Pokémon, by the bottom left corner on a Trainer,
top right on an Energy. This looks for it there by normalised
cross-correlation with a template of the stamp (edition-stamp.png, the mean
of 137 stamps, 48 x 36 at a card width of 600) under a mask of its dark
strokes. A score of STAMP_THRESHOLD (0.60) or more is a stamp.

**e-Card reverse holo** (`ecard_reverse`). Expedition, Aquapolis and Skyridge
print the e-Reader dot code in two strips, along the bottom and down the
left edge. A reverse holo has no dot code there: its strips are holo, with
evenly spaced bars across them. So do the holo rares and the special
Energy, on every copy, so bars mean "no dot code", not "reverse holo"; the
check applies to a card whose standard copy keeps the dot code. `reading`
turns the score into "bars", "dot_code" or None (unsure).
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FRAME_W, FRAME_H = 600, 825

# --- 1st Edition stamp --------------------------------------------------------

# Where a stamp can sit: x0, x1, y0, y1 in the 600 x 825 frame, and the
# template's sizes there (a Trainer's and an Energy's stamps are smaller).
REGIONS = (
    ("pokemon", 0, 110, 355, 510, (0.9, 1.0, 1.1)),
    ("trainer", 60, 180, 700, 800, (0.6, 0.7, 0.8, 0.9)),
    ("energy", 450, 595, 15, 125, (0.6, 0.7, 0.8, 0.9, 1.0)),
)
MASK_DARK = 110  # template pixels darker than this, grown, are the stamp
MASK_GROW = 5
STAMP_THRESHOLD = 0.60

_TPL = None


def _template():
    import cv2

    global _TPL
    if _TPL is None:
        tpl = cv2.imread(str(HERE / "edition-stamp.png"), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            raise FileNotFoundError(HERE / "edition-stamp.png")
        tpl = tpl.astype(np.float32)
        dark = (tpl < MASK_DARK).astype(np.uint8)
        mask = cv2.dilate(dark, np.ones((MASK_GROW, MASK_GROW), np.uint8)).astype(np.float32)
        _TPL = (tpl, mask)
    return _TPL


def edition_stamp(card_rgb):
    """{"score", "region", "x", "y", "scale"} for a straightened card, RGB
    uint8 (any size; the scanner passes its 960 x 1340 crop)."""
    import cv2

    tpl, mask = _template()
    grey = cv2.cvtColor(np.ascontiguousarray(card_rgb), cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(grey, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    best = {"score": -1.0, "region": None, "x": 0, "y": 0, "scale": 1.0}
    for name, x0, x1, y0, y1, scales in REGIONS:
        region = frame[y0:y1, x0:x1]
        for s in scales:
            w, h = int(round(tpl.shape[1] * s)), int(round(tpl.shape[0] * s))
            t = cv2.resize(tpl, (w, h), interpolation=cv2.INTER_AREA)
            m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            res = cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED, mask=m)
            res[~np.isfinite(res)] = -1
            _, v, _, loc = cv2.minMaxLoc(res)
            if v > best["score"]:
                best = {"score": round(float(v), 4), "region": name, "x": loc[0] + x0, "y": loc[1] + y0, "scale": s}
    return best


# --- e-Card reverse holo --------------------------------------------------------

# Bottom strip: x0, x1 along it, the y window it slides in, band height.
BOTTOM = (260, 540, 735, 825, 30)
# Left strip: y0, y1 along it, the x window it slides in, band width.
LEFT = (80, 720, 0, 75, 30)
PERIODS = (12, 28)  # the bars' spacing, px at a card width of 600
SLIDE_STEP = 3
# A card whose own (standard) picture scores under APPLIES keeps the dot
# code, so the check applies to it. A photo of it showing bars (bottom strip
# BARS_BOTTOM or more and left BARS_LEFT or more) is its reverse holo; a mean
# of STANDARD or less shows the dot code; between, it can't say.
APPLIES = 0.65
BARS_BOTTOM = 0.75
BARS_LEFT = 0.72
STANDARD = 0.60


def _strength(band, axis):
    """Bars' share of a band's texture; axis is the one averaged away."""
    import cv2

    b = cv2.GaussianBlur(band, (0, 0), 1.5) - cv2.GaussianBlur(band, (0, 0), 8)
    spread = b.std()
    if spread < 1e-3:
        return 0.0
    p = b.mean(axis=axis)
    p = p - p.mean()
    f = np.fft.rfft(p)
    k = np.arange(len(f))
    period = len(p) / np.maximum(k, 1e-9)
    f[(period < PERIODS[0]) | (period > PERIODS[1])] = 0
    return float(np.fft.irfft(f, len(p)).std() / spread)


def ecard_reverse(card_rgb):
    """{"score", "bottom", "left"} for a straightened card, RGB uint8, any size."""
    import cv2

    grey = cv2.cvtColor(np.ascontiguousarray(card_rgb), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(grey, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    x0, x1, ya, yb, h = BOTTOM
    bottom = max(_strength(g[y:y + h, x0:x1], 0) for y in range(ya, yb - h + 1, SLIDE_STEP))
    y0, y1, xa, xb, w = LEFT
    left = max(_strength(g[y0:y1, x:x + w], 1) for x in range(xa, xb - w + 1, SLIDE_STEP))
    return {"score": round((bottom + left) / 2, 4), "bottom": round(bottom, 4), "left": round(left, 4)}


def reading(s):
    """"bars", "dot_code" or None (unsure) for an `ecard_reverse` result."""
    if s["bottom"] >= BARS_BOTTOM and s["left"] >= BARS_LEFT:
        return "bars"
    if s["score"] <= STANDARD:
        return "dot_code"
    return None
