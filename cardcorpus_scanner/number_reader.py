"""Stage 5: read the collector number.

A small CRNN reads grey strips cut from where numbers are printed (six
places, `slots`), each at several offsets and two scales: 180 strips per
card. The free read is the number most strips agree on; its vote count is
the confidence (the CTC softmax is near 1.0 always, so don't gate on it).
The constrained read (`fit`) scores how well the strips show each of a
list of known numbers, which still works on strips too blurry for a free
read.
"""
import re

STRIP_H, STRIP_W = 48, 384


def parse_nd(s):
    m = re.fullmatch(r"(\d{1,3})/(\d{1,3})", s)
    if not m:
        return None
    n, d = int(m.group(1)), int(m.group(2))
    if d < 10 or d > 300 or n < 1 or n > d + 60:
        return None
    return (n, d)


# letter-coded numbers: "TG14/TG30", "SM-P123", "016"
CODE_RE = re.compile(r"^[0-9A-Z][0-9A-Z/.\-]{1,11}$")


def plausible_code(s):
    return bool(CODE_RE.fullmatch(s) and any(c.isdigit() for c in s) and len(s) >= 3)


def printed_nd(number):
    """A catalogue number as printed at the card's foot ("17/64"), or None
    for numbers that aren't (an old Japanese card's "No. 006" is a Pokédex
    number; promo codes)."""
    t = re.sub(r"\s+", "", number or "")
    return t if re.fullmatch(r"\d{1,3}/\d{1,3}", t) else None


def _crnn(ncls):
    import torch.nn as nn

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 1)),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 1)),
            )
            self.proj = nn.Linear(128 * 3, 192)
            self.rnn = nn.LSTM(192, 128, num_layers=2, bidirectional=True, batch_first=True)
            self.head = nn.Linear(256, ncls)

        def forward(self, x):
            f = self.conv(x)
            b, c, h, w = f.shape
            f = self.proj(f.permute(0, 3, 1, 2).reshape(b, w, c * h))
            f, _ = self.rnn(f)
            return self.head(f)

    return CRNN()


def build(charset):
    """The CRNN for a charset, without weights: (batch, 1, 48, 384) grey
    strips in [0, 1] -> (batch, 96, len(charset) + 1) logits, class 0 the CTC
    blank."""
    return _crnn(len(charset) + 1)


class Reader:
    """settings: the number reader's json (charset, strip size, slots, and
    the set sizes the free read prefers: `denominators`); numbers: every
    card's printed number, to count those set sizes from when the json has
    none."""

    # strip offsets scanned: a box can sit a few % off the card
    GRID = {"box": ((-0.025, -0.012, 0, 0.012, 0.025), (-0.04, 0, 0.04), (0.92, 1.12)),
            "aligned": ((-0.008, 0, 0.008), (0,), (0.92, 1.12))}

    def __init__(self, weights, settings, numbers, device="cpu"):
        """weights: a safetensors file, or a state dict already in memory."""
        import torch

        from .tower import state_dict

        self.torch = torch
        self.charset = settings["charset"]
        self.strip_h, self.strip_w = settings.get("strip", [STRIP_H, STRIP_W])
        self.slots = settings["slots"]
        self.model = build(self.charset)
        self.model.load_state_dict(state_dict(weights))
        self.model.eval().to(device)
        self.device = device
        if settings.get("denominators"):
            # the build's list: every set size printed on at least five cards
            self.denoms = set(settings["denominators"])
            return
        denoms = {}
        for number in numbers or ():
            m = re.fullmatch(r"(\d{1,3})\s*/\s*(\d{1,3})", (number or "").strip())
            if m:
                d = int(m.group(2))
                denoms[d] = denoms.get(d, 0) + 1
        self.denoms = {d for d, c in denoms.items() if c >= 5}

    def _to_tensor(self, strip):
        import numpy as np

        w = min(self.strip_w, max(16, int(strip.width * self.strip_h / max(1, strip.height))))
        strip = strip.resize((w, self.strip_h))
        arr = np.full((self.strip_h, self.strip_w), 0.5, dtype=np.float32)
        arr[:, :w] = np.asarray(strip, np.float32) / 255.0
        return self.torch.from_numpy(arr)[None]

    def _decode(self, logits):
        best = logits.argmax(-1).tolist()
        conf = logits.softmax(-1).max(-1).values.mean().item()
        out, prev = [], 0
        for t in best:
            if t != 0 and t != prev:
                out.append(self.charset[t - 1])
            prev = t
        return "".join(out), conf

    def strips(self, img, grid="box"):
        """Every strip variant (slot x offset x scale) of an upright card
        (PIL), as (variants, 1, 48, 384) tensors."""
        img = img.convert("L")
        W, H = img.size
        dys, dxs, scs = self.GRID[grid]
        variants = []
        for s in self.slots:
            for dy in dys:
                for dx in dxs:
                    for sc in scs:
                        bw, bh = s["w"] * W * 1.55 * sc, s["h"] * H * 2.1 * sc
                        cx, cy = (s["cx"] + dx) * W, (s["cy"] + dy) * H
                        variants.append(self._to_tensor(img.crop(
                            (int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)))))
        return variants

    def strip_logits(self, img, grid="box"):
        """CRNN outputs for every strip variant: (variants, T, classes)."""
        variants = self.strips(img, grid)
        out = []
        with self.torch.no_grad():
            for i in range(0, len(variants), 64):
                batch = self.torch.stack(variants[i:i + 64]).to(self.device)
                out.append(self.model(batch).float().cpu())
        return self.torch.cat(out)

    def read(self, img):
        """The free read of an upright card (PIL), or None."""
        return self.read_logits(self.strip_logits(img))

    def read_logits(self, logits):
        """Free read: the parse most strip variants agree on, or None.
        {"text", "n", "d", "code", "conf", "votes"}."""
        votes = {}
        for j in range(logits.shape[0]):
            text, conf = self._decode(logits[j])
            nd = parse_nd(text)
            key = nd if nd else (text if plausible_code(text) else None)
            if key is not None:
                v = votes.setdefault(key, {"text": text, "conf": 0.0, "votes": 0})
                v["votes"] += 1
                v["conf"] = max(v["conf"], conf)
        if not votes:
            return None
        best_key, v = max(votes.items(), key=lambda kv: (
            kv[1]["votes"]
            + (6 if isinstance(kv[0], tuple) and kv[0][1] in self.denoms else 0)
            # n/d reads outrank code reads at equal votes: codes are noisier
            # and only matter when no n/d is printed
            + (2 if isinstance(kv[0], tuple) else 0),
            kv[1]["conf"]))
        nd = best_key if isinstance(best_key, tuple) else None
        return {"text": v["text"], "n": nd[0] if nd else None, "d": nd[1] if nd else None,
                "code": None if nd else v["text"],
                "conf": round(v["conf"], 3), "votes": v["votes"]}

    def spellings(self, number):
        """How a catalogue number may be printed: as listed, and with the
        numerator zero-padded to the denominator's width (004/102)."""
        t = re.sub(r"\s+", "", number or "")
        out = {t}
        m = re.fullmatch(r"(\d{1,3})/(\d{1,3})", t)
        if m:
            n, d = m.groups()
            out |= {f"{int(n)}/{d}", f"{int(n):0{len(d)}d}/{d}"}
        return [x for x in out if x and all(c in self.charset for c in x)]

    def fit(self, logits, numbers):
        """Constrained read: {number: log-likelihood of the strips showing
        it} (CTC, best strip variant, best spelling)."""
        F = self.torch.nn.functional
        lp = logits.log_softmax(-1).permute(1, 0, 2)  # (T, variants, classes)
        T, V = lp.shape[0], lp.shape[1]
        out = {}
        for number in numbers:
            best = None
            for sp in self.spellings(number):
                tgt = self.torch.tensor([self.charset.index(c) + 1 for c in sp])
                nll = F.ctc_loss(lp, tgt.repeat(V, 1), self.torch.full((V,), T, dtype=self.torch.long),
                                 self.torch.full((V,), len(tgt), dtype=self.torch.long),
                                 reduction="none")
                ll = float(-nll.min())
                best = ll if best is None else max(best, ll)
            if best is not None:
                out[number] = round(best, 3)
        return out
