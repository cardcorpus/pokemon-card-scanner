"""The scanner: a photo in, every card in it named.

Per card (SPEC.md has every number):

  1. detect        QuadDet finds each card's four corners
  2. orient, embed each turn of the card is straightened and fingerprinted
                   by the matcher; the orientation head says which turn is
                   upright
  3. retrieve      the fingerprint's nearest pictures in the index, one
                   candidate per card
  4. straighten    the card again at full resolution (960 x 1340)
  5. number        the collector number, read from strips of the card
  6. pick          the print picker's say among prints of one painting
  7. fuse, verify  the candidates re-ordered by picture similarity plus the
                   weighted evidence; the print verifier compares the card
                   with each sibling print's reference picture
  8. 1st Edition stamp score
  9. e-Card reverse holo score

    from cardcorpus_scanner import Scanner
    scanner = Scanner.load("path/to/weights")   # or Scanner.load() from the Hub
    result = scanner.scan("binder-page.jpg")
"""
import json
import re
from pathlib import Path

import numpy as np

from . import marks, tower
from .detector import Detector, containment
from .index import Index
from .number_reader import Reader, printed_nd
from .verifier import MIRROR, Pictures, Verifier

DEFAULTS = {
    "detector": {"conf": 0.2, "contain": 0.75, "model": {}},
    "retrieval": {"top": 20, "candidates": 10},
    "picker": {"size": [448, 320]},
    "number": {"votes_gate": 5, "numfit_gate": -5.0, "numfit_scale": 5.0,
               "numfit_abstain_votes": 10, "numfit_abstain_margin": 1.0},
    "twin": {"sim_gap": 0.02},
    "orientation": {"match_margin": 0.08, "match_floor": 0.90},
    "verifier": {"max_prints": 5, "sigma_d": 2.5, "tau": 0.3, "min_mass": 0.005, "border": 0.05,
                 "glare": 0.96, "art_box": [0.07, 0.09, 0.93, 0.54]},
    "fusion": {"num": 0.2, "code": 0.04, "set": 0.03, "twin": 0.008, "pick": 0.1, "verify": 0.1,
               "numfit": 0.2, "prior": 0.02},
    # the rules around the stages (SPEC.md, "Around the stages")
    "rules": {"work_px": 3000, "detail_px": 320, "cut_off": True, "left_out": True, "rescue_conf": 0.05,
              "rescue_sim": 0.80, "rescue_overlap": 0.5, "junk_conf": 0.35, "junk_sim": 0.7, "back_sim": 0.60,
              "realigns": True, "verifier_yields": True, "twin_tie": 0.03, "rules_out": True},
}
HUB_REPO = "cardcorpus/pokemon-card-scanner"


def load_config(folder):
    cfg = json.loads(json.dumps(DEFAULTS))
    for section, vals in json.loads((Path(folder) / "pipeline.json").read_text()).items():
        if isinstance(vals, dict) and isinstance(cfg.get(section), dict):
            cfg[section].update(vals)
        else:
            cfg[section] = vals
    return cfg


def persp_coeffs(srcq, dstq):
    a, b = [], []
    for (x, y), (u, v) in zip(srcq, dstq):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.extend([u, v])
    return np.linalg.solve(np.array(a, float), np.array(b, float)).tolist()


def touches_frame(quad, width, height):
    """Whether a box reaches the photo's edge: within half a percent of its
    longer side."""
    margin = 0.005 * max(width, height)
    return bool(((quad[:, 0] < margin) | (quad[:, 1] < margin) | (quad[:, 0] > width - margin)
                 | (quad[:, 1] > height - margin)).any())


# A box is a card cut off by the photo's edge when it touches the edge and
# its area is under this share of the photo's median card. Measured on 1,594
# boxes of real photos: it caught 74 of 80 slivers the matcher could make
# nothing of, and flagged 3 of 1,514 whole cards.
CUT_OFF_AREA = 0.6


def polygon_area(quad):
    x, y = np.asarray(quad, dtype=np.float64).T
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def cut_off_flags(quads, width, height):
    """Which boxes are of cards cut off by the photo's edge: a snippet of a
    card mostly out of frame, which the detector finds readily and the
    matcher can make nothing of. The median is taken over the boxes clear
    of the edge when there are any, so several snippets can't pass for the
    usual size."""
    touching = [touches_frame(np.asarray(q), width, height) for q in quads]
    areas = [polygon_area(q) for q in quads]
    clear = [a for a, t in zip(areas, touching) if not t]
    sizes = clear or areas
    if not sizes:
        return []
    median = float(np.median(sizes))
    return [t and a < CUT_OFF_AREA * median for a, t in zip(areas, touching)]


def covered_fraction(box, other):
    """How much of `box` lies inside `other` (both convex), as a share of
    `box`'s area."""
    import cv2

    area = polygon_area(box)
    if area <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(np.float32(box).reshape(-1, 1, 2), np.float32(other).reshape(-1, 1, 2))
    return float(inter) / area


def shared_name(name):
    """The name every printing of a card carries: without a bracketed note
    ("Misty's Goldeen (LV.10)"), accents, capitals or apostrophes."""
    import unicodedata

    if not name:
        return None
    base = name.split(" (")[0]
    folded = "".join(ch for ch in unicodedata.normalize("NFD", base) if unicodedata.category(ch) != "Mn")
    return folded.casefold().replace("'", "").replace("\u2019", "")

# How far a pick can be trusted, as the app says it (MatchQuality): a
# strong picture match (SURE_SIM) is sure when no other picture is within
# SURE_MARGIN of it, or when no candidate the evidence and the alignment
# check leave in is within SURE_FUSED of it by fused score.
SURE_SIM = 0.88
SURE_MARGIN = 0.06
SURE_FUSED = 0.15
RULE_OUT_CHECKS = 4


def ident(c):
    """A candidate's identity: its print, else its picture's handle."""
    return c.get("print") or c.get("handle")


def match_quality(ordered, scores, back, likely_wrong, ruled_out, ident=ident):
    """"back", "likely_wrong", "sure", "printing_unclear" (only other prints
    of the same card are close) or "unsure", for candidates in their final
    order with their fused scores."""
    if back:
        return "back"
    if likely_wrong:
        return "likely_wrong"
    if not ordered or ordered[0]["sim"] < SURE_SIM:
        return "unsure"
    best, top = ordered[0], scores[0]
    picture = [c for c in ordered[1:] if best["sim"] - c["sim"] < SURE_MARGIN]
    rivals = [c for c, s in zip(ordered[1:], scores[1:]) if top - s < SURE_FUSED and ident(c) not in ruled_out]
    if not rivals or not picture:
        return "sure"
    name = shared_name(best.get("name"))
    return "printing_unclear" if name is not None and all(shared_name(c.get("name")) == name for c in rivals) \
        else "unsure"

# The lead past which the app takes its best guess instead of asking, when
# the doubt is only which print, or between two cards (isDecided).
DECIDED_LEAD = 0.05


def is_decided(quality, ordered, scores, verify_missing):
    """Whether the best guess stands without asking: a "printing_unclear"
    with one name close, or an "unsure" with at most two, and a fused lead
    of DECIDED_LEAD or more. A doubt only between prints that the verifier
    should have settled but couldn't (verify_missing: two or more of its
    prints close before it ran, one without a score) is never decided.
    ordered, scores: the final order and fused scores."""
    if quality not in ("printing_unclear", "unsure") or not ordered:
        return False
    if len(ordered) > 1 and scores[0] - scores[1] < DECIDED_LEAD:
        return False
    top = scores[0]
    names = {shared_name(c.get("name")) for k, (c, s) in enumerate(zip(ordered, scores))
             if k == 0 or top - s < SURE_FUSED} - {None}
    if verify_missing and len(names) <= 1:
        return False
    return len(names) <= (1 if quality == "printing_unclear" else 2)




def is_holo(row):
    """A holo print (not a reverse holo), whose art window the print
    verifier leaves out: by the card's default finish, else its rarity."""
    row = row or {}
    finish = row.get("default_finish")
    if finish:
        return finish != "non_holo" and not finish.startswith("reverse")
    rarity = (row.get("rarity") or "").lower()
    return "holo" in rarity and "reverse" not in rarity


def pick_device(want="auto"):
    """'auto' -> cuda, then Apple silicon's mps, then cpu."""
    import torch

    if want and want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Scanner:
    W, H = 384, 536  # the straightened card the matcher and orientation head see
    HW, HH = 960, 1340  # full resolution: number, picker, verifier, marks
    HALF_TURN_FLOOR = 0.80
    HALF_TURN_MARGIN = 0.02

    @classmethod
    def load(cls, weights=None, **kw):
        """weights: a local folder, or None for the Hub's copy (needs
        huggingface_hub; it is downloaded once and cached)."""
        if weights is None or not Path(weights).exists():
            from huggingface_hub import snapshot_download

            weights = snapshot_download(weights or HUB_REPO)
        return cls(weights, **kw)

    def __init__(self, weights, device="cpu", number_reader=True, picker=True, verifier=True,
                 mirror=MIRROR, picture_cache=None, picture_folder=None, picture_width=None):
        W = Path(weights)
        cfg = load_config(W)
        index = Index(W / "index")
        models = W / "models"
        det = cfg["detector"]
        from safetensors.torch import load_file

        heads = load_file(str(models / "heads.safetensors"))
        pick = None
        if picker and index.picker_embeddings is not None and (models / "picker.safetensors").exists():
            pspec = {**cfg["embedder"], **cfg["picker"].get("backbone", {}), "size": list(cfg["picker"]["size"])}
            pick = (tower.load(models / "picker.safetensors", pspec, device), tower.picker_transform(pspec))
        reader = None
        if number_reader:
            settings = json.loads((models / "number-reader.json").read_text())
            numbers = [r.get("number") for r in index.cards.values() if r.get("kind") == "card"]
            reader = Reader(models / "number-reader.safetensors", settings, numbers, device)
        ver = None
        if verifier and (mirror or picture_folder):
            vcfg = {k: v for k, v in cfg["verifier"].items() if k != "max_prints"}
            ver = Verifier(Pictures(mirror, picture_cache, picture_folder, copy_width=picture_width), **vcfg)
        self._setup(cfg, index, device,
                    detector=Detector(models / "detector.safetensors", det.get("model"), det["conf"], device),
                    matcher=tower.load(models / "matcher.safetensors", cfg["embedder"], device),
                    preprocess=tower.preprocess(cfg["embedder"]),
                    heads=heads, picker=pick, reader=reader, verifier=ver, mirror=mirror)

    def _setup(self, cfg, index, device, detector, matcher, preprocess, heads, picker=None, reader=None,
               verifier=None, mirror=MIRROR):
        """The scanner from parts already made: its settings (load_config),
        an Index (or anything with the same fields), and the models. The
        detector is called with a photo (h, w, 3 RGB) and gives [(score,
        quad)]; matcher(x) fingerprints a batch preprocess() made; heads holds
        the "back_prototype", "orient_probe" and "orient_hidden" tensors (any
        may be missing); picker is (tower, transform) or None. Scanner.load
        makes them from a weights folder; other loaders (another bundle
        format, a research model) can hand in their own."""
        import torch

        self.torch = torch
        self.cfg = cfg
        self.device = device
        self.index = index
        self.cards = self.index.cards
        self.matrix = torch.from_numpy(self.index.embeddings)
        # each print's scan prior: -1 for a print a scan names only on clear
        # evidence (a World Championships deck's reprint of a booster card)
        self.prior = {r["print"]: float(r["scan_prior"]) for r in self.cards.values()
                      if r.get("print") and r.get("scan_prior")}
        # A second view with no print of its own stands for the first card
        # with a print in its picture group.
        self.groups = self.index.groups
        by_group = {}
        for h, row in self.cards.items():
            g = self.groups.get(h)
            if g and row.get("print"):
                by_group.setdefault(g, h)
        self.stand_in = {}
        for h, row in self.cards.items():
            if not row.get("print"):
                sib = by_group.get(self.groups.get(h, ""))
                if sib:
                    self.stand_in[h] = sib
        # each print's own picture (its catalogue picture)
        self.own = {row["print"]: h for h, row in self.cards.items() if row.get("print") and row.get("kind") == "card"}
        self.row_family = np.array([(self.cards.get(self.stand_in.get(h, h)) or {}).get("family") or ""
                                    for h in self.index.handles])

        self.detector = detector
        self.matcher = matcher
        self.preprocess = preprocess
        self.back_proto = heads.get("back_prototype")
        self.orient_w = heads.get("orient_probe")
        self.orient_hidden = heads.get("orient_hidden")

        self.picker = None
        if picker is not None and self.index.picker_embeddings is not None:
            self.picker, self.pick_tf = picker
            self.pick_emb = torch.from_numpy(self.index.picker_embeddings)
            self.pick_rows, self.pick_group = {}, {}
            for i, m in enumerate(self.index.picker_rows):
                self.pick_rows.setdefault(m["print"], []).append(i)
                self.pick_group[m["print"]] = m["group"]
            self.pick_members = {}
            for p, g in self.pick_group.items():
                self.pick_members.setdefault(g, []).append(p)

        self.reader = reader
        self.verifier = verifier
        self.mirror = (mirror or "").rstrip("/")

    # --- helpers ------------------------------------------------------------

    def embed(self, imgs):
        """Unit-length fingerprints (CPU tensor) for PIL images."""
        torch = self.torch
        with torch.no_grad():
            x = torch.stack([self.preprocess(im) for im in imgs]).to(self.device)
            f = self.matcher(x)
            return (f / f.norm(dim=-1, keepdim=True)).float().cpu()

    def family_mask(self, families):
        """Bool tensor over index rows where the row's language family
        (international, japanese, zh-hans) is allowed; None = all."""
        if not families:
            return None
        return self.torch.from_numpy(np.isin(self.row_family, list(families)))

    def candidate(self, handle, score):
        row = self.cards.get(handle, {})
        out = {
            "handle": handle,
            "print": row.get("print"),
            "set": row.get("set"),
            "card": row.get("card"),
            "name": row.get("name"),
            "set_name": row.get("set_name"),
            "number": row.get("number"),
            "family": row.get("family"),
            "language": row.get("language"),
            "release": row.get("release"),
            "artwork_group": row.get("artwork_group"),
            "release_date": row.get("release_date"),
            "sim": round(score, 4),
        }
        if row.get("variant"):
            out["variant"] = row["variant"]
        if row.get("picture") and self.mirror:
            out["picture"] = f"{self.mirror}/m/{row['picture']}"
        return out

    def dedupe(self, scores, idxs, limit=10, keep=None):
        """Index rows best first, one entry per card: the index holds several
        pictures of many cards, and one card must not take several places.
        keep(card row) -> bool filters rows."""
        out, seen = [], set()
        for score, idx in zip(scores, idxs):
            if score <= -2.0:  # a row the language filter left out
                break
            h = self.index.handles[idx]
            h = self.stand_in.get(h, h)
            row = self.cards.get(h, {})
            if keep is not None and not keep(row):
                continue
            ident = row.get("print") or (row.get("name"), row.get("set_name"), row.get("number"), row.get("family"))
            if ident in seen:
                continue
            seen.add(ident)
            if len(out) >= limit:
                break
            out.append(self.candidate(h, score))
        return out

    def rank(self, feat, limit=10, families=None, keep=None):
        """One fingerprint against the whole index, for a search of your own
        (keep filters card rows, families the language)."""
        sims = self.matrix @ feat
        mask = self.family_mask(families)
        if mask is not None:
            sims = sims.masked_fill(~mask, -2.0)
        if keep is None:
            top = self.torch.topk(sims, min(limit * 4, sims.shape[0]))
            return self.dedupe(top.values.tolist(), top.indices.tolist(), limit)
        order = self.torch.argsort(sims, descending=True)
        n_ok = int((sims > -2.0).sum())
        return self.dedupe(sims[order[:n_ok]].tolist(), order[:n_ok].tolist(), limit, keep)

    # --- fusion ---------------------------------------------------------------

    def evidence(self, cands, nr=None, page_sets=None, own_top=None, pick=None, ver=None, nfit=None):
        """Each candidate's evidence (same order as cands). page_sets counts
        the photo's best candidates' releases including this card's own
        (own_top, which is taken off)."""
        ncfg = self.cfg["number"]
        gate = ncfg["votes_gate"]
        votes = (nr or {}).get("votes") or 0
        read = (nr["n"], nr["d"]) if nr and nr.get("n") is not None and nr.get("d") is not None else None
        code = (nr or {}).get("code")
        fold = lambda x: re.sub(r"[^0-9A-Z]", "", (x or "").upper())
        pick_best = max(pick.values()) if pick else 0.0
        present = {c.get("print") for c in cands}
        ver = {p: v for p, v in (ver or {}).items() if p in present}
        ver_best = max(ver.values()) if ver else 0.0
        # constrained read: how well each candidate's number explains the
        # strips, against the best; only when the strips clearly show one
        lls = {n: nfit[n] for n in (printed_nd(c.get("number")) for c in cands) if n in (nfit or {})}
        nf_best = max(lls.values()) if len(lls) >= 2 else None
        if nf_best is not None and nf_best < ncfg["numfit_gate"]:
            nf_best = None
        # the strips plainly show a number none of the candidates have
        # (often a neighbouring card's): the constrained read counts for none
        if self.numfit_abstains(cands, nr, nfit):
            nf_best = None
        # reprint twins (one painting, near-tied similarity): the earlier
        # print, as reprints are rarer in hand than the cards they copy
        twin = set()
        by_art = {}
        for c in cands:
            if c.get("artwork_group"):
                by_art.setdefault(c["artwork_group"], []).append(c)
        for group in by_art.values():
            best_sim = max(c["sim"] for c in group)
            tied = [c for c in group if best_sim - c["sim"] <= self.cfg["twin"]["sim_gap"] and c.get("release_date")]
            if len(tied) >= 2:
                twin.add(min(tied, key=lambda c: c["release_date"])["handle"])
        out = []
        for c in cands:
            f = {}
            mm = re.search(r"(\d{1,3})\s*/\s*(\d{1,3})", c.get("number") or "")
            cnd = (int(mm.group(1)), int(mm.group(2))) if mm else None
            if read and cnd:
                if cnd == read and votes >= gate:
                    f["num"] = 1.0
            elif not read and code and votes >= gate and fold(c.get("number")) and \
                    fold(c.get("number")) == fold(code):
                f["code"] = 1.0
            rel = c.get("release")
            if rel and page_sets and page_sets.get(rel, 0) - (1 if rel == own_top else 0) >= 1:
                f["set"] = 1.0
            if c["handle"] in twin:
                f["twin"] = 1.0
            if pick and c.get("print") in pick:
                f["pick"] = pick[c["print"]] - pick_best  # <= 0
            if len(ver) > 1 and c.get("print") in ver:
                f["verify"] = ver[c["print"]] - ver_best  # <= 0
            if nf_best is not None:
                n = printed_nd(c.get("number"))
                if n is None:
                    f["numfit"] = -1.0
                elif n in lls:
                    f["numfit"] = max(-1.0, (lls[n] - nf_best) / ncfg["numfit_scale"])  # <= 0
            if self.prior.get(c.get("print")):
                f["prior"] = self.prior[c["print"]]
            out.append(f)
        return out

    def numfit_abstains(self, cands, nr, nfit):
        """Whether the strips plainly show a number that is none of the
        candidates': the free read is sure of it, no candidate carries it,
        and it fits the strips better than every candidate's by the margin."""
        ncfg = self.cfg["number"]
        gate = ncfg.get("numfit_abstain_votes")
        if not gate or not nr or not nfit or (nr.get("votes") or 0) < gate or \
                nr.get("n") is None or nr.get("d") is None:
            return False
        read_fit = nfit.get(f"{nr['n']}/{nr['d']}")
        if read_fit is None:
            return False
        best = None
        for c in cands:
            mm = re.search(r"(\d{1,3})\s*/\s*(\d{1,3})", c.get("number") or "")
            if mm and (int(mm.group(1)), int(mm.group(2))) == (nr["n"], nr["d"]):
                return False
            n = printed_nd(c.get("number"))
            if n in nfit:
                best = nfit[n] if best is None else max(best, nfit[n])
        return best is not None and read_fit - best >= ncfg["numfit_abstain_margin"]

    def fuse(self, cands, nr=None, page_sets=None, own_top=None, pick=None, with_features=False, ver=None, nfit=None,
             scores_out=None):
        """Candidates re-ordered by similarity + weighted evidence (each
        keeps its own similarity). scores_out, a list, gets each candidate's
        fused score, in cands' order."""
        feats = self.evidence(cands, nr, page_sets, own_top, pick, ver, nfit)
        w = self.cfg["fusion"]
        score = [c["sim"] + sum(w.get(k, 0.0) * v for k, v in f.items()) for c, f in zip(cands, feats)]
        if scores_out is not None:
            scores_out[:] = score
        order = sorted(range(len(cands)), key=lambda i: -score[i])
        ordered = [cands[i] for i in order]
        if not with_features:
            return ordered
        return ordered, [{"print": cands[i].get("print"), "sim": cands[i]["sim"], **feats[i]} for i in range(len(cands))]

    def number_fit(self, strip, cands, nr=None):
        """{printed number: strip log-likelihood} for the candidates' n/d
        numbers, and the free read's own beside them; or None."""
        if self.reader is None or strip is None:
            return None
        nums = sorted({n for n in (printed_nd(c.get("number")) for c in cands) if n})
        if len(nums) < 2:
            return None
        if nr and nr.get("n") is not None and nr.get("d") is not None and f"{nr['n']}/{nr['d']}" not in nums:
            nums.append(f"{nr['n']}/{nr['d']}")
        return self.reader.fit(strip, nums)

    def verify(self, hi, cands, keep=None):
        """Print verifier scores {print: -1..1} for the prints among cands
        (fused order) that share the best one's picture group, painting or
        name; or None."""
        if self.verifier is None or not cands or hi is None:
            return None
        group = lambda c: self.groups.get(self.own.get(c.get("print"), c["handle"]))
        top = cands[0]
        g, art, name = group(top), top.get("artwork_group"), (top.get("name") or "").lower()
        sibs = [c for c in cands if c.get("print") and
                ((g and group(c) == g) or (art and c.get("artwork_group") == art) or
                 (name and (c.get("name") or "").lower() == name))][:self.cfg["verifier"]["max_prints"]]
        if len(sibs) < 2:
            return None
        if keep is not None:
            keep["siblings"] = sibs
        own = {c["print"]: self.own.get(c["print"], c["handle"]) for c in sibs}
        pictures = {p: self.index.verifier_pictures.get(p) or (self.cards.get(h) or {}).get("picture") or h
                    for p, h in own.items()}
        holo = [p for p in pictures if is_holo(self.cards.get(own[p]))]
        try:
            return self.verifier.score(np.asarray(hi.convert("RGB")), pictures, holo, keep=keep)
        except Exception:  # a missing picture or an odd image never fails the scan
            return None

    # --- stages -----------------------------------------------------------------

    def detect(self, img):
        """[(quad 4 x 2 in photo pixels, confidence)] after containment."""
        arr = np.asarray(img)
        found = containment(self.detector(arr), img.width, img.height, self.cfg["detector"]["contain"])
        return [(np.array(q, dtype=np.float32), s) for s, q in found]

    def orient_embed(self, img, quad):
        """Straighten the box at 384 x 536 in each turn of its corners, and
        keep the one that reads upright. (crop, oriented quad, fingerprint,
        orientation probability, that turn's index similarities) or None."""
        from PIL import Image

        W, H = self.W, self.H
        c = quad.mean(0)
        q0 = quad[np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))]
        crops, rolls = [], []
        for start in range(4):
            q = np.roll(q0, -start, axis=0)
            if self.orient_w is not None or np.linalg.norm(q[1] - q[0]) <= np.linalg.norm(q[2] - q[1]):
                coeffs = persp_coeffs([(0, 0), (W, 0), (W, H), (0, H)], q)
                crops.append(img.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC))
                rolls.append(q)
        if not crops:
            return None
        feats = self.embed(crops)
        sims = feats @ self.matrix.T
        if self.orient_w is not None:
            h = feats
            if self.orient_hidden is not None:
                h = (h @ self.orient_hidden[:, :-1].T + self.orient_hidden[:, -1]).clamp(min=0)
            logits = h @ self.orient_w[:, :-1].T + self.orient_w[:, -1]
            upright = self.torch.softmax(logits, dim=1)[:, 0]
            best = probe = int(upright.argmax())
            # another turn whose best match is clearly better wins (a card
            # printed sideways reads upright to the head when turned)
            ocfg = self.cfg.get("orientation") or {}
            if ocfg.get("match_margin") is not None:
                top = [round(float(v), 4) for v in sims.max(dim=1).values]
                alt = max(range(len(top)), key=lambda k: top[k])
                if alt != best and top[alt] - top[best] >= ocfg["match_margin"] and top[alt] >= ocfg["match_floor"]:
                    best = alt
            # a card that reads alike both ways up: the half turn wins when
            # the index finds the same card there, better
            if best == probe and len(rolls) == 4 and not touches_frame(quad, img.width, img.height):
                turn = (best + 2) % 4
                if self.half_turn_wins(sims[best], sims[turn]):
                    best = turn
            orient_p = round(float(upright[best]), 4)
        else:
            best = int(sims.max(dim=1).values.argmax())
            orient_p = None
        return crops[best], rolls[best], feats[best], orient_p, sims[best]

    def half_turn_wins(self, chosen, turned):
        def top(sims):
            value, row = sims.max(dim=0)
            h = self.index.handles[int(row)]
            return float(value), (self.cards.get(self.stand_in.get(h, h)) or {}).get("name")
        (was, name), (now, turned_name) = top(chosen), top(turned)
        return bool(name) and turned_name == name and now >= was + self.HALF_TURN_MARGIN and now >= self.HALF_TURN_FLOOR

    def retrieve(self, sims, fam_mask=None):
        rcfg = self.cfg["retrieval"]
        if fam_mask is not None:
            sims = sims.masked_fill(~fam_mask, -2.0)
        top = self.torch.topk(sims, min(rcfg["top"], sims.shape[0]))
        return self.dedupe(top.values.tolist(), top.indices.tolist(), rcfg["candidates"])

    def hires(self, img, quad):
        from PIL import Image

        HW, HH = self.HW, self.HH
        return img.transform((HW, HH), Image.PERSPECTIVE,
                             persp_coeffs([(0, 0), (HW, 0), (HW, HH), (0, HH)], np.array(quad, float)),
                             Image.BICUBIC)

    def pick(self, hi, cands):
        """Print picker scores {print: cosine} for the prints of the best
        candidate's painting group (when it has several), or None."""
        top = cands[0].get("print") if cands else None
        if self.picker is None or hi is None or top not in self.pick_group:
            return None
        prints = self.pick_members[self.pick_group[top]]
        if len(prints) < 2:
            return None
        torch = self.torch
        x = self.pick_tf(hi.convert("RGB"))[None].to(self.device)
        with torch.no_grad():
            pf = self.picker(x)
            pf = (pf / pf.norm(dim=-1, keepdim=True)).float().cpu()[0]
        return {p: float((self.pick_emb[self.pick_rows[p]] @ pf).max()) for p in prints}

    # --- the rules around the stages -------------------------------------------

    def detect_boxes(self, img, rules):
        """Stage 1 with the box rules: [(quad, confidence, cut_off)]. With
        left_out on, boxes down to rescue_conf; each flagged when it is a
        sliver of a card cut off by the photo's edge. The photo's usual card
        size comes from the boxes over the detector's threshold, so lower
        ones don't change which of those are cut off; each lower one is
        judged against them."""
        thr = self.cfg["detector"]["conf"]
        low = rules["rescue_conf"] if rules["left_out"] and rules["rescue_conf"] is not None else thr
        old = getattr(self.detector, "conf", None)
        if low < thr and old is not None:
            self.detector.conf = low
        try:
            found = self.detect(img)
        finally:
            if old is not None:
                self.detector.conf = old
        if not rules["cut_off"]:
            # every box, low ones too: the rescue (left_out) decides on those
            return [(q, c, False) for q, c in found]
        sure = [q for q, c in found if c >= thr]
        if not sure:
            flags = cut_off_flags([q for q, _ in found], img.width, img.height)
            return [(q, c, f) for (q, c), f in zip(found, flags)]
        sure_flags = iter(cut_off_flags(sure, img.width, img.height))
        return [(q, c, next(sure_flags) if c >= thr else cut_off_flags(sure + [q], img.width, img.height)[-1])
                for q, c in found]

    def left_out(self, items, rules):
        """The boxes to drop after stage 3, as {index: reason}. Walking by
        confidence: a box over the detector's threshold goes when it is
        likely junk (an unsure detector and a weak best match, not a card
        back); one under it is kept only when its best match is strong and
        it overlaps no card already kept, either way round."""
        thr = self.cfg["detector"]["conf"]
        dropped, kept = {}, []
        for i in sorted(range(len(items)), key=lambda i: -items[i]["conf"]):
            d = items[i]
            top = d["cands"][0]["sim"] if d["cands"] else 0.0
            back = d["back_sim"] is not None and d["back_sim"] >= rules["back_sim"]
            if d["conf"] >= thr:
                if not back and d["conf"] < rules["junk_conf"] and top < rules["junk_sim"]:
                    dropped[i] = "junk"
                else:
                    kept.append(d["q"])
                continue
            strong = top >= rules["rescue_sim"]
            on_kept = any(covered_fraction(d["q"], k) >= rules["rescue_overlap"]
                          or covered_fraction(k, d["q"]) >= rules["rescue_overlap"] for k in kept)
            if strong and not on_kept:
                kept.append(d["q"])
            else:
                dropped[i] = "unsure"
        return dropped

    def check_picture(self, c):
        """The picture a candidate is checked against: its print's own."""
        h = self.own.get(c.get("print"), c["handle"]) if c.get("print") else c["handle"]
        return (self.cards.get(h) or {}).get("picture")

    def check(self, d, c, prep):
        """Does the card line up with candidate c's picture? None when it
        can't be told."""
        picture = self.check_picture(c)
        if self.verifier is None or picture is None or d.get("hi") is None:
            return None
        try:
            # one grey copy per photo (the work copy, or the original for
            # small cards), shared by its cards, dropped after the photo
            grays = self.__dict__.setdefault("_grays", {})
            src = d["source"]
            if id(src) not in grays or grays[id(src)][0] is not src:
                import cv2

                grays[id(src)] = (src, cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2GRAY))
            gray = grays[id(src)][1]
            return self.verifier.check(np.asarray(d["hi"].convert("RGB")), picture, gray, d["source_quad"], prep)
        except Exception:  # an odd picture never fails the scan
            return None

    def settle(self, d, ordered, feats, scores, ver, prep, rules):
        """After fusion: the pick checked against its own picture; a pick
        that won't line up (or lines up only beside its box) gives way to
        the best match by picture when that one lines up (realigns); and
        between two near-tied prints of one card that the picture match and
        the picker both put the other way, the verifier gives way
        (verifier_yields). Returns (order, what it found)."""
        out = {}
        if self.verifier is None or not ordered or not (rules["realigns"] or rules["verifier_yields"] or rules["rules_out"]):
            return ordered, out
        cands = d["cands"]
        place = {id(c): i for i, c in enumerate(cands)}
        order = [place[id(c)] for c in ordered]
        checks = {}

        def check(i):
            if i not in checks:
                checks[i] = self.check(d, cands[i], prep)
            return checks[i]

        top = order[0]
        pick = check(top)
        realigned = yielded = False
        if rules["realigns"] and top != 0 and pick is not None and \
                (not pick["aligned"] or self.verifier.off_box(pick)):
            if self.verifier.gives_way(pick, check(0)):
                order.remove(0)
                order.insert(0, 0)
                realigned = True
        if rules["verifier_yields"] and not realigned and len(order) > 1 and ver:
            first, second = order[0], order[1]
            a, b = cands[first], cands[second]
            va, vb = ver.get(a.get("print")), ver.get(b.get("print"))
            if scores[first] - scores[second] < rules["twin_tie"] and shared_name(a.get("name")) is not None \
                    and shared_name(a.get("name")) == shared_name(b.get("name")) and b["sim"] > a["sim"] \
                    and (feats[second].get("pick") or 0) > (feats[first].get("pick") or 0) \
                    and va is not None and vb is not None and va > vb:
                order[0], order[1] = second, first
                yielded = True
                check(second)
        # another card close behind a doubted pick that plainly won't line
        # up with the photo, while the pick does, isn't a rival (ruledOut)
        ruled = []
        first = order[0]
        best = cands[first]
        doubted = best["sim"] < SURE_SIM or any(best["sim"] - cands[i]["sim"] < SURE_MARGIN for i in order[1:])
        lined = check(first)
        if rules["rules_out"] and doubted and lined is not None and lined["aligned"]:
            tried = 0
            name = shared_name(best.get("name"))
            for i in order[1:]:
                if tried >= RULE_OUT_CHECKS:
                    break
                other = cands[i]
                if scores[first] - scores[i] >= SURE_FUSED or \
                        (name is not None and shared_name(other.get("name")) == name):
                    continue
                tried += 1
                if self.verifier.plainly_other(check(i), lined):
                    ruled.append(ident(other))
        if ruled:
            out["ruled_out"] = ruled
        final = checks.get(order[0])
        if final is not None:
            out["alignment"] = {k: final[k] for k in ("aligned", "pass", "inliers", "scan_features", "box_error")}
            # plainly not this picture: both had detail enough and no pass came near a fit
            if not final["aligned"] and final["scan_features"] >= 500 and final["passes"] and \
                    final["passes"][0]["features"] >= 650 and all(x["inliers"] < 8 for x in final["passes"]):
                out["likely_wrong"] = True
        if realigned:
            out["realigned"] = True
        if yielded:
            out["yielded"] = True
        return [cands[i] for i in order], out

    def scan(self, photo, families=None, crops_dir=None, evidence=False, rules=None, on_first=None,
             on_progress=None, fingerprints=False):
        """Every card in a photo (a path or a PIL image). Returns {"width",
        "height", "cards": [...], "cut_off": [...], "left_out": [...]}, each
        card with its corners, its five best candidates in order and the
        evidence behind them.

        families limits the candidates to some language families. rules
        overrides the settings of the rules around the stages (SPEC.md,
        "Around the stages"): rules={"cut_off": False} matches the slivers
        of cards cut off by the photo's edge too. on_first, when given, is
        called with the first answer, by picture alone (stages 1 to 3: the
        same keys, each card's candidates in retrieval order and
        "first": true, and each card's "box"), before the later stages run.

        on_progress, when given, is called as the scan goes, so a page can
        show each card the moment it is known: {"step": "boxes", "width",
        "height", "cut_off", "boxes": [{"box", "quad", "conf"}]} once the
        cards are found; {"step": "pick", "box", "quad", "conf",
        "candidates"} as each is fingerprinted (a box the first answer then
        leaves out may have had one); and {"step": "card", "index", "card"}
        as each card of the first answer is double-checked, the card just
        as it will be in the answer's "cards" at that index.

        Each card carries "box", its place among the detector's boxes (the
        crop crops_dir saves for it is "<photo>-<box>.jpg"); fingerprints=True
        adds its "fingerprint" (the matcher's, a torch tensor), for searches of
        your own with rank()."""
        from PIL import Image, ImageOps

        rules = {**self.cfg["rules"], **(rules or {})}
        name = None
        if not isinstance(photo, Image.Image):
            name = Path(photo).stem
            photo = Image.open(photo)
        img = ImageOps.exif_transpose(photo).convert("RGB")
        # the photo is shrunk to work_px; small cards are cut from it as it was
        detail = None
        if rules["work_px"] and max(img.size) > rules["work_px"]:
            detail = img
            sc = rules["work_px"] / max(img.size)
            img = img.resize((round(img.width * sc), round(img.height * sc)), Image.LANCZOS)
        size = {"width": img.width, "height": img.height}
        if detail is not None:
            size["original"] = {"width": detail.width, "height": detail.height}
        fam_mask = self.family_mask(families)
        rnd = lambda q: [[round(float(x), 1), round(float(y), 1)] for x, y in np.asarray(q).tolist()]
        thr = self.cfg["detector"]["conf"]
        boxes = self.detect_boxes(img, rules)
        cut_off = [{"quad": rnd(q), "conf": round(c, 3)} for q, c, f in boxes if f and c >= thr]
        progress = on_progress or (lambda event: None)
        progress({"step": "boxes", **size, "cut_off": cut_off,
                  "boxes": [{"box": di, "quad": rnd(q), "conf": round(c, 3)}
                            for di, (q, c, f) in enumerate(boxes) if not f]})

        # stages 2 and 3 for every box worth naming: a low box cut off by the edge never is
        items = []
        for di, (quad, conf, edge) in enumerate(boxes):
            if edge:
                continue
            oe = self.orient_embed(img, quad)
            if oe is None:
                continue
            crop, q, feat, orient_p, sims = oe
            items.append({
                "box": di, "conf": round(conf, 3), "q": q, "crop": crop,
                # the first corner is the card's top left
                "quad": rnd(q),
                "feat": feat, "orient_p": orient_p,
                "back_sim": round(float(feat @ self.back_proto), 4) if self.back_proto is not None else None,
                "cands": self.retrieve(sims, fam_mask),
            })
            progress({"step": "pick", "box": di, "quad": items[-1]["quad"], "conf": items[-1]["conf"],
                      "candidates": items[-1]["cands"][:5]})
        dropped = self.left_out(items, rules) if rules["left_out"] else {}
        left_out = [{"quad": items[i]["quad"], "conf": items[i]["conf"], "reason": r,
                     "top": items[i]["cands"][0] if items[i]["cands"] else None}
                    for i, r in sorted(dropped.items())]
        items = [d for i, d in enumerate(items) if i not in dropped]
        if crops_dir is not None:
            Path(crops_dir).mkdir(parents=True, exist_ok=True)
            for d in items:
                d["crop"].save(Path(crops_dir) / f"{name or 'photo'}-{d['box']:02d}.jpg", quality=88)

        def card_head(d):
            card = {"box": d["box"], "quad": d["quad"], "conf": d["conf"]}
            if d["back_sim"] is not None:
                card["back_sim"] = d["back_sim"]
            if d["orient_p"] is not None:
                card["orient_p"] = d["orient_p"]
            return card

        if on_first is not None:
            on_first({**size, "first": True, "cut_off": cut_off, "left_out": left_out,
                      "cards": [{**card_head(d), "candidates": d["cands"][:5]} for d in items]})

        # cards on one page tend to share a set: the photo's best candidates' releases
        page_sets = {}
        for d in items:
            if d["cands"]:
                rel = d["cands"][0].get("release")
                if rel:
                    page_sets[rel] = page_sets.get(rel, 0) + 1

        # stages 4 to 9, a card at a time
        cards = []
        for d in items:
            q, cands = d["q"], d["cands"]
            # a card tiny in the photo is cut from the photo as it was given
            source, sq = img, q
            width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
            if detail is not None and rules["detail_px"] and width < rules["detail_px"]:
                source = detail
                sq = q * np.array([detail.width / img.width, detail.height / img.height], dtype=np.float32)
            hi = self.hires(source, sq)
            strip, number_read = (self.reader.strip_logits(hi), None) if self.reader is not None else (None, None)
            if strip is not None:
                number_read = self.reader.read_logits(strip)
            rgb = np.asarray(hi.convert("RGB"))
            d = {
                **d,
                "number_read": number_read,
                "pick": self.pick(hi, cands),
                "hi": hi, "source": source, "source_quad": sq, "detail": source is not img,
                "nfit": self.number_fit(strip, cands, number_read),
                "stamp": marks.edition_stamp(rgb)["score"],
                "ecard": marks.ecard_reverse(rgb),
            }
            nr = d["number_read"]
            own_top = d["cands"][0].get("release") if d["cands"] else None
            pick = d["pick"]
            nfit = d["nfit"]
            # the verifier compares the prints of the card that wins on
            # everything else, so it runs on the fused order
            held, unverified = {}, []
            ver = self.verify(d["hi"], self.fuse(d["cands"], nr, page_sets, own_top, pick, nfit=nfit, scores_out=unverified), keep=held)
            scores = []
            ordered, feats = self.fuse(d["cands"], nr, page_sets, own_top, pick, with_features=True, ver=ver,
                                       nfit=nfit, scores_out=scores)
            ordered, settled = self.settle(d, ordered, feats, scores, ver, held.get("prep"), rules)
            fused = {id(c): sc for c, sc in zip(d["cands"], scores)}
            quality = match_quality(ordered, [fused.get(id(c), c["sim"]) for c in ordered],
                                    None not in (d["back_sim"], rules["back_sim"]) and d["back_sim"] >= rules["back_sim"],
                                    settled.get("likely_wrong"), set(settled.get("ruled_out", ())))
            # the verifier's prints within reach before it ran, one of them
            # without a score (its picture wasn't to be had): verifyMissing
            before = {id(c): sc for c, sc in zip(d["cands"], unverified)}
            reach = max(unverified) if unverified else 0.0
            close = [c for c in held.get("siblings") or [] if reach - before.get(id(c), -9.0) < SURE_FUSED]
            missing = len(close) >= 2 and any((ver or {}).get(c.get("print")) is None for c in close)
            decided = is_decided(quality, ordered, [fused.get(id(c), c["sim"]) for c in ordered], missing)
            card = {**card_head(d),
                    "number_read": ({k: nr.get(k) for k in ("text", "votes", "n", "d", "code")} if nr else None),
                    "candidates": ordered[:5]}
            if evidence:
                card["evidence"] = feats
            if fingerprints:
                card["fingerprint"] = d["feat"]
            if pick:
                card["pick"] = {"print": max(pick, key=pick.get),
                                "margin": round(max(pick.values()) - sorted(pick.values())[-2], 4)}
            if ver:
                card["verify"] = ver
            if nfit:
                card["numfit"] = nfit
            card["edition_stamp"] = d["stamp"]
            card["ecard_reverse"] = d["ecard"]["score"]
            card["ecard_reading"] = marks.reading(d["ecard"])
            if d["detail"]:
                card["detail"] = True
            card.update(settled)
            card["quality"] = quality
            if decided:
                card["decided"] = True
            progress({"step": "card", "index": len(cards), "card": card})
            cards.append(card)
        self._grays = {}
        return {**size, "cards": cards, "cut_off": cut_off, "left_out": left_out}
