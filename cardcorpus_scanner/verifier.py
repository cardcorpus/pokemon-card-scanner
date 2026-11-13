"""Part of stage 7: which print of one painting is this card?

The matcher learns paintings, so Base Set, Base Set 2 and Legendary
Collection Charizard (or a holo and its non-holo twin) look the same to it.
Those prints differ in small places: the set symbol, the collector number,
the copyright line, the foil. This finds those places from the prints'
reference pictures and checks which picture the card agrees with there:

1. align: SIFT keypoints on the card and on each print's picture, ratio-test
   matching and a MAGSAC homography. Everything is resampled into the frame
   of one picture (the anchor), so a point on the card is the same pixel in
   every image however loosely the card was boxed.
2. normalise: grey, local contrast normalisation (removes lighting, white
   balance and print colour), and the card resampled through the pictures'
   resolution so both sides carry the same detail.
3. compare per pair of prints (a, b): the difference map |a - b| marks where
   the two pictures differ; the score is how much closer the card is to a
   than to b there: +1 = looks like a, -1 = like b, 0 = no difference or no
   evidence. The art window is left out when either print is holo.
4. per print: its worst pairwise score; -1 for a print whose picture doesn't
   align while a sibling's does (other art).

No training. The reference pictures come from a local folder, a disk cache
or the image mirror.
"""
import os
import threading
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

W, H = 480, 670   # working frame
REF_PX = 320      # the card is resampled through this width, as the pictures were compared at it
MIRROR = "https://img.cardcorpus.com"


def _cv2():
    import cv2

    return cv2


def _resample(gray, src_px=REF_PX):
    """Grey uint8 image -> (W, H) float32 in [0, 1] carrying src_px of detail."""
    cv2 = _cv2()
    h, w = gray.shape[:2]
    if w > src_px:
        gray = cv2.resize(gray, (src_px, round(src_px * h / w)), interpolation=cv2.INTER_AREA)
    return cv2.resize(gray, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0


def _norm(g, sigma=6.0, eps=1e-3):
    """Local contrast normalisation: structure without lighting or colour."""
    cv2 = _cv2()
    d = g - cv2.GaussianBlur(g, (0, 0), sigma)
    return d / np.sqrt(cv2.GaussianBlur(d * d, (0, 0), sigma) + eps)


def _sane(Hm, size=(W, H)):
    """The mapped frame must stay a convex quad of plausible size. size:
    the (width, height) of the frame mapped, the working frame unless said
    otherwise."""
    cv2 = _cv2()
    sw, sh = size
    c = cv2.perspectiveTransform(np.float32([[[0, 0]], [[sw, 0]], [[sw, sh]], [[0, sh]]]), Hm).reshape(4, 2)
    area = cv2.contourArea(c)
    return cv2.isContourConvex(c.astype(np.float32)) and 0.25 * W * H < area < 4 * W * H


def card_region(gray, quad, grow):
    """The card's region of the photo (grey uint8), grown by `grow` about its
    middle and straightened into a frame `grow` times the working frame,
    carrying REF_PX of detail across the card. Returns (region uint8, the
    homography from photo pixels to it)."""
    cv2 = _cv2()
    quad = np.float32(quad).reshape(4, 2)
    middle = quad.mean(axis=0, dtype=np.float32)
    grown = (middle + (quad - middle) * np.float32(grow)).astype(np.float32)
    fw, fh = int(round(W * grow)), int(round(H * grow))
    # drawn at twice the detail wanted, then area-averaged down to it, so a
    # large photo doesn't alias
    dw, dh = int(round(REF_PX * grow)), int(round(REF_PX * grow * H / W))
    fine = np.float32([[0, 0], [2 * dw, 0], [2 * dw, 2 * dh], [0, 2 * dh]])
    drawn = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(grown, fine), (2 * dw, 2 * dh), flags=cv2.INTER_LINEAR)
    small = cv2.resize(drawn, (dw, dh), interpolation=cv2.INTER_AREA)
    region = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_CUBIC)
    frame = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])
    return region, cv2.getPerspectiveTransform(grown, frame)


class Pictures:
    """Reference pictures by picture id: memory -> local folder (files named
    by id) -> disk cache -> the image mirror. copy_width fetches the mirror's
    sized copy (240, 480 or 960 px wide) in place of the original."""

    def __init__(self, base_url=MIRROR, cache_dir=None, folder=None, keep=256, copy_width=None, user_agent=None):
        self.base = (base_url or "").rstrip("/")
        self.copy_width = copy_width
        self.cache = Path(cache_dir) if cache_dir else None
        self.folder = Path(folder) if folder else None
        self.user_agent = user_agent or "cardcorpus-scanner"
        self.mem, self.keep, self.lock = OrderedDict(), keep, threading.Lock()

    def _path(self, picture):
        return f"s1/{self.copy_width}/{picture}.avif" if self.copy_width else f"m/{picture}"

    def _bytes(self, picture):
        path = self._path(picture)
        flat = path.replace("/", "_")
        for d, name in ((self.folder, picture), (self.cache, flat)):
            if d is not None and (d / name).exists():
                return (d / name).read_bytes()
        if not self.base:
            return None
        try:
            req = urllib.request.Request(f"{self.base}/{path}", headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = r.read()
        except Exception:
            return None
        if self.cache is not None:
            tmp = self.cache / f"{flat}.{os.getpid()}.{threading.get_ident()}.part"
            try:
                self.cache.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(data)
                tmp.replace(self.cache / flat)
            except OSError:
                tmp.unlink(missing_ok=True)
        return data

    def get(self, picture):
        """Grey uint8 array (native size) or None."""
        cv2 = _cv2()
        with self.lock:
            if picture in self.mem:
                self.mem.move_to_end(picture)
                return self.mem[picture]
        data = self._bytes(picture)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE) if data else None
        with self.lock:
            self.mem[picture] = img
            while len(self.mem) > self.keep:
                self.mem.popitem(last=False)
        return img


class Verifier:
    def __init__(self, pictures, sigma_d=2.5, tau=0.3, min_mass=5e-3, border=0.05, glare=0.96,
                 art_box=(0.07, 0.09, 0.93, 0.54), art_keep=0.25):
        self.pictures = pictures
        self.cfg = {"sigma_d": sigma_d, "tau": tau, "min_mass": min_mass, "border": border}
        self.glare, self.art_box = glare, tuple(art_box)
        self.art_keep = art_keep
        self.local = threading.local()  # one SIFT detector per thread
        self.feats, self.lock = OrderedDict(), threading.Lock()
        self.pool = ThreadPoolExecutor(4)

    def _features(self, g01, cache_key=None):
        cv2 = _cv2()
        if cache_key is not None:
            with self.lock:
                if cache_key in self.feats:
                    self.feats.move_to_end(cache_key)
                    return self.feats[cache_key]
        if not hasattr(self.local, "sift"):
            self.local.sift = cv2.SIFT_create(nfeatures=3000)
        kp, desc = self.local.sift.detectAndCompute((g01 * 255).astype(np.uint8), None)
        f = (np.float32([k.pt for k in kp]), desc)
        if cache_key is not None:
            with self.lock:
                self.feats[cache_key] = f
                while len(self.feats) > 256:
                    self.feats.popitem(last=False)
        return f

    @staticmethod
    def align(fa, fb, ratio=0.75, size=(W, H), stats=None):
        """Homography taking image b's pixels into image a's frame, and its
        inlier count; (None, n) when the two don't align. size: b's frame,
        for the sane check. stats, a dict, gets the pairs past the ratio
        test, the inliers and whether the mapped frame was sane."""
        cv2 = _cv2()
        (pa, da), (pb, db) = fa, fb
        if da is None or db is None or len(pa) < 8 or len(pb) < 8:
            return None, 0
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
        if stats is not None:
            stats.update(matches=len(good), inliers=0, sane=False)
        if len(good) < 12:
            return None, len(good)
        src = pb[[m.queryIdx for m in good]]
        dst = pa[[m.trainIdx for m in good]]
        Hm, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 4.0, maxIters=5000, confidence=0.999)
        n = int(mask.sum()) if mask is not None else 0
        sane = Hm is not None and _sane(Hm, size)
        if stats is not None:
            stats.update(inliers=n, sane=bool(sane))
        if Hm is None or n < 15 or not sane:
            return None, n
        return Hm, n

    # The art window on classic layouts, in a reference picture's frame,
    # ending just above the set symbol under the art. When either print of a
    # pair is holo it is left out of their comparison: a holo picture shows
    # the foil as a dark speckled texture over the art, while a photo shows
    # the painted background with a sheen, so the art would vote for
    # whichever picture renders foil the photo's way, not for the print.
    # Two pictures rendered alike differ over only a few spots of it: when
    # they differ over less than art_keep of it, it is compared like the
    # rest of the card.

    def pair(self, nP, nA, nB, valid, cfg, holo=False):
        """+1: the card agrees with A where A and B differ; -1: with B."""
        cv2 = _cv2()
        sd = cfg["sigma_d"]
        d = cv2.GaussianBlur(np.abs(nA - nB), (0, 0), sd)
        if holo:
            x0, y0, x1, y1 = self.art_box
            rows, cols = slice(int(y0 * H), int(y1 * H)), slice(int(x0 * W), int(x1 * W))
            inside = valid[rows, cols] > 0
            differing = float((d[rows, cols][inside] > cfg["tau"]).sum()) / max(int(inside.sum()), 1)
            if not (inside.any() and differing < self.art_keep):
                valid = valid.copy()
                valid[rows, cols] = 0
        w = np.clip(d - cfg["tau"], 0, None) * valid
        mass = float(w.sum()) / (W * H)
        if mass < cfg["min_mass"]:
            return 0.0, mass
        ea = cv2.GaussianBlur((nP - nA) ** 2, (0, 0), sd)
        eb = cv2.GaussianBlur((nP - nB) ** 2, (0, 0), sd)
        return float((w * (eb - ea)).sum() / max((w * (ea + eb)).sum(), 1e-9)), mass

    def prepare(self, card_rgb, pictures, holo=()):
        """Align the card and the pictures into one frame. card_rgb: the
        straightened card (any size); pictures: {print: picture id}, best
        candidate first (the anchor); holo: the prints that are holo."""
        cv2 = _cv2()
        ids = list(pictures)
        grays = dict(zip(ids, self.pool.map(lambda p: self.pictures.get(pictures[p]), ids)))
        ids = [p for p in ids if grays[p] is not None]
        if len(ids) < 2:
            return None
        g = card_rgb if card_rgb.ndim == 2 else cv2.cvtColor(card_rgb, cv2.COLOR_RGB2GRAY)
        photo = _resample(g)
        fp = self._features(photo)
        refs = {p: _resample(grays[p]) for p in ids}
        fr = dict(zip(ids, self.pool.map(lambda p: self._features(refs[p], pictures[p]), ids)))
        anchor, Hp, n_in, failed, st = None, None, 0, [], {}
        for p in ids:
            st = {}
            Hp, n_in = self.align(fr[p], fp, stats=st)
            if Hp is not None:
                anchor = p
                break
            failed.append(p)
        if anchor is None:
            return None
        ones = np.ones((H, W), np.float32)
        warp = lambda im, Hm: cv2.warpPerspective(im, Hm, (W, H), flags=cv2.INTER_LINEAR)
        P = warp(photo, Hp)
        valid = warp(ones, Hp) > 0.99
        valid &= ~(cv2.dilate((P > self.glare).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
        aligned = {anchor: refs[anchor]}
        for p in ids:
            if p == anchor:
                continue
            Hb, _ = self.align(fr[anchor], fr[p])
            if Hb is None:
                failed.append(p)
                continue
            aligned[p] = warp(refs[p], Hb)
            valid &= warp(ones, Hb) > 0.99
        if len(aligned) < 2 and not failed:
            return None
        return {"anchor": anchor, "inliers": n_in, "photo": P, "refs": aligned, "valid": valid.astype(np.float32),
                "holo": set(holo), "failed": failed, "H": Hp, "anchor_picture": pictures[anchor], "anchor_stats": st,
                "features": fp, "size": (g.shape[1], g.shape[0])}

    def compare(self, prep, **cfg):
        """{print: worst pairwise score} and the pair details."""
        cfg = {**self.cfg, **cfg}
        bx, by = int(cfg["border"] * W), int(cfg["border"] * H)
        valid = np.zeros((H, W), np.float32)
        valid[by:H - by, bx:W - bx] = prep["valid"][by:H - by, bx:W - bx]
        nP = _norm(prep["photo"])
        normed = {p: _norm(g) for p, g in prep["refs"].items()}
        live = list(normed)
        sums, pairs = {p: [] for p in live}, {}
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                holo = a in prep["holo"] or b in prep["holo"]
                s, mass = self.pair(nP, normed[a], normed[b], valid, cfg, holo)
                sums[a].append(s)
                sums[b].append(-s)
                pairs[(a, b)] = (round(s, 4), round(mass, 5))
        # a print scores its worst head-to-head comparison: averaging let a
        # third print decide between two others
        out = {p: round(float(min(v)), 4) if v else 0.0 for p, v in sums.items()}
        out.update({p: -1.0 for p in prep["failed"]})
        return out, pairs

    def score(self, card_rgb, pictures, holo=(), keep=None):
        """{print: score in [-1, 1]} for the prints that could be compared,
        or None. keep, a dict, gets the preparation ("prep") for check() to
        reuse."""
        prep = self.prepare(card_rgb, pictures, holo)
        if keep is not None:
            keep["prep"] = prep
        if prep is None:
            return None
        return self.compare(prep)[0]

    # --- the check of a pick against its own picture -------------------------

    OFF_BOX = 0.25      # lined up, but the box this far off its picture at the worst corner (card widths)
    TELLING = 500       # picture features a failure to line up needs to count
    NEAR_FIT = 8        # inliers under this in every pass: nowhere near a fit
    SOLID_FIT = 60      # inliers a pick needs for another's share of them to count
    WELL_PLACED = 0.12  # lined up right under the box

    def check(self, card_rgb, picture, photo_gray=None, quad=None, prep=None):
        """Whether the straightened card lines up with a picture: from the
        anchor's homography when it is the anchor's picture (prep, from
        score(keep=...)), else by matching the card's features with the
        picture's, and failing that on a region of the photo 1.4 times wider
        than the box (photo_gray, grey uint8, with the box `quad` in its
        pixels, the card's top left first). Returns {"aligned", "pass" (1 or
        2, 0 for none), "inliers", "scan_features", "passes": [{features,
        matches, inliers, sane}], "box_error": how far the box's worst corner
        is from the card's, in card widths}, or None when the picture can't
        be had."""
        cv2 = _cv2()
        gray = self.pictures.get(picture)
        if gray is None:
            return None
        fr = self._features(_resample(gray), picture)
        out = {"aligned": False, "pass": 0, "inliers": 0, "scan_features": len(fr[0]), "passes": [],
               "box_error": None}
        g = card_rgb if card_rgb.ndim == 2 else cv2.cvtColor(card_rgb, cv2.COLOR_RGB2GRAY)
        ch, cw = g.shape[:2]
        frame_corners = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
        corners = np.float32(quad).reshape(4, 2) if quad is not None else frame_corners
        to_frame = np.diag([W / cw, H / ch, 1.0]) @ cv2.getPerspectiveTransform(corners, frame_corners)
        M = None
        if prep is not None and prep.get("anchor_picture") == picture and prep.get("H") is not None:
            st = prep.get("anchor_stats") or {}
            M = prep["H"] @ to_frame
            out["inliers"] = prep["inliers"]
            out["passes"].append({"features": len(prep["features"][0]), "matches": st.get("matches", 0),
                                  "inliers": prep["inliers"], "sane": True})
            out["pass"] = 1
        else:
            feats = prep["features"] if prep is not None and prep.get("size") == (cw, ch) else \
                self._features(_resample(g))
            st = {}
            Hm, n = self.align(fr, feats, stats=st)
            out["passes"].append({"features": len(feats[0]), "matches": st.get("matches", 0),
                                  "inliers": st.get("inliers", 0), "sane": st.get("sane", False)})
            if Hm is not None:
                M = Hm @ to_frame
                out["pass"], out["inliers"] = 1, n
            elif photo_gray is not None and quad is not None:
                # a box that cuts the card off: a region 1.4 times wider
                region, to_region = card_region(photo_gray, corners, 1.4)
                if not hasattr(self.local, "sift"):
                    self.local.sift = cv2.SIFT_create(nfeatures=3000)
                kp, desc = self.local.sift.detectAndCompute(region, None)
                fg = (np.float32([k.pt for k in kp]), desc)
                wide = {}
                Hm, n = self.align(fr, fg, size=(region.shape[1], region.shape[0]), stats=wide)
                out["passes"].append({"features": len(fg[0]), "matches": wide.get("matches", 0),
                                      "inliers": wide.get("inliers", 0), "sane": wide.get("sane", False)})
                if Hm is not None:
                    M = Hm @ to_region
                    out["pass"], out["inliers"] = 2, n
        if M is None:
            return out
        out["aligned"] = True
        boxed = cv2.perspectiveTransform(corners.reshape(1, 4, 2).astype(np.float32), M).reshape(4, 2)
        worst = 0.0
        for (x, y), (cx, cy) in zip(boxed / np.float32([W, H]), ((0, 0), (1, 0), (1, 1), (0, 1))):
            # in card widths: a card is 670 / 480 as tall as it is wide
            worst = max(worst, float(np.hypot(x - cx, (y - cy) * 670 / 480)))
        out["box_error"] = worst
        return out

    @classmethod
    def off_box(cls, a):
        """Lined up, but with the box at least a quarter of a card's width
        off its picture: the picture sits beside the box. A card upside down
        in its box is off by more than a card's width and isn't counted."""
        return bool(a and a["aligned"] and a["box_error"] is not None and cls.OFF_BOX < a["box_error"] < 1)

    @classmethod
    def plainly_other(cls, other, pick):
        """Whether another card, checked beside a pick that lined up as
        `pick`, plainly isn't this picture: its picture had detail enough
        and no pass came near a fit, or the pick rests on many points and
        this one found at most a quarter as many, lined up or not (cards of
        one era share a frame, text box and Energy symbols)."""
        if not other or other["scan_features"] < cls.TELLING or not other["passes"]:
            return False
        if not other["aligned"] and all(x["inliers"] < cls.NEAR_FIT for x in other["passes"]):
            return True
        best = max(x["inliers"] for x in other["passes"])
        return pick["inliers"] >= cls.SOLID_FIT and best * 4 <= pick["inliers"]

    @classmethod
    def gives_way(cls, pick, best):
        """Whether a pick checked against its picture gives way to the best
        match by picture, checked in turn: when the pick doesn't line up and
        the best does, or when the pick lines up only well off the box and
        the best lines up right under it."""
        if not best or not best["aligned"]:
            return False
        if not pick["aligned"]:
            return True
        return cls.off_box(pick) and (best["box_error"] if best["box_error"] is not None else 1) < cls.WELL_PLACED
