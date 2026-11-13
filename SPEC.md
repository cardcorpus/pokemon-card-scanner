# How the scanner works, with every number

What `Scanner.scan` does to a photo, stage by stage. Every setting named
`section.key` is in the weights' `pipeline.json`, which the scanner reads;
the values here are this release's. Code: `cardcorpus_scanner/scanner.py`
and the module each stage names.

The rules around the stages (which boxes are named, the work size, the
check of the pick against its own picture in stage 7) are in
`pipeline.json` under `rules`; `scan(..., rules={...})` changes any
of them for one scan, and each is described where it acts below.

The photo is decoded with its EXIF orientation applied, then shrunk to
3000 px on its longer side (Lanczos, `rules.work_px`) when it is larger;
every stage works on that copy, and corners are in its pixels (the
result's `width` and `height`; `original` is the photo's own size). The
photo as given is kept for small cards (stage 4).

## Weights folder

| file | |
|---|---|
| `pipeline.json` | every setting below |
| `models/detector.safetensors` | QuadDet; its shape is `detector.model` |
| `models/matcher.safetensors` | the matcher tower; its spec is `embedder` |
| `models/heads.safetensors` | `orient_hidden` (256 x 513), `orient_probe` (4 x 257), `back_prototype` (512) |
| `models/picker.safetensors` | the print picker tower; its spec is `embedder` with `picker.backbone` over it |
| `models/number-reader.safetensors`, `.json` | the CRNN; its charset, strip size, strip places (`slots`) and preferred set sizes (`denominators`) |
| `index/` | see `cardcorpus_scanner/index.py` |

## 1. Detect (`detector.py`)

- The photo scaled so its longer side is 1280 (sizes rounded half to even)
  by OpenCV's bilinear rule, placed at the top left of the smallest canvas
  that holds it (960 x 1280, 1280 x 960 or 1280 x 1280), grey (114)
  elsewhere; RGB in [0, 1].
- Output on a grid at a quarter of the canvas (stride 4), 15 channels: the
  card-centre score (zero unless the cell is its 3 x 3 neighbourhood's
  peak), the centre's offset in its cell (2), log size (1), the four
  corners relative to the centre in units of size (8), and a corner
  heatmap with its own offsets (3).
- Cards: every centre peak above `detector.conf` (0.20), most confident
  first, at most 300; size = exp(clamp(log size, -4, 8)) x 4. Each corner
  snaps to the most confident corner peak (score > 0.1) within 0.08 x size
  of it, if any. Corners go back to photo pixels by the scale on each axis.
- Containment: every box rasterised at 400 px on the photo's longer side;
  walking boxes by confidence, a box covered 75% or more
  (`detector.contain`) by an already-kept box is dropped (a card's art
  window can look like a card).
- Unsure boxes: the detector runs down to 0.05 (`rules.rescue_conf`), so
  that a box it is unsure of can still be named when its picture plainly
  matches (stage 3, "Left out").
- Cut off by the edge: a box touches the photo's edge when a corner lies
  within 0.005 x the photo's longer side of it. A box that touches the edge
  and whose area is under 0.6 of the median area of the boxes clear of the
  edge (of all boxes, when none is clear) is a sliver of a card mostly out
  of frame. The median is taken over the boxes at or over `detector.conf`;
  each lower box is judged against those. A sliver is not matched: one at
  or over `detector.conf` is listed under `cut_off` with its corners, a
  lower one is dropped silently. `rules={"cut_off": False}`
  (`--keep-cut-off`) matches slivers like any other card. Cards left out
  don't count towards the page's sets (stage 7).

## 2. Orient and fingerprint (`scanner.py`, `tower.py`)

- The 4 corners ordered by angle around their centroid
  (`atan2(y - cy, x - cx)` ascending). The 4 rolls of that order are the 4
  ways the card can be upright.
- Each roll straightened to 384 x 536 (perspective warp, bicubic): corner
  i to (0, 0), (384, 0), (384, 536), (0, 536).
- Matcher: the shorter side resized to 224 (bilinear, antialiased), the
  centre 224 x 224 square, RGB in [0, 1], ImageNet mean and std; DINOv3
  ViT-S/16 (pooled class token, 384) and a linear map to 512; L2
  normalised. Its attention clamps q.k to +-64,000 (`embedder.logit_limit`,
  see `tower.clamped_attention`), which changes nothing in 32 bits and
  keeps 16-bit runtimes from overflowing.
- Orientation head: h = max(0, f V[:, :512]^T + V[:, 512]) with V =
  `orient_hidden`, logits = h W[:, :256]^T + W[:, 256] with W =
  `orient_probe`, softmax over the 4 rows (up, right, down, left); P(upright)
  is column 0. The roll with the highest P(upright) is kept (`orient_p`),
  unless another roll's best index similarity (4 decimals) beats that
  roll's by `orientation.match_margin` (0.08) and reaches
  `orientation.match_floor` (0.90): a card printed along its long edge
  reads upright to the head when turned a quarter.
- Half turn: when the head's roll was kept and the box doesn't touch the
  photo's edge (within 0.5% of its longer side), the roll turned 180° wins
  if its best index match is the same card name, 0.02 more similar, and
  at least 0.80.
- `back_sim` = fingerprint . `back_prototype`.

## 3. Retrieve

- Cosine similarity to every index row (float16, held as float32). A
  language filter (`families`) sets other families' rows to -2.
- The top 20 rows (`retrieval.top`), best first: a picture with no card of
  its own (a second view) stands for the first card with one in its
  picture group (`dupe-groups.json`); one candidate per print (else per
  name, set, number and family) until 10 (`retrieval.candidates`). Each
  candidate keeps its picture's similarity `sim`, to 4 decimals.

### Left out

After stage 3, walking the boxes by confidence (`rules.left_out`):

- A box at or over `detector.conf` is junk, and dropped, when it is under
  0.35 (`rules.junk_conf`), its best candidate's `sim` is under 0.7
  (`rules.junk_sim`) and it isn't a card back (`back_sim` under 0.60,
  `rules.back_sim`). Else it is kept.
- A box under `detector.conf` is kept only when its best `sim` is 0.80 or
  more (`rules.rescue_sim`) and it covers, or is covered by, no box already
  kept by half its area or more (`rules.rescue_overlap`; the overlap of the
  two convex boxes over the one's own area). Else it is dropped as unsure.

Dropped boxes are listed under `left_out` with their corners, confidence,
reason (`junk` or `unsure`) and best match (`top`), and are not matched
further; `crops_dir` saves a crop only for the cards kept. The first
picks (`scan(..., on_first=...)`) are sent after this, with each kept
card's candidates in retrieval order. With `scan(..., on_progress=...)`
the boxes are sent as soon as they are found, and each box's first pick
as it is fingerprinted; stages 4 to 9 then run a card at a time, and each
card is sent the moment it is settled, as it will be in the answer.

## 4. Straighten at full resolution

The oriented corners straightened again at 960 x 1340, bicubic, for the
number reader, the picker, the verifier and the marks. From the 3000 px
copy, except for a card under 320 px wide in it (the mean of its top and
bottom edges, `rules.detail_px`), which is straightened from the photo as
given, its corners scaled to it; such a card has `"detail": true`.

## 5. Read the number (`number_reader.py`)

- Strips: for each of the 6 `slots` (centre and size as card fractions),
  each dy in {-0.025, -0.012, 0, 0.012, 0.025}, dx in {-0.04, 0, 0.04},
  scale in {0.92, 1.12}: a box w x 1.55 x scale by h x 2.1 x scale around
  (cx + dx, cy + dy) cut from the grey card, edges truncated to whole
  pixels: 180 strips.
- Each strip resized to height 48 keeping its aspect (width 16 to 384,
  bicubic), padded right to 384 with 0.5, values in [0, 1]. CRNN: 4 conv
  blocks (32, 64, 128, 128 channels, batch norm and ReLU; pools 2, 2,
  (2, 1), (2, 1)), a linear 384 to 192, a 2-layer bidirectional LSTM (128),
  a linear to 41 classes (CTC blank and `0123456789/No.-ABCDEFGHIJKLMOPQRSTUVWXYZ`),
  96 frames.
- Free read: greedy CTC decode per strip; a parse is n/d (`\d{1,3}/\d{1,3}`,
  1 <= n <= d + 60, 10 <= d <= 300) or a letter code
  (`^[0-9A-Z][0-9A-Z/.\-]{1,11}$` with a digit, 3 characters or more). Votes
  per parse; the winner by votes, + 6 when its d is in `denominators`
  (set sizes printed on 5 or more cards), + 2 for an n/d; ties by the best
  strip's confidence. Output `text`, `n`, `d`, `code`, `votes`; it counts
  in fusion only with 5 votes or more (`number.votes_gate`).
- Constrained read (`numfit`): for each candidate whose printed number is
  an n/d, the CTC log-likelihood of the strip showing it, best over the
  180 strips and its spellings (as printed, unpadded, numerator padded to
  the denominator's width), 3 decimals. Needs 2 different numbers among
  the candidates. The free read's own n/d is scored beside them when it is
  none of theirs.

## 6. Pick the print

Only when the best candidate is in a picker group of 2 prints or more
(`index/picker-rows.json`): the full-resolution card resized to 448 x 320
(h x w, bicubic), RGB in [0, 1], mean and std 0.5; PE-Core S/16 with its
own projection to 512; L2 normalised. Each print of the group scores its
best cosine over its gallery rows.

## 7. Fuse and verify

Page evidence: each card's first candidate's release, counted over the
photo. Each candidate's features (absent = 0):

| feature | value |
|---|---|
| `num` | 1 when the free read's n/d equals the candidate's number with 5 votes or more |
| `code` | 1 when there is no n/d read, a code read with 5 votes or more, and the candidate's number equals it (letters and digits only, upper case) |
| `set` | 1 when another card on the photo has the candidate's release first |
| `twin` | 1 for the earliest release among 2 or more candidates of one painting within 0.02 of that painting's best `sim` (`twin.sim_gap`) |
| `pick` | picker score less the best picker score (<= 0) |
| `verify` | verifier score less the best (<= 0), when 2 candidates or more have one |
| `numfit` | when the best constrained log-likelihood among the candidates' numbers is -5 or more (`number.numfit_gate`): (ll - best) / 5 (`numfit_scale`), floored at -1; -1 for a candidate with no printed n/d. Absent for all when the free read is an n/d with 10 votes or more that no candidate has and that fits the strips 1 or more better than every candidate's (`numfit_abstain_votes`, `numfit_abstain_margin`): the strip shows another card's number |
| `prior` | -1 for a print a scan should name only on clear evidence (a World Championships deck's reprint of a booster card), from the index |

Score = `sim` + sum(weight x feature), weights `fusion`: num 0.2, code
0.04, set 0.03, twin 0.008, pick 0.1, verify 0.1, numfit 0.2, prior 0.02.
Candidates are sorted by score, ties keeping retrieval order; the first is
the pick. The verifier runs between two fusions: fuse without it, verify
the fused order, fuse again with it.

**Print verifier** (`verifier.py`). Siblings: candidates that share the
best candidate's picture group, painting or name (case-insensitive), the
first 5 in fused order (`verifier.max_prints`). Each one's reference
picture is its own picture in the dataset, or the one
`index/verifier-pictures.json` names. Pictures come from a local folder,
the disk cache or the image mirror (`https://img.cardcorpus.com/m/<id>`,
the original); a sibling whose picture can't be fetched is left out.

1. Working frame 480 x 670. The card and every picture resampled through
   320 px wide (area-downscaled to 320 wide, then bicubic up), grey, [0, 1].
2. SIFT (OpenCV, 3,000 features). Each card descriptor's 2 nearest among
   the picture's (brute force, L2), kept when nearest < 0.75 x second;
   12 matches or more; homography by MAGSAC (4 px, 5,000 iterations,
   confidence 0.999), accepted with 15 inliers or more and a convex mapped
   frame of 0.25 to 4 times the frame's area. The anchor is the first
   sibling the card aligns to; the card and every other picture are warped
   into its frame. A sibling that won't align scores -1.
3. Valid: inside every warp, not glare (card pixel > 0.96, dilated 9 x 9),
   not within 5% of the edge (`verifier.border`).
4. Local contrast normalisation: d = g - G6(g), n = d / sqrt(G6(d²) + 1e-3).
5. Each pair (a, b): D = G2.5(|n_a - n_b|), w = max(D - 0.3, 0) x valid
   (`tau`). When either print is holo (its default finish a holo other than
   reverse, else its rarity says holo), the art window (x 7-93%, y 9-54%,
   `art_box`) is left out of w, unless D > 0.3 over less than a quarter of
   the window: the pictures are then alike and differ by a mark printed on
   the art. Mass = sum(w) / (480 x 670); under 0.005 (`min_mass`) the pair
   scores 0. Else e_a = G2.5((n_card - n_a)²), e_b likewise, and
   s_ab = sum(w (e_b - e_a)) / sum(w (e_a + e_b)), in [-1, 1].
6. A print's score is its worst pairwise score, 4 decimals.

**The pick's own picture** (`rules.realigns`, `rules.verifier_yields`;
needs the verifier). After the second fusion the pick is checked against
its print's own picture: whether the card lines up with it, and where.

1. When the picture is the verifier's anchor, its homography. Else the
   card (from stage 4, at 480 x 670, through 320 px as in step 1) and the
   picture are matched as in step 2 (pass 1).
2. Failing that, a region of the photo around the box, grown 1.4 times
   about its centre (the box may cut the card off): warped (bilinear) to
   twice 320 x 1.4 px wide, area-downscaled to 320 x 1.4 wide, then bicubic
   to 480 x 1.4 by 670 x 1.4; SIFT on it as it is (grey, 8 bits), matched
   as in step 2 with the region's size for the area check (pass 2).
3. When it lines up, `box_error` is how far the box's worst corner lands
   from the picture's, in card widths (heights scaled by 670 / 480).

Then:

- Realign: when the pick isn't the best candidate by picture and either
  doesn't line up or lines up off its box (`box_error` over 0.25 and under
  1), the best by picture is checked too; it becomes the pick
  (`"realigned": true`) when it lines up and either the pick didn't or it
  lines up right under the box (`box_error` under 0.12).
- Verifier yields: otherwise, when the first two in fused order are within
  0.03 by score (`rules.twin_tie`), share a name (without a bracketed note,
  accents, capitals or apostrophes), the second has the higher `sim` and
  the higher `pick` feature, and the verifier scored both and put the
  first ahead, they swap (`"yielded": true`).
- The final pick's check is reported as `alignment` (`aligned`, `pass`,
  `inliers`, `scan_features`, `box_error`). `"likely_wrong": true` when it
  didn't line up though both had detail enough: 500 picture features or
  more, 650 card features or more in pass 1, and under 8 inliers in every
  pass.
- Ruled out (`rules.rules_out`): when the pick is in doubt (its `sim` under
  0.88, or another candidate's within 0.06 of it) and lines up, up to 4
  other candidates of another name within 0.15 of it by fused score are
  checked against their own pictures, best first. One is ruled out
  (`ruled_out`) when its picture has 500 features or more and either no
  pass came within 8 inliers, or the pick rests on 60 inliers or more and
  it found a quarter as many or fewer.

**How sure** (`quality`):

- `back` when `back_sim` is 0.60 or more; else `likely_wrong` as above.
- `unsure` when the pick's `sim` is under 0.88.
- `sure` when no other candidate's `sim` is within 0.06 of the pick's, or
  none not ruled out is within 0.15 of it by fused score.
- Else `printing_unclear` when every such close candidate shares the
  pick's name, `unsure` when not.

`"decided": true` marks a `printing_unclear` or `unsure` pick that is
taken as the card without asking: it leads the next by 0.05 or more by
fused score, and the candidates within 0.15 of it carry one name
(`printing_unclear`) or at most two (`unsure`). A doubt only between
prints of one card is not decided when two or more of them are within
0.15 of the best before the verifier and one of those has no verifier
score.

## 8. 1st Edition stamp (`marks.py`)

Wizards of the Coast's English sets (Base Set to Neo Destiny) print the
stamp below the picture on the left on a Pokémon, by the bottom left
corner on a Trainer, top right on an Energy.

- The full-resolution card grey (OpenCV), resized to 600 x 825 (area).
- Template `edition-stamp.png` (48 x 36, the mean of 137 stamps); mask: its
  pixels under 110, dilated 5 x 5.
- Normalised cross-correlation with the mask in three regions (x0-x1,
  y0-y1): Pokémon 0-110, 355-510 at template scales 0.9, 1.0, 1.1; Trainer
  60-180, 700-800 at 0.6 to 0.9; Energy 450-595, 15-125 at 0.6 to 1.0.
- `edition_stamp` = the best value, 4 decimals; 0.60 or more is a stamp.

## 9. e-Card reverse holo (`marks.py`)

Expedition, Aquapolis and Skyridge print the e-Reader dot code along the
bottom and down the left edge of a standard copy, and evenly spaced holo
bars there on a reverse holo (and on every holo rare).

- The card grey, 600 x 825 (area).
- Bottom strip: x 260-540, bands 30 rows high at y = 735, 738 ... 795;
  left strip: y 80-720, bands 30 columns wide at x = 0, 3 ... 45.
- Per band: b = G1.5(band) - G8(band); the profile along the strip less
  its mean; strength = the standard deviation of the profile kept to
  periods of 12 to 28 px over b's standard deviation. Each strip keeps its
  best band.
- `ecard_reverse` = the mean of the two strips' strengths. `ecard_reading`
  is `bars` where the bottom strip is 0.75 or more and the left 0.72 or
  more, `dot_code` where the mean is 0.60 or less, else null
  (`marks.reading`). The web page shows the reverse holo copy for bars
  where the dataset says the standard copy prints the dot code and the
  reverse holo bars (the copy's `eReaderStrip`), and the 1st Edition copy
  where the stamp was found and the standard copy prints none.

## Index (`index/`)

| file | |
|---|---|
| `embeddings.npy` | rows x 512 float16, unit length |
| `rows.json` | each row's picture handle |
| `cards.jsonl.gz` | one line per handle: `kind` (`card`: the card's own picture; `variant`: a picture of one copy; `view`: a second view), `set`, `card`, `variant`, `name`, `set_name`, `number`, `family`, `language`, `release`, `artwork_group`, `release_date`, `rarity`, `default_finish`, `scan_prior`, `picture` |
| `dupe-groups.json` | handle to its group of near-identical pictures |
| `picker-embeddings.npy`, `picker-rows.json` | the print picker's gallery |
| `verifier-pictures.json` | print to the picture the verifier compares it with, where that isn't its own |
| `release.json` | the dataset version and models the index was built from |

A handle is the picture's id on the dataset's image mirror, or `view-<n>`
for a picture that isn't in the dataset. `release` and `artwork_group` are
opaque tokens: equal tokens mean the same set release, or the same
painting.
