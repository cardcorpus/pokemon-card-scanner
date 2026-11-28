# CardCorpus Scanner

Name Pokémon trading cards in photos. Point it at a binder page, a table
of cards or a single card, and it finds every card in the picture and tells
you which print each one is, in any of sixteen languages (English,
Japanese, Simplified and Traditional Chinese, Korean, Thai, Indonesian,
French, German, Italian, Spanish, Portuguese, Latin American Spanish,
Dutch, Russian and Polish), from 1996 to this year's
sets.

Each answer is the card's address in the [CardCorpus dataset](https://cardcorpus.com)
([github.com/cardcorpus/pokemon-card-dataset](https://github.com/cardcorpus/pokemon-card-dataset)),
an open dataset of every Pokémon card, so everything else about the card
(its set, rarity, illustrator, copies, other languages, pictures) is one
request away.

This repository is the reference implementation in Python. The weights and
the reference index are on the Hugging Face Hub at
[cardcorpus/pokemon-card-scanner](https://huggingface.co/cardcorpus/pokemon-card-scanner).

![The local web page with a binder page scanned](docs/web-page.png)

## Install

Python 3.10 or newer:

```sh
pip install "cardcorpus-scanner @ git+https://github.com/cardcorpus/pokemon-card-scanner"
```

or, from a clone, `pip install -e .`. It uses the fastest hardware it
finds: an NVIDIA GPU, then Apple silicon's GPU, then the CPU, and says
which when it starts (`scanner ready in ... on mps`). `--device cpu`,
`cuda` or `mps` picks one yourself. On the CPU it runs the models through
ONNX Runtime, which gives the same answers about twice as fast as PyTorch:
each model is converted once, checked against PyTorch, and kept in
`~/.cache/cardcorpus-scanner/onnx`. `--runtime torch` keeps PyTorch. The first run
downloads the weights and index from the Hub, about 572 MB, and
keeps them in the Hugging Face cache.

## Scan photos

```sh
cardcorpus-scan examples/vintage-english-binder.jpg
```

```text
vintage-english-binder.jpg: 12 cards
   1. Nidoqueen - Jungle 23/64  [english/jungle/23]  0.981
   2. Muk - Fossil 28/62  [english/fossil/28]  0.980
   3. Vileplume - Jungle 31/64  [english/jungle/31]  0.972
   4. Clefable - Jungle 17/64  [english/jungle/17]  0.978
   5. Victreebel - Jungle 30/64  [english/jungle/30]  0.975
   6. Magneton - Fossil 26/62  [english/fossil/26]  0.985
   7. Hypno - Fossil 23/62  [english/fossil/23]  0.965
   8. Venomoth - Jungle 29/64  [english/jungle/29]  0.985
   9. Gengar - Fossil 20/62  [english/fossil/20]  0.987
  10. Raichu - Fossil 29/62  [english/fossil/29]  0.966
  11. Wigglytuff - Jungle 32/64  [english/jungle/32]  0.968
  12. Haunter - Fossil 21/62  [english/fossil/21]  0.980
```

Each line is the scanner's first pick: name, set and number, the card's
address in the dataset and how closely the card's picture matched (cosine
similarity, 0 to 1). `--json out.json` writes everything: every card's
corners in the photo, its five best candidates in order and the evidence
behind them. A sliver of a card cut off by the photo's edge is left out,
as there's too little of it to name; `--keep-cut-off` matches it anyway.

### Languages

The index holds prints in sixteen languages. By default a card is matched
against English, Japanese and Simplified Chinese prints, as the app does.
Many cards in the other languages share a picture with one of those three
(a French Pikachu is the English one translated), so matching against every
catalog picks the right card in the wrong language more often: in our
photos, about 15 points more English and Japanese cards come back as a
translated twin. Name the catalogs you're scanning instead:

```sh
cardcorpus-scan --catalog german binder-page.jpg
cardcorpus-scan --catalog japanese --catalog korean binder-page.jpg
cardcorpus-scan --catalog all binder-page.jpg       # every catalog
```

`--catalog` takes the dataset's catalogs, one per language (`english`,
`japanese`, `zh-hans`, `korean`, `zh-hant`, `thai`, `indonesian`,
`french`, `german`, `italian`, `spanish`, `portuguese`,
`latin-american-spanish`, `dutch`, `russian`, `polish`), or `all`,
repeatable. Fewer catalogs means fewer look-alikes to choose from.

The **language check** runs on every card: a small model looks at the
card's name bar, foot and rules text, and when it is sure the card is
printed in another language than the pick's, the line says so, with that
language's print of the card:

```text
   3. Pikachu - Base Set 58/102  [english/base-set/58]  0.941
      looks German (0.97): german/base-set/58
```

The pick stays. `--switch-language` lets the check move such a card to
that print when its picture matches about as well, and
`--no-language-check` turns the check off.

Spain's and Latin America's Spanish prints carry the same words, so the
check can't tell them apart. When a scan allows no Spanish print, the check
suggests one Spanish, Spain's by default; `--default-spanish
latin-american-spanish` makes it Latin America's. Between the two, a card
moves only on the code and mark printed in its corner (`PBL ES`, `PBL LA`).

`--read-language` goes further for cards whose prints in two languages
match the photo all but equally (an English card and its Thai or
Indonesian twin, a Simplified and a Traditional Chinese print): it reads
the card's language with the language model, then the set code and
language mark printed in its corner (`MEW EN`, `M6a F`), then, if still
unsure, OCR, and puts the print in that language first. The OCR step needs
`pip install "cardcorpus-scanner[ocr]"` (Apple's Vision on a Mac, PP-OCRv5
through RapidOCR elsewhere). `--no-language-model` loads neither model.

Boxes the detector is unsure of are named only when their picture plainly
matches, and each pick is checked against its own picture before it is
given ([SPEC.md](SPEC.md)).

## The web page

```sh
cardcorpus-scan serve
```

opens a page at http://127.0.0.1:8000 where you can drop, paste or choose a
photo, or take one with your webcam, and choose a language or keep "Any
language". The first picks show as soon as the
cards are fingerprinted, and are replaced by the scanner's answer once each
card has been double-checked. Each card found is outlined on the
photo and shown beside it with what the dataset says about it: set,
number, rarity, illustrator, its copies (1st Edition, reverse holo and so
on) and its description. A card the language check flags says which
language it looks like, with a button to show that language's print.

The page and the scanner run on your computer and your photos stay there.
The page reads card details and pictures from cardcorpus.com, and the
print verifier (below) fetches the reference pictures it compares with
from the dataset's image mirror. It listens on 127.0.0.1 only, unless you
pass `--host`.

## From Python

```python
from cardcorpus_scanner import Scanner

scanner = Scanner.load()                      # or Scanner.load("path/to/weights")
result = scanner.scan("binder-page.jpg")      # a path or a PIL image
for card in result["cards"]:
    best = card["candidates"][0]
    print(best["name"], best["set_name"], best["number"], best["print"], best["sim"])
```

A candidate's `print` is `"<set>/<card>"`: `english/base-set/4` is
card `4` in https://pokemon.cardcorpus.com/v1/sets/english/base-set.json.
Where the scanner's picture shows one particular copy of the card, the
candidate also has its `variant` (an id in that card's `variants`).

`scanner.scan` returns:

| field | |
|---|---|
| `width`, `height` | the size the scanner worked at: the photo after its EXIF orientation, shrunk to 3000 px on its longer side when larger |
| `original` | the photo's own size, when it was shrunk |
| `cards[].box` | the card's place among the detector's boxes (a crop saved with `crops_dir` is `<photo>-<box>.jpg`) |
| `cards[].quad` | the card's four corners in those pixels, the card's top left first |
| `cards[].conf` | the detector's confidence |
| `cards[].candidates` | the five best candidates, best first: `print`, `set`, `card`, `variant`, `name`, `set_name`, `number`, `language`, `sim`, `picture` |
| `cards[].number_read` | what the number reader read off the card: `text`, `n`, `d` (as in 4/102), `code` (a letter code such as SV049), `votes` |
| `cards[].pick`, `verify`, `numfit` | the print picker's, print verifier's and constrained number read's scores per print |
| `cards[].edition_stamp` | how much the card's 1st Edition stamp place looks like the stamp (0.60 or more: a stamp; Wizards of the Coast's English sets) |
| `cards[].ecard_reverse`, `ecard_reading` | how much the e-Reader strips look like a reverse holo's foil bars (Expedition, Aquapolis, Skyridge), and what that reads as: `bars`, `dot_code` or null |
| `cards[].back_sim` | similarity to a card back (0.60 or more: probably a back) |
| `cards[].orient_p` | how sure the orientation head is that the card was read the right way up |
| `cards[].alignment` | whether the card lines up with its pick's own picture: `aligned`, `pass`, `inliers`, `scan_features`, `box_error` |
| `cards[].likely_wrong` | true when the card had detail enough and still didn't line up with its pick's picture |
| `cards[].realigned`, `yielded` | true when that check changed the pick (SPEC.md, stage 7) |
| `cards[].quality` | how sure the pick is: `sure`, `printing_unclear` (only other prints of the same card are close), `unsure`, `likely_wrong` or `back`; `decided` is true when an unclear pick still leads clearly enough to take |
| `cards[].ruled_out` | close candidates of another name that plainly don't line up with the card |
| `cards[].detail` | true for a card small in the photo, read from the photo at its own size |
| `cards[].language_guess` | the language model's best language for the card over the twelve it knows (every language but Dutch, Russian and Polish, with Spain's and Latin America's Spanish as one): `language`, `confidence` |
| `cards[].language_check` | when the model is sure the card is in another language than the pick's: `suggested`, `confidence`, and that language's print of the card (`print`, `name`, `set_name`, `number`, `picture`, `sim`, `linked`: true when the dataset links it to the pick), `switched` (and `original`) when it moved the pick |
| `cards[].language`, `language_switched` | what the language reader read (`language`, `why`, and `corner`, the corner reader's read) and whether it put another language's print first; only with `rules={"reads_language": True}` |
| `cut_off` | cards cut off by the photo's edge, left out unmatched: their `quad` and `conf` |
| `left_out` | boxes the scanner judged not worth naming: `quad`, `conf`, `reason` (`junk` or `unsure`), and `top`, the best match it had |

`scan(..., evidence=True)` adds each candidate's evidence terms;
`scan(..., fingerprints=True)` each card's `fingerprint` (a torch tensor),
for a search of your own with `scanner.rank()`.
`scan(..., on_first=f)` calls `f` with the first picks (each card's
candidates by picture alone) before the slower stages run.
`scan(..., on_progress=f)` calls `f` as the scan goes: with the boxes once
the cards are found, with each card's first pick as it is fingerprinted,
and with each card as it is double-checked, so a page can show every card
the moment it is known.
`scan(..., rules={"cut_off": False})` changes one of the rules in
`pipeline.json`'s `rules` for that scan: `reads_language`,
`language_check`, `language_check_switch` and `language_check_confidence`
are the language settings above.
`scan(..., languages=["ja", "en"])` and `scan(..., catalogs=["japanese"])`
keep to prints in those languages or catalogs (with neither, English,
Japanese and Simplified Chinese; `catalogs="all"` for every one); `scanner.languages` and
`scanner.catalogs` list those the index holds.
`Scanner.load(..., language_model=False, corner_reader=False)` loads
without the language model or the corner reader.

## How it works

Each card goes through nine steps, all on your machine. [SPEC.md](SPEC.md)
gives every number.

1. **Detect.** QuadDet, a small corner detector (1.4M parameters), finds
   each card's four corners.
2. **Orient and fingerprint.** Each card is straightened in its four
   possible turns and fingerprinted by the matcher (a fine-tuned DINOv3
   ViT-S/16); an orientation head says which turn is upright.
3. **Retrieve.** The fingerprint's nearest pictures in the reference
   index: 256,216 pictures of 214,178 cards.
4. **Straighten** the card again at full resolution.
5. **Read the number.** A small CRNN reads the collector number from the
   bottom of the card.
6. **Pick the print.** Reprints share a painting, and the matcher sees
   paintings. The print picker (a fine-tuned PE-Core S/16) looks at the
   whole card and scores the prints of one painting.
7. **Fuse and verify.** Candidates are re-ordered by picture similarity
   plus weighted evidence (the number read, the other cards on the page,
   the picker), and the print verifier lines the card up with each
   sibling print's reference picture and checks the places where those
   pictures differ: the set symbol, the number, the copyright line.
8. **1st Edition stamp** score.
9. **e-Card reverse holo** score.

Then the **language check**: the language model (a fine-tuned LCNet-050)
says which language the card is printed in, and a card it is sure is in
another language than its pick's is flagged. With `--read-language`, ties
between one card's prints in two languages are read: the language model,
then the corner reader (the set code and language mark), then OCR.

## Accuracy

On photos the models never saw in training (one card counted right when
the first pick is the right print):

| photos | cards | right (default) | right (`--catalog all`) |
|---|---|---|---|
| binder pages of vintage cards, at the camera's full resolution | 641 | 629 (98.1%) | 618 (96.4%) |
| the same photos shrunk to 3000 px | 641 | 630 (98.3%) | 618 (96.4%) |
| a collection's binder pages | 1,263 | 1,234 (97.7%) | 1,179 (93.3%) |
| cards photographed with a phone | 139 | 122 (87.8%) | 101 (72.7%) |
| varied real-world photos, many small, blurred or at an angle | 672 | 585 (87.1%) | 431 (64.1%) |

Measured on 26 November 2026 with these models and the index of that day.
The default keeps to English, Japanese and Simplified Chinese prints;
`--catalog all` matches against all sixteen languages, and nearly all of
the cards it loses come back as the same card printed in another language
(an English card as its Italian or German print, a Japanese one as its
Korean print). Cards cut off by the photo's edge are left out, as the
scanner does by default.

A binder page of 9 to 12 cards takes about 3.6 s on Apple silicon's GPU
(`--device mps`, an Apple M6) and about 12.5 s on its CPU.

It knows the cards in the index it was built with (the dataset as of
27 November 2026). A card released after that isn't in it, and the scanner
names the closest card it knows. It reads which print a card is, not its
condition, and it wasn't tested on cards in graded slabs.

## Examples

[examples/](examples/) has photos of binder pages with what each card
really is, checked by eye, and what the scanner says about it with these
weights and the default languages. `pytest` checks the scanner still says
it.

## Licence

The code is under the Apache License 2.0 ([LICENSE](LICENSE)). The weights
and index on the Hub carry their own licences, listed in the model card:
the matcher is a fine-tune of DINOv3 and under the DINOv3 License; the rest
are Apache 2.0. Built with DINOv3.

Pokémon and Pokémon character names are trademarks of Nintendo, Creatures
and GAME FREAK. Card pictures and text © The Pokémon Company, Nintendo,
Creatures, GAME FREAK and, for older cards, Wizards of the Coast. This
project is not produced, endorsed, sponsored or supported by, or affiliated
with, any of them.
