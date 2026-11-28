# Changelog

## 1.2.1 (27 November 2026)

**Index.** Rebuilt from the dataset of 27 November 2026: 256,216
pictures of 214,178 cards. Its card ids follow the dataset's: the
Japanese and Traditional Chinese Black Bolt and White Flare sets drop
`japanese-` from their ids, and Korean card numbers keep their leading
zeros (`korean/lost-abyss/038`).

## 1.2.0 (26 November 2026)

**Index.** Rebuilt from the dataset of 26 November 2026: 256,216
pictures of 214,180 cards, among them its new cards and pictures.

**Upside-down cards at the photo's edge.** A whole card touching the
photo's edge can now be read turned 180°, as one inside the photo always
could; only a card cut off by the edge keeps the turn it was found at.

**Examples.** Each example's JSON labels every card with the print it
really is, and the examples README says where the default scan picks
another (two Traditional Chinese cards in `chinese-binder.jpg` scan as
Simplified unless Traditional Chinese is searched; the language check
flags one of them).

**Faster on a CPU.** On the CPU the scanner now runs its models through
ONNX Runtime instead of PyTorch: the same answers, about twice as fast (a
binder page of nine cards in about 11 s instead of 18 s on four server
cores). Each model is converted the first time it is needed, checked
against PyTorch, and cached in `~/.cache/cardcorpus-scanner/onnx`; one
that doesn't match stays on PyTorch. `--runtime torch` (`Scanner(...,
runtime="torch")`) keeps PyTorch; GPUs (`cuda`, `mps`) are unchanged. The
package now needs `onnx` for the conversion.

## 1.1.0 (21 November 2026)

**Sixteen languages.** The index now holds 256,022 pictures of
214,029 cards in English, Japanese, Simplified and Traditional
Chinese, Korean, Thai, Indonesian, French, German, Italian, Spanish,
Portuguese, Latin American Spanish, Dutch, Russian and Polish (1.0.0:
English, Japanese and Simplified Chinese). The language model knows twelve
of them: it doesn't flag a card as Dutch, Russian or Polish yet. Prints are
addressed by the dataset's new catalog ids: `english/base-set/4`, where 1.0.0
said `international/base-set/4`.

**Spanish twins.** Spain's and Latin America's Spanish prints share their
text, so only the corner's printed code (`PBL` and `PBL LA`) tells them
apart. A new setting, `default_spanish` (`spanish`, or
`latin-american-spanish`; `--default-spanish`), is the one Spanish the
language check suggests, and only when a scan allows no Spanish print; the
other is never suggested. The reader moves a card between the two only on
the corner reader's exact read, and `spanish_corner` (on) reads the corner
on every card whose pick has a twin in the other Spanish.

**Language check.** A new model, the language model (a fine-tuned
LCNet-050, `models/language-classifier.onnx`), looks at every card's name
bar, foot and rules text. Each card gets `language_guess`, and when the
model is sure the card is in another language than its pick's, a
`language_check` with that language's print of the card. The pick stays
unless you ask (`--switch-language`, `rules={"language_check_switch": True}`).
`--no-language-check` turns it off.

**Language reader** (`--read-language`, off by default). When the pick
has a print of the same card in another language that matches the photo
all but as well, the card's language is read in three steps, each only
when the one before is unsure: the language model, then the corner reader
(a new model, `models/corner-reader.safetensors`, that reads the set code
and language mark printed at the card's foot, such as `MEW EN` or
`M6a F`, against each set's printed code), then OCR (Apple's Vision on a
Mac, PP-OCRv5 elsewhere; `pip install "cardcorpus-scanner[ocr]"`).

**Choosing catalogs.** A scan still keeps to English, Japanese and
Simplified Chinese unless told otherwise, as the app does (1.0.0's index
held only those three):
matching against every catalog picks more cards in the wrong language,
since many share a picture with their translations. A card in another
language is flagged by the language check with its own print.
`--catalog german` (or `scan(catalogs=[...])`) keeps to the catalogs you
name, any catalog in the index, and `--catalog all` matches against every
one; `scanner.catalogs` lists them (`scan(languages=[...])` and
`scanner.languages` do the same by printed language). The web page offers
a chip per catalog in the index and shows the language check on each card
it flags.

**Also:**

- `onnxruntime` is a new dependency (the language model).
- `Scanner.load(..., language_model=False, corner_reader=False)` and
  `--no-language-model` load without the new models.
- The index can come in parts (`index/release.json`'s `parts`), and
  `Scanner.load(..., catalogs=[...])` loads only the parts holding those
  catalogs. A card row's catalog is `catalog` (1.0.0's `family` is still
  read).

The detector, matcher, orientation head, number reader and print picker
are the 1.0.0 models.
