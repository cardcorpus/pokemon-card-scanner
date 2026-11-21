"""The language reader: which language a card is printed in, read off its
text, for prints of one card in two languages that the matcher can't tell
apart (a Simplified and a Traditional Chinese print, an English one and its
Thai, Indonesian or European twin). It runs only on such a tie, after
fusion, and puts the print in the language read first (SPEC.md, "Language
reader"); ScannerKit's LanguageReader is the same rule.

The rule reads lines of text with their confidence and where their top sits
on the card; any OCR engine can supply them (`read`'s `engine`). Apple's
Vision text recognition is what the app uses; `rapidocr_engine` is an open
one (PP-OCRv5 through RapidOCR, Apache-2.0).
"""
import json
import re
from pathlib import Path

# Regions as fractions of the straightened card, from its top left: x, y, width, height.
TEXT = (0.04, 0.55, 0.92, 0.31)   # attacks and abilities, a Trainer's rules
CODE = (0.0, 0.88, 0.6, 0.11)     # the foot: credit, set code and language mark, number
CARD = (0.0, 0.0, 1.0, 1.0)       # the whole card, when the rest settles nothing
LATIN = ("en", "fr", "de", "es", "it", "pt", "id")
LATIN_PASS = ["en-US", "fr-FR", "de-DE", "es-ES", "it-IT", "pt-BR"]
MINIMUM_SCRIPT = 2.0
CODE_WEIGHT = 3
_chars = json.loads((Path(__file__).parent / "script-characters.json").read_text())
SIMPLIFIED_ONLY, TRADITIONAL_ONLY = set(_chars["simplified"]), set(_chars["traditional"])
MARK = re.compile(r"\b[A-Z0-9]{2,5}\s(EN|FR|DE|ES|IT|PT)\b")
SIMPLIFIED_CODE = re.compile(r"\b[A-Za-z0-9]{1,6}\s?C\b")
TRADITIONAL_CODE = re.compile(r"\b[A-Za-z0-9.\-]{1,6}\s?F\b")
LONG_WORDS_NEEDED = 2   # function words of two letters or more the winning language needs
FUNCTION_WORDS = {
    "en": {"the", "your", "this", "that", "of", "to", "and", "is", "if", "from", "all", "opponent's", "pokémon", "damage", "each"},
    "fr": {"le", "la", "les", "de", "des", "du", "et", "votre", "vos", "ce", "cette", "un", "une", "est", "à", "dégâts", "adverse", "chaque"},
    "de": {"der", "die", "das", "und", "dein", "deine", "deinen", "ein", "eine", "ist", "von", "zu", "mit", "schadenspunkte", "gegnerischen", "jedes"},
    "es": {"el", "la", "los", "las", "de", "del", "y", "tu", "tus", "este", "esta", "un", "una", "es", "daño", "rival", "cada"},
    "it": {"il", "la", "lo", "gli", "le", "di", "del", "e", "tuo", "tua", "questo", "questa", "un", "una", "è", "danni", "avversario", "ogni"},
    "pt": {"o", "a", "os", "as", "de", "do", "da", "e", "seu", "sua", "este", "esta", "um", "uma", "é", "dano", "oponente", "cada"},
    "id": {"dan", "yang", "dari", "ini", "itu", "ke", "di", "kamu", "lawan", "setiap", "kerusakan"},
}


def passes(languages):
    """The OCR passes worth running to tell `languages` apart: one per script, Latin languages sharing one."""
    out = []
    if "ko" in languages: out.append(("ko-KR", ["ko-KR"]))
    if "ja" in languages: out.append(("ja-JP", ["ja-JP"]))
    if languages & {"zh-Hans", "zh-Hant"}: out.append(("zh-Hans", ["zh-Hans"]))
    if "th" in languages: out.append(("th-TH", ["th-TH"]))
    if languages & set(LATIN): out.append(("latin", LATIN_PASS))
    return out


def _weight(lines, test):
    return sum(line["c"] * sum(1 for ch in line["t"] if test(ord(ch))) for line in lines)


def _hangul(o): return 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F
def _kana(o): return 0x3040 <= o <= 0x30FF
def _han(o): return 0x4E00 <= o <= 0x9FFF
def _thai(o): return 0x0E00 <= o <= 0x0E7F
def _latin(o): return 0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A or 0xC0 <= o <= 0x24F


def script(region):
    """The script a region's passes read most of ("ko", "ja", "zh", "th"), or None with too little."""
    every = [line for lines in region.values() for line in lines]
    weights = [("ko", _weight(region.get("ko-KR", []), _hangul)), ("ja", 2 * _weight(region.get("ja-JP", []), _kana)),
               ("zh", _weight(region.get("zh-Hans", []), _han)),
               # Thai over what the Latin pass read: a Thai pass on an English card invents Thai letters, fewer
               # than the English it holds
               ("th", _weight(every, _thai) - _weight(region.get("latin", []), _latin))]
    best = weights[0]
    for w in weights[1:]:
        if w[1] > best[1]: best = w
    return best[0] if best[1] >= MINIMUM_SCRIPT else None


def script_characters(lines):
    read = {ch for line in lines for ch in line["t"]}
    return len(read & SIMPLIFIED_ONLY), len(read & TRADITIONAL_ONLY)


def decide(reading, languages):
    """(language or None, why) from a reading {"text": {pass: lines}, "code": lines, "name": {pass: lines}};
    a line is {"t": text, "c": confidence, "top": its top as a fraction of the card's height}."""
    def answer(language, why):
        return (language, why) if language in languages else (None, why)
    foot = " ".join(line["t"] for line in reading.get("code", []) if line["top"] > 0.84)
    text, name = reading.get("text", {}), reading.get("name", {})
    family = script(text) or script(name)
    if family in ("ko", "ja", "th"):
        return answer(family, "script")
    if family == "zh":
        simplified, traditional = script_characters(text.get("zh-Hans", []))
        ns, nt = script_characters(name.get("zh-Hans", []))
        simplified, traditional, why = simplified + ns, traditional + nt, "characters"
        if TRADITIONAL_CODE.search(foot): traditional += CODE_WEIGHT; why = "code"
        if SIMPLIFIED_CODE.search(foot): simplified += CODE_WEIGHT; why = "code"
        if simplified > traditional: return answer("zh-Hans", why)
        if traditional > simplified: return answer("zh-Hant", why)
        return None, "characters tie"
    mark = MARK.search(foot)
    if mark: return answer(mark.group(1).lower(), "mark")
    words = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", " ".join(l["t"] for l in text.get("latin", []) + name.get("latin", [])).lower())
    counts = sorted(((sum(w in fw for w in words), lang) for lang, fw in FUNCTION_WORDS.items() if lang in languages),
                    key=lambda x: (-x[0], x[1]))
    if not counts or counts[0][0] < 2: return None, "no latin evidence"
    # one-letter words alone prove nothing: on foil, misread specks come out as "a", "e" and "o"
    if sum(len(w) >= 2 and w in FUNCTION_WORDS[counts[0][1]] for w in words) < LONG_WORDS_NEEDED: return None, "no latin evidence"
    second = counts[1][0] if len(counts) > 1 else 0
    return (answer(counts[0][1], "words") if counts[0][0] >= 2 * second else (None, "words tie"))


def _region(crop, box):
    w, h = crop.size; x, y, bw, bh = box
    return crop.crop((round(x * w), round(y * h), min(w, round((x + bw) * w)), min(h, round((y + bh) * h))))


def read(crop, languages, engine):
    """The language of `crop` (a PIL image of the straightened card) among `languages`: the text box with one
    pass per script in play, the foot when Chinese or a Latin language is in play, the whole card only when
    those settle nothing. `engine(image, ocr_languages)` returns [(text, confidence, top as a fraction of image)]."""
    languages = set(languages)
    def lines(box, langs):
        return [{"t": t, "c": c, "top": box[1] + top * box[3]} for t, c, top in engine(_region(crop, box), langs)]
    todo = passes(languages)
    reading = {"text": {key: lines(TEXT, langs) for key, langs in todo}, "code": [], "name": {}}
    if languages & {"zh-Hans", "zh-Hant"} or languages & set(LATIN):
        reading["code"] = lines(CODE, LATIN_PASS)
    language, why = decide(reading, languages)
    if language is None:
        reading["name"] = {key: lines(CARD, langs) for key, langs in todo}
        language, why = decide(reading, languages)
    return language, why


def vision_engine(image, langs):
    """Apple's Vision text recognition (macOS, pyobjc-framework-Vision), as the app runs it."""
    import io
    import Quartz, Vision  # noqa: E401
    from Foundation import NSData
    buf = io.BytesIO(); image.convert("RGB").save(buf, "PNG")
    data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)
    request.setRecognitionLanguages_(langs)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    handler.performRequests_error_([request], None)
    out = []
    for o in request.results() or []:
        c = o.topCandidates_(1)
        if c: out.append((str(c[0].string()), float(c[0].confidence()), 1 - (o.boundingBox().origin.y + o.boundingBox().size.height)))
    return out


_rapid = {}
def rapidocr_engine(image, langs):
    """PP-OCRv5 mobile through RapidOCR (pip install rapidocr onnxruntime; Apache-2.0): its multi-script
    recognizer (Simplified, Traditional, Japanese, English) for the Chinese and Japanese passes, and its
    Korean, Thai and Latin recognizers for theirs."""
    import numpy as np
    from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR
    key = {"ko-KR": "KOREAN", "th-TH": "TH"}.get(langs[0], "LATIN" if langs == LATIN_PASS else "CH")
    if key not in _rapid:
        _rapid[key] = RapidOCR(params={
            "Global.log_level": "warning", "Det.ocr_version": OCRVersion.PPOCRV5, "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Det.lang_type": LangDet.CH, "Det.model_type": ModelType.MOBILE, "Rec.model_type": ModelType.MOBILE,
            "Rec.lang_type": getattr(LangRec, key)})
    arr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    r = _rapid[key](arr)
    if r.boxes is None: return []
    h = arr.shape[0]
    return [(t, float(s), float(np.min(np.array(b)[:, 1])) / h) for b, t, s in zip(r.boxes, r.txts, r.scores)]


# Every language the dataset prints cards in, in its catalogs' order (pokemon.cardcorpus.com/v1/index.json), with
# its name. The index holds some of them (Scanner.languages); the language model tells the first twelve apart.
LANGUAGE_NAMES = {"en": "English", "ja": "Japanese", "zh-Hans": "Simplified Chinese", "ko": "Korean",
                  "zh-Hant": "Traditional Chinese", "th": "Thai", "id": "Indonesian", "fr": "French", "de": "German",
                  "it": "Italian", "es": "Spanish", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "pl": "Polish",
                  "es-419": "Latin American Spanish"}


# The language reader without OCR: a small image classifier (LCNet-050, 0.6 M parameters) over the card's name bar,
# foot and rules text, the same model ScannerKit runs as Core ML (LanguageClassifier.swift).
CLASSIFIER_LANGUAGES = ["en", "fr", "de", "es", "it", "pt", "ja", "ko", "zh-Hans", "zh-Hant", "th", "id"]
CLASSIFIER_WIDTH = 448
CHECK_SWITCH_MARGIN = 0.03   # how far under the pick's sim another language's print may be and still replace it
_NAME, _FOOT, _RULES = ((0.0, 0.02, 0.75, 0.11), 80), ((0.0, 0.88, 0.75, 0.99), 80), ((0.04, 0.58, 0.96, 0.84), 176)
# by the model's input height: the regions (x0, y0, x1, y1 as fractions of the card, then height; stacked in order), the
# least share among the languages in play for an answer, the least confidence over all languages for a language check
CLASSIFIER_LAYOUTS = {
    160: {"parts": (_NAME, _FOOT), "share": 0.8, "check": 0.95},
    336: {"parts": (_NAME, _FOOT, _RULES), "share": 0.95, "check": 0.94},
}
_MEAN, _STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def guess_from(logits):
    import numpy as np
    logits = np.asarray(logits, np.float64)
    best = int(np.argmax(logits))
    return CLASSIFIER_LANGUAGES[best], float(1 / np.exp(logits - logits[best]).sum())


def language_check(guess, pick_language, least):
    """{"suggested", "confidence"} when `guess` is sure of a language other than the pick's, else None. The pick stays."""
    found, confidence = guess
    if pick_language is None or found == pick_language or confidence < least:
        return None
    return {"suggested": found, "confidence": confidence}


def classifier_class(lang, languages):
    """The language among `languages` a model class answers for. The model has no Latin American Spanish: its
    Spanish class answers for Latin American Spanish when that is the only Spanish in play, and for Spain's when both
    are (the corner's code alone moves a pick between the two; SPEC.md, "Spanish twins")."""
    if lang == "es" and "es" not in languages and "es-419" in languages:
        return "es-419"
    return lang if lang in languages else None


def read_from(logits, languages, share):
    """(language or None, why): the language among `languages` holding `share` of the model's probability over
    them (classifier_class)."""
    import numpy as np
    logits = np.asarray(logits, np.float64)
    p = {}
    for i, lang in enumerate(CLASSIFIER_LANGUAGES):
        named = classifier_class(lang, languages)
        if named is not None:
            p[named] = float(np.exp(logits[i] - logits.max()))
    if not p:
        return None, "classifier: none in play"
    best = max(p, key=p.get)
    held = p[best] / sum(p.values())
    return (best if held >= share else None), f"classifier {held:.2f}"


class Classifier:
    """`path` an ONNX file (onnxruntime). `read(crop, languages)` -> (language or None, why)."""

    def __init__(self, path):
        import onnxruntime
        self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.layout = CLASSIFIER_LAYOUTS.get(self.session.get_inputs()[0].shape[2], CLASSIFIER_LAYOUTS[160])

    def input(self, crop):
        import numpy as np
        from PIL import Image
        w, h = crop.size
        parts = []
        for (x0, y0, x1, y1), height in self.layout["parts"]:
            part = crop.convert("RGB").crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
            parts.append(np.asarray(part.resize((CLASSIFIER_WIDTH, height), Image.BICUBIC), np.float32) / 255)
        x = (np.concatenate(parts, 0) - np.array(_MEAN, np.float32)) / np.array(_STD, np.float32)
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def logits(self, crop):
        import numpy as np
        return self.session.run(None, {self.session.get_inputs()[0].name: self.input(crop)})[0][0].astype(np.float64)

    def guess(self, crop):
        """The likeliest of all the model's languages and its share of the probability over all of them."""
        return guess_from(self.logits(crop))

    def read(self, crop, languages):
        return read_from(self.logits(crop), languages, self.layout["share"])
