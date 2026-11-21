"""cardcorpus-scan: name the cards in photos, or serve the local web page.

    cardcorpus-scan photo.jpg [photo.jpg ...]       a line per card
    cardcorpus-scan --json out.json photo.jpg ...   everything, as JSON
    cardcorpus-scan serve [--port 8000]             the web page on localhost

--weights takes a local copy of the weights; without it they come from the
Hugging Face Hub (downloaded once, then cached).
"""
import argparse
import json
import sys
import time
from pathlib import Path


def _scanner(args):
    from .scanner import Scanner, pick_device

    t = time.time()
    scanner = Scanner.load(args.weights, device=pick_device(args.device), verifier=not args.no_verifier,
                           picture_cache=args.picture_cache, language_model=not args.no_language_model,
                           corner_reader=not args.no_language_model)
    print(f"scanner ready in {time.time() - t:.1f} s on {scanner.device} "
          f"({scanner.index.release.get('cards', '?')} cards in the index)", file=sys.stderr)
    return scanner


def _common(ap):
    ap.add_argument("--weights", help="folder with pipeline.json, models/ and index/ (default: the Hub's copy)")
    ap.add_argument("--device", default="auto",
                    help="cpu, cuda or mps; auto (the default) picks an NVIDIA GPU, then Apple silicon's GPU, then the CPU")
    ap.add_argument("--no-verifier", action="store_true",
                    help="skip the print verifier (it fetches reference pictures from the image mirror)")
    ap.add_argument("--picture-cache", default=str(Path.home() / ".cache" / "cardcorpus-scanner" / "pictures"),
                    help="where reference pictures are kept once fetched")
    ap.add_argument("--no-language-model", action="store_true",
                    help="don't load the language model or the corner reader: no language check, and "
                    "--read-language goes straight to OCR")


def _rules(args):
    """The rules the flags change (pipeline.json's `rules`), or None."""
    rules = {}
    if args.keep_cut_off:
        rules["cut_off"] = False
    if args.read_language:
        rules["reads_language"] = True
    if args.no_language_check:
        rules["language_check"] = False
    if args.switch_language:
        rules["language_check_switch"] = True
    if getattr(args, "default_spanish", None):
        rules["default_spanish"] = args.default_spanish
    return rules or None


def _check_choices(ap, given, have, flag):
    """Refuse a --catalog the index doesn't hold, naming those it does."""
    unknown = [g for g in given or [] if g not in have]
    if unknown:
        ap.error(f"{flag} {', '.join(unknown)}: this index has {', '.join(have)}")


def scan_main(argv):
    ap = argparse.ArgumentParser(prog="cardcorpus-scan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    _common(ap)
    ap.add_argument("--json", help="write every result here")
    ap.add_argument("--catalog", action="append", metavar="NAME",
                    help="only cards of this catalog, as the dataset names it: english, japanese, zh-hans, korean, "
                    "zh-hant, thai, indonesian, french, german, italian, spanish, portuguese, latin-american-spanish, dutch, "
                    "russian, polish, or all (repeatable; "
                    "default: english, japanese and zh-hans, as the app does; the language check still flags a "
                    "card from another catalog)")
    ap.add_argument("--keep-cut-off", action="store_true",
                    help="also match cards cut off by the photo's edge (left out by default)")
    ap.add_argument("--read-language", action="store_true",
                    help="when the pick has a print of the same card in another language as close by picture, "
                    "read the card's language (language model, corner, then OCR) and put that print first")
    ap.add_argument("--no-language-check", action="store_true",
                    help="skip the language check: by default each card whose text the language model is sure "
                    "is in another language than its pick's is flagged (the pick stays)")
    ap.add_argument("--switch-language", action="store_true",
                    help="let the language check move a flagged card to that language's print when it matches "
                    "about as well")
    ap.add_argument("--default-spanish", choices=["spanish", "latin-american-spanish"],
                    help="the Spanish a language check suggests for a card in Spanish when the scan allows no Spanish "
                    "print (default: spanish). Spain's and Latin America's print the same text, so neither is ever "
                    "suggested over the other, and the reader moves a card between them only on the corner's code")
    ap.add_argument("photos", nargs="+")
    args = ap.parse_args(argv)
    scanner = _scanner(args)
    _check_choices(ap, args.catalog, scanner.catalogs + ["all"], "--catalog")
    from .language import LANGUAGE_NAMES

    rules = _rules(args)
    results = []
    for photo in args.photos:
        t = time.time()
        result = scanner.scan(photo, catalogs=args.catalog, rules=rules)
        result["file"] = Path(photo).name
        results.append(result)
        print(f"{Path(photo).name}: {len(result['cards'])} cards, {time.time() - t:.1f} s")
        for i, card in enumerate(result["cards"], 1):
            best = card["candidates"][0] if card["candidates"] else None
            if best is None:
                print(f"  {i:2d}. no match")
                continue
            number = f" {best['number']}" if best.get("number") else ""
            print(f"  {i:2d}. {best['name']} - {best['set_name']}{number}  [{best['print']}]  {best['sim']:.3f}")
            check = card.get("language_check")
            if check and not check.get("switched"):
                name = LANGUAGE_NAMES.get(check["suggested"], check["suggested"])
                where = f": {check['print']}" if check.get("linked") and check.get("print") else ""
                print(f"      looks {name} ({check['confidence']:.2f}){where}")
            elif check:
                print(f"      moved from {check.get('original')}: the card looks "
                      f"{LANGUAGE_NAMES.get(check['suggested'], check['suggested'])}")
        if result["cut_off"]:
            n = len(result["cut_off"])
            print(f"  {n} {'card' if n == 1 else 'cards'} at the photo's edge left out")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, ensure_ascii=False))


def serve_main(argv):
    ap = argparse.ArgumentParser(prog="cardcorpus-scan serve", description="The scanner's web page, on this machine only.")
    _common(ap)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 keeps it to this machine")
    args = ap.parse_args(argv)
    from .web import serve

    serve(_scanner(args), args.host, args.port)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["serve"]:
        return serve_main(argv[1:])
    return scan_main(argv)


if __name__ == "__main__":
    main()
