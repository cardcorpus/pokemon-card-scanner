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
                           picture_cache=args.picture_cache)
    print(f"scanner ready in {time.time() - t:.1f} s ({scanner.index.release.get('cards', '?')} cards in the index)",
          file=sys.stderr)
    return scanner


def _common(ap):
    ap.add_argument("--weights", help="folder with pipeline.json, models/ and index/ (default: the Hub's copy)")
    ap.add_argument("--device", default="auto", help="cpu, cuda, mps or auto")
    ap.add_argument("--no-verifier", action="store_true",
                    help="skip the print verifier (it fetches reference pictures from the image mirror)")
    ap.add_argument("--picture-cache", default=str(Path.home() / ".cache" / "cardcorpus-scanner" / "pictures"),
                    help="where reference pictures are kept once fetched")


def scan_main(argv):
    ap = argparse.ArgumentParser(prog="cardcorpus-scan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    _common(ap)
    ap.add_argument("--json", help="write every result here")
    ap.add_argument("--language", action="append", choices=["international", "japanese", "zh-hans"],
                    help="only cards of this language family (repeatable)")
    ap.add_argument("--keep-cut-off", action="store_true",
                    help="also match cards cut off by the photo's edge (left out by default)")
    ap.add_argument("photos", nargs="+")
    args = ap.parse_args(argv)
    scanner = _scanner(args)
    results = []
    for photo in args.photos:
        t = time.time()
        result = scanner.scan(photo, families=args.language, rules={"cut_off": False} if args.keep_cut_off else None)
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
