"""Name Pokémon trading cards in photos.

    from cardcorpus_scanner import Scanner
    scanner = Scanner.load()                 # the weights from the Hugging Face Hub
    result = scanner.scan("binder-page.jpg")
    for card in result["cards"]:
        best = card["candidates"][0]
        print(best["name"], best["set_name"], best["number"], best["print"])

Each candidate's `print` ("<set>/<card>") is the card's address in the
CardCorpus dataset: https://pokemon.cardcorpus.com/v1/sets/<set>.json, card <card>.
"""
from .scanner import Scanner, pick_device

__version__ = "1.2.1"
__all__ = ["Scanner", "pick_device", "__version__"]
