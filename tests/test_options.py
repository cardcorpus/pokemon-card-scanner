"""The scan's language options: the CLI's flags, and the language check's bar (SPEC.md, "Language check")."""
import argparse

import pytest


def flags(*argv):
    from cardcorpus_scanner import cli
    ap = argparse.ArgumentParser()
    cli._common(ap)
    ap.add_argument("--keep-cut-off", action="store_true")
    ap.add_argument("--read-language", action="store_true")
    ap.add_argument("--no-language-check", action="store_true")
    ap.add_argument("--switch-language", action="store_true")
    return cli._rules(ap.parse_args(list(argv)))


def test_no_flags_keep_pipeline_json_rules():
    assert flags() is None


def test_each_flag_sets_its_rule():
    assert flags("--keep-cut-off", "--read-language", "--no-language-check", "--switch-language") == {
        "cut_off": False, "reads_language": True, "language_check": False, "language_check_switch": True}


def test_a_catalog_the_index_lacks_is_refused_with_those_it_has():
    from cardcorpus_scanner import cli
    ap = argparse.ArgumentParser()
    cli._check_choices(ap, ["english", "japanese"], ["english", "japanese", "zh-hans"], "--catalog")
    with pytest.raises(SystemExit):
        cli._check_choices(ap, ["dutch"], ["english", "japanese"], "--catalog")


def test_languages_come_in_the_catalogs_order():
    np = pytest.importorskip("numpy")
    from cardcorpus_scanner.scanner import Scanner
    s = Scanner.__new__(Scanner)
    s.row_language = np.array(["fr", "", "ja", "en", "zh-Hans", "en", "ko"])
    assert s.languages == ["en", "ja", "zh-Hans", "ko", "fr"]
    s.row_catalog = np.array(["french", "english", "japanese", "english", "zh-hans", "english", "korean"])
    assert s.catalogs == ["english", "japanese", "zh-hans", "korean", "french"]
    assert s.catalog_languages["french"] == "fr"


def test_a_scan_keeps_to_english_japanese_and_simplified_chinese_unless_told():
    np = pytest.importorskip("numpy")
    from cardcorpus_scanner.scanner import Scanner
    s = Scanner.__new__(Scanner)
    s.row_language = np.array(["en", "ja", "zh-Hans", "fr", "ko"])
    s.row_catalog = np.array(["english", "japanese", "zh-hans", "french", "korean"])
    assert s.scan_catalogs(None) == ["english", "japanese", "zh-hans"]
    assert s.scan_catalogs(["french"]) == ["french"]
    assert s.scan_catalogs("all") is None and s.scan_catalogs(["all"]) is None
    assert s.scan_catalogs(None, languages=["fr"]) is None  # languages narrow the scan themselves
    t = Scanner.__new__(Scanner)
    t.row_language, t.row_catalog = np.array(["fr"]), np.array(["french"])
    assert t.scan_catalogs(None) is None  # an index with none of the three: every catalog
    old = Scanner.__new__(Scanner)  # a 1.0.0 index, whose English catalog is "international"
    old.row_language = np.array(["en", "ja", "zh-Hans"])
    old.row_catalog = np.array(["international", "japanese", "zh-hans"])
    assert old.scan_catalogs(None) == ["japanese", "zh-hans", "international"]


def checker(pick_sim, other_sim):
    """A scanner whose index holds one German print of the pick's card, `other_sim` from the photo."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from cardcorpus_scanner.scanner import Scanner

    s = Scanner.__new__(Scanner)
    s.torch = torch
    s.cfg = {"retrieval": {"candidates": 10}}
    s.row_language = np.array(["en", "de"])
    s.matrix = torch.tensor([[pick_sim], [other_sim]])
    s.language_classifier = type("C", (), {"layout": {"check": 0.94}})()
    german = {"print": "german/x/1", "language": "de", "same_card": "c1", "name": "Glurak", "set_name": "Basis",
              "number": "4/102", "picture": "https://img.example/1.avif"}
    s.dedupe = lambda values, idxs, limit: [{**german, "sim": v} for v, i in zip(values, idxs) if i == 1]
    pick = {"print": "english/x/1", "language": "en", "same_card": "c1", "sim": pick_sim}
    return s, {"feat": torch.tensor([1.0])}, pick


def test_a_lowered_bar_flags_more_cards_but_switches_none_the_models_bar_would_not():
    s, d, pick = checker(0.90, 0.89)
    check, order = s.check_language(d, ("de", 0.90), [pick], {"language_check_confidence": 0.85,
                                                             "language_check_switch": True})
    assert check["suggested"] == "de" and not check["switched"] and order[0] is pick
    assert check["name"] == "Glurak" and check["linked"]
    check, order = s.check_language(d, ("de", 0.97), [pick], {"language_check_switch": True})
    assert check["switched"] and order[0]["print"] == "german/x/1"


def test_the_models_own_bar_holds_without_an_override():
    s, d, pick = checker(0.90, 0.89)
    check, _ = s.check_language(d, ("de", 0.90), [pick], {})
    assert check is None
