"""Spanish twins (SPEC.md, "Spanish twins"): the cases ScannerKit's SpanishTwinsTests mirror one for one."""
import argparse

from cardcorpus_scanner import cli
from cardcorpus_scanner.language import CLASSIFIER_LANGUAGES, read_from
from cardcorpus_scanner.scanner import (DEFAULTS, LAYOUTS, SURE_MARGIN, corner_holds, corner_overrides, reads_corner,
                                        spanish_enabled, spanish_suggestion, spanish_switch_allowed, spanish_twins)


def test_a_scan_allows_spanish_with_either_catalog_or_language_or_every_catalog():
    assert spanish_enabled(None, None)
    assert spanish_enabled(["english", "spanish"], None)
    assert spanish_enabled(["latin-american-spanish"], None)
    assert not spanish_enabled(["english", "japanese", "zh-hans"], None)
    assert spanish_enabled(None, ["en", "es-419"])
    assert not spanish_enabled(None, ["en", "ja", "zh-Hans"])
    # languages narrow the scan, whatever the catalogs
    assert not spanish_enabled(["spanish"], ["en"])


def test_a_check_suggests_only_the_default_spanish_and_only_with_no_spanish_allowed():
    assert spanish_suggestion("es", False) == "es"
    assert spanish_suggestion("es-419", False) == "es"
    assert spanish_suggestion("es", False, "latin-american-spanish") == "es-419"
    assert spanish_suggestion("es-419", False, "latin-american-spanish") == "es-419"
    assert spanish_suggestion("es", True) is None
    assert spanish_suggestion("es-419", True, "latin-american-spanish") is None
    assert spanish_suggestion("fr", True) == "fr"
    assert spanish_suggestion("en", False, "latin-american-spanish") == "en"


def test_the_reader_moves_between_the_spanish_prints_only_on_the_corners_exact_code():
    assert spanish_switch_allowed("es", "es-419", "corner PBL LA")
    assert spanish_switch_allowed("es-419", "es", "corner MEG")
    assert not spanish_switch_allowed("es", "es-419", "corner mark PBL LA")
    assert not spanish_switch_allowed("es-419", "es", "classifier 0.99")
    assert not spanish_switch_allowed("es", "es-419", "words")
    assert spanish_switch_allowed("en", "es", "classifier 0.99")
    assert spanish_switch_allowed("es", "en", "words")
    assert spanish_switch_allowed("es", "es", "classifier 0.99")


def test_a_spanish_pick_has_twins_only_in_the_other_spanish_close_by_picture():
    pick = {"language": "es", "same_card": "a", "sim": 0.95}
    la = {"language": "es-419", "same_card": "a", "sim": 0.95 - SURE_MARGIN / 2}
    far = {"language": "es-419", "same_card": "a", "sim": 0.95 - SURE_MARGIN * 2}
    other = {"language": "es-419", "same_card": "b", "sim": 0.95}
    en = {"language": "en", "same_card": "a", "sim": 0.95}
    assert spanish_twins([pick, la, far, other, en], pick) == [pick, la]
    assert spanish_twins([pick, far, other, en], pick) is None
    assert spanish_twins([en, pick], en) is None


def test_the_settings_default_to_spain_and_the_corner_and_the_flag_names_the_other():
    assert DEFAULTS["rules"]["default_spanish"] == "spanish"
    assert DEFAULTS["rules"]["spanish_corner"] is True
    ap = argparse.ArgumentParser()
    for flag in ("--keep-cut-off", "--read-language", "--no-language-check", "--switch-language"):
        ap.add_argument(flag, action="store_true")
    ap.add_argument("--default-spanish", choices=["spanish", "latin-american-spanish"])
    assert cli._rules(ap.parse_args(["--default-spanish", "latin-american-spanish"])) == {
        "default_spanish": "latin-american-spanish"}
    assert cli._rules(ap.parse_args([])) is None


def test_latin_american_spanish_shares_the_international_layout():
    assert "es-419" in LAYOUTS[0] and "es" in LAYOUTS[0]


def test_the_classifiers_spanish_answers_for_latin_american_spanish_when_it_is_the_only_spanish():
    spanish = [4.0 if lang == "es" else 0.0 for lang in CLASSIFIER_LANGUAGES]
    english = [4.0 if lang == "en" else 0.0 for lang in CLASSIFIER_LANGUAGES]
    assert read_from(spanish, {"es-419", "en"}, 0.9)[0] == "es-419"
    assert read_from(english, {"es-419", "en"}, 0.9)[0] == "en"
    assert read_from(spanish, {"es", "en"}, 0.9)[0] == "es"
    # both Spanish: the model's Spanish answers as Spain's, and only the corner's code moves a pick between them
    assert read_from(spanish, {"es", "es-419", "en"}, 0.9)[0] == "es"
    assert read_from(spanish, {"es-419"}, 0.9)[0] == "es-419"


def test_the_reader_reads_the_corner_on_a_spanish_pick_whatever_the_language_model_said():
    both = {"es", "es-419", "en"}
    assert reads_corner(None, "en", {"en", "fr"})
    assert reads_corner("en", "es", both)
    assert reads_corner("es", "en", both)
    assert not reads_corner("en", "en", both)
    assert not reads_corner("en", "es", {"es", "en"})
    # the corner's exact code beats a sure reading of another language; a mark alone doesn't
    assert corner_overrides("en", "es-419", "corner PBL LA")
    assert not corner_overrides("en", "es-419", "corner mark LA")
    assert corner_overrides("es", "es", "corner mark ES")
    assert not corner_overrides("en", None, "corner: nothing read")
    assert corner_overrides(None, None, "corner: nothing read")
    assert not corner_overrides("en", "fr", "corner FLF")


def test_an_exact_corner_read_holds_the_pick_against_its_other_spanish_print():
    read = lambda language, why: {"language": language, "why": why}
    es, la = {"language": "es", "same_card": "a"}, {"language": "es-419", "same_card": "a"}
    en = {"language": "en", "same_card": "a"}
    # another card of the same name (another number) in the other Spanish: the corner settles only the language
    other_la = {"language": "es-419", "same_card": "b"}
    assert corner_holds(read("es", "corner MEG ES"), es, la)
    assert corner_holds(read("es-419", "corner PBL LA"), la, es)
    assert not corner_holds(read("es", "corner MEG ES"), es, other_la)
    assert not corner_holds(read("es", "corner mark ES"), es, la)
    assert not corner_holds(read("es", "classifier 0.99"), es, la)
    assert not corner_holds(read("es-419", "corner PBL LA"), es, la)
    assert not corner_holds(read("es", "corner MEG ES"), es, en)
    assert not corner_holds(read("es", "corner MEG ES"), es, dict(es))
    assert not corner_holds(None, es, la)
