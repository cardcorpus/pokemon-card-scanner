"""The language reader's rule on lines as an OCR engine reads them: the same cases as ScannerKit's
LanguageReaderTests, so the two read alike."""
from cardcorpus_scanner.language import CLASSIFIER_LANGUAGES, decide, guess_from, language_check


def lines(texts, top=0.6):
    return [{"t": t, "c": 1.0, "top": top} for t in texts]


def test_hangul_in_the_text_box_is_korean():
    reading = {"text": {"ko-KR": lines(["상대의 배틀포켓몬에게"]), "ja-JP": lines(["0 1ハ"])}}
    assert decide(reading, {"ja", "ko"})[0] == "ko"


def test_kana_in_the_text_box_is_japanese():
    reading = {"text": {"ko-KR": lines(["1"]), "ja-JP": lines(["相手のバトルポケモンにダメージ"])}}
    assert decide(reading, {"ja", "ko"})[0] == "ja"


def test_simplified_only_characters_are_simplified():
    assert decide({"text": {"zh-Hans": lines(["对手的战斗宝可梦造成伤害"])}}, {"zh-Hans", "zh-Hant"})[0] == "zh-Hans"


def test_traditional_only_characters_are_traditional():
    assert decide({"text": {"zh-Hans": lines(["對手的戰鬥寶可夢造成傷害"])}}, {"zh-Hans", "zh-Hant"})[0] == "zh-Hant"


def test_the_codes_suffix_outweighs_a_few_characters():
    reading = {"text": {"zh-Hans": lines(["伊布", "这"])}, "code": lines(["M6a F 106/080"], top=0.9)}
    assert decide(reading, {"zh-Hans", "zh-Hant"})[0] == "zh-Hant"


def test_characters_both_scripts_share_settle_nothing():
    assert decide({"text": {"zh-Hans": lines(["伊布", "皮皮"])}}, {"zh-Hans", "zh-Hant"})[0] is None


def test_a_language_mark_names_a_latin_card():
    reading = {"text": {"latin": lines(["Flip a coin. If heads, your opponent's Active Pokémon"])},
               "code": lines(["PAF DE 089/091"], top=0.9)}
    assert decide(reading, {"en", "de"})[0] == "de"


def test_function_words_name_a_latin_card_without_a_mark():
    reading = {"text": {"latin": lines(["Lancez une pièce. Si c'est face, le Pokémon Actif de votre adversaire"])}}
    assert decide(reading, {"en", "fr", "de"})[0] == "fr"


def test_a_language_not_asked_about_is_no_answer():
    assert decide({"text": {"ko-KR": lines(["상대의 배틀포켓몬에게"])}}, {"zh-Hans", "zh-Hant"})[0] is None


def test_the_reader_asks_only_about_the_same_card_in_other_languages_alike_by_picture():
    from cardcorpus_scanner.scanner import languages_in_play
    pick = {"name": "Eevee", "language": "zh-Hans", "same_card": "c1", "sim": 0.95}
    assert languages_in_play([pick, {"name": "Eevee", "language": "zh-Hant", "same_card": "c1", "sim": 0.94},
                              {"name": "Pikachu", "language": "ja", "same_card": "c2", "sim": 0.94}], pick) == {"zh-Hans", "zh-Hant"}
    assert languages_in_play([pick, {"language": "zh-Hant", "same_card": "c1", "sim": 0.85}], pick) == set()
    assert languages_in_play([pick, {"language": "fr", "same_card": "c2", "sim": 0.95}], pick) == set()
    assert languages_in_play([pick, {"language": "en", "same_card": "c1", "sim": 0.95}], pick) == set()


GYARADOS = {"print": "base", "name": "Gyarados", "number": "6/102", "language": "en", "artwork_group": "a", "same_card": "c1", "sim": 0.90}
REPRINT = {"print": "evo-es", "name": "Gyarados", "number": "34/108", "language": "es", "artwork_group": "a", "same_card": "c2", "sim": 0.89}
FRENCH = {"print": "base-fr", "name": "Gyarados", "number": "6/102", "language": "fr", "artwork_group": "a", "same_card": "c1", "sim": 0.88}


def reader(found):
    from cardcorpus_scanner.scanner import Scanner

    class Read:
        def read(self, crop, languages):
            return found, "test"
    scanner = Scanner.__new__(Scanner)
    scanner.language_classifier = Read()
    scanner.corner_reader = None
    return scanner


def test_a_reprint_in_another_set_is_not_the_cards_other_language():
    from cardcorpus_scanner.scanner import languages_in_play
    assert languages_in_play([GYARADOS, REPRINT], GYARADOS) == set()
    cands = [GYARADOS, REPRINT, FRENCH]
    assert languages_in_play(cands, GYARADOS) == {"en", "fr"}
    order, out = reader("es").read_language({"cands": cands, "hi": None}, cands)
    assert order[0] is GYARADOS and "language_switched" not in out


def test_the_language_read_moves_the_pick_to_that_print_of_the_card():
    cands = [GYARADOS, REPRINT, FRENCH]
    order, out = reader("fr").read_language({"cands": cands, "hi": None}, cands)
    assert order[0] is FRENCH and out["language_switched"]


def test_a_card_with_no_links_is_not_read():
    from cardcorpus_scanner.scanner import languages_in_play
    pick = {"name": "Eevee", "language": "en", "sim": 0.9}
    assert languages_in_play([pick, {"name": "Eevee", "language": "fr", "sim": 0.9}], pick) == set()


def logits_for(language, lead):
    return [lead if name == language else 0.0 for name in CLASSIFIER_LANGUAGES]


def test_the_check_names_a_language_the_model_is_sure_of_when_the_pick_differs():
    check = language_check(guess_from(logits_for("de", 12.0)), "en", 0.95)
    assert check["suggested"] == "de" and check["confidence"] > 0.99


def test_the_check_says_nothing_when_the_pick_is_in_that_language():
    assert language_check(guess_from(logits_for("de", 12.0)), "de", 0.95) is None


def test_the_check_says_nothing_when_the_model_is_unsure_or_the_pick_has_no_language():
    assert language_check(guess_from(logits_for("de", 1.0)), "en", 0.95) is None
    assert language_check(guess_from(logits_for("de", 12.0)), None, 0.95) is None


def test_one_letter_words_alone_name_no_language():
    assert decide({"text": {"latin": lines(["a e o a e o .. a"])}}, {"en", "pt"})[0] is None
    assert decide({"text": {"latin": lines(["o dano a um dos seu oponente"])}}, {"en", "pt"})[0] == "pt"


# the corner reader's rule (corner.py); ScannerKit's port should pass the same cases
def corner(codes):
    from cardcorpus_scanner.corner import CornerReader
    c = CornerReader.__new__(CornerReader)
    c.codes, c.min_votes = codes, 5
    return c


MEW = {"set": "english/scarlet-violet-151", "catalog": "english", "language": "en"}
MEW_FR = {"set": "scarlet-violet-151", "catalog": "french", "language": "fr"}
M4_JA = {"set": "japanese/m4", "catalog": "japanese", "language": "ja"}
M4_TH = {"set": "thai/m4", "catalog": "thai", "language": "th"}
CODES = {"english/scarlet-violet-151": "MEW EN", "french/scarlet-violet-151": "MEW FR", "japanese/m4": "M4", "thai/m4": "M4 T"}


def test_a_printed_code_and_mark_settle_the_language():
    assert corner(CODES).settle({"text": "MEW FR", "votes": 20}, [MEW, MEW_FR], {"en", "fr"})[0] == "fr"


def test_a_read_without_its_mark_settles_nothing():
    assert corner(CODES).settle({"text": "M4", "votes": 40}, [M4_JA, M4_TH], {"ja", "th"})[0] is None


def test_a_mark_no_print_in_play_has_settles_nothing():
    assert corner(CODES).settle({"text": "MEW DE", "votes": 40}, [MEW, MEW_FR], {"en", "fr"})[0] is None


def test_few_votes_or_promo_settle_nothing():
    assert corner(CODES).settle({"text": "MEW FR", "votes": 4}, [MEW, MEW_FR], {"en", "fr"})[0] is None
    assert corner({**CODES, "french/scarlet-violet-151": "PROMO"}).settle({"text": "PROMO", "votes": 40}, [MEW, MEW_FR], {"en", "fr"})[0] is None


def test_simplified_chinese_codes_end_in_c():
    from cardcorpus_scanner.corner import marked
    assert marked("151C") and marked("CSV9.5C") and marked("CMPC") and marked("M6A F")
    assert not marked("SV2A") and not marked("PW") and not marked("PMC") and not marked("PROMC")


def test_the_mark_alone_names_a_language_whose_set_has_no_code_on_record():
    a = {"set": "zh-hans/x", "catalog": "zh-hans", "language": "zh-Hans"}
    b = {"set": "zh-hant/y", "catalog": "zh-hant", "language": "zh-Hant"}
    assert corner({}).settle({"text": "30THC", "votes": 30}, [a, b], {"zh-Hans", "zh-Hant"})[0] == "zh-Hans"
    assert corner({"zh-hans/x": "CBB1C"}).settle({"text": "30THC", "votes": 30}, [a, b], {"zh-Hans", "zh-Hant"})[0] is None


def test_the_datasets_marks_name_the_languages():
    # corner-reader.json carries the dataset's marks (vocabulary.json `codeMark`): Latin American Spanish's "LA"
    from cardcorpus_scanner.corner import MARKS, mark_language, marked
    marks = {**MARKS, "LA": "es-419"}
    assert marked("PBL LA", marks) and not marked("PBL LA")
    assert mark_language("PBL LA", marks) == "es-419"
    assert mark_language("151C") == "zh-Hans" and mark_language("M4 T") == "th"
    la = {"set": "latin-american-spanish/pitch-black", "catalog": "latin-american-spanish", "language": "es-419"}
    es = {"set": "spanish/pitch-black", "catalog": "spanish", "language": "es"}
    c = corner({"latin-american-spanish/pitch-black": "PBL LA", "spanish/pitch-black": "PBL ES"})
    c.marks = marks
    assert c.settle({"text": "PBL LA", "votes": 30}, [la, es], {"es", "es-419"})[0] == "es-419"
    assert c.settle({"text": "PBL ES", "votes": 30}, [la, es], {"es", "es-419"})[0] == "es"
