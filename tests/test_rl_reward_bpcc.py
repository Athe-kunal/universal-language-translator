"""Tests for rl.reward_components."""

from rl.reward_components import language_switch_penalty, repetition_penalty


def test_repetition_penalty_clean_text_is_zero():
    text = "यह एक सामान्य वाक्य है जो बिना किसी दोहराव के अनुवाद को दर्शाता है।"
    assert repetition_penalty(text) == 0.0


def test_repetition_penalty_short_text_is_zero():
    assert repetition_penalty("नमस्ते दुनिया") == 0.0


def test_repetition_penalty_short_phrase_loop_is_max():
    text = "लिए गए लिए गए लिए गए लिए गए लिए गए"
    assert repetition_penalty(text) == 1.0


def test_repetition_penalty_longer_loop_is_partial_not_max():
    phrase = "अ ब स द ई फ"
    text = " ".join([phrase, phrase, "जी हाँ बिलकुल सही बात है"])
    penalty = repetition_penalty(text)
    assert 0.0 < penalty < 1.0


def test_language_switch_penalty_pure_hindi_is_zero():
    text = "मुझे बाज़ार से सेब और केले खरीदने हैं।"
    assert language_switch_penalty(text) == 0.0


def test_language_switch_penalty_numbers_are_exempt():
    text = "मैं 2024 में 50% छूट पर 3.5 किलो सेब लाया।"
    assert language_switch_penalty(text) == 0.0


def test_language_switch_penalty_english_word_is_penalized():
    text = "मुझे यह laptop बहुत पसंद है क्योंकि यह तेज़ है।"
    penalty = language_switch_penalty(text)
    assert 0.0 < penalty < 1.0


def test_language_switch_penalty_all_english_is_max():
    text = "I really like this laptop because it is fast."
    assert language_switch_penalty(text) == 1.0


def test_language_switch_penalty_empty_text_is_zero():
    assert language_switch_penalty("") == 0.0
