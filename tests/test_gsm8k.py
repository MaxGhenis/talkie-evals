from talkie_evals.gsm8k import extract_prediction, gold_answer, normalize_number


def test_normalize_number_strips_common_formatting():
    assert normalize_number("$1,200.00") == "1200"
    assert normalize_number("45%") == "45"
    assert normalize_number("0.5000") == "0.5"


def test_gold_answer_extracts_gsm8k_marker():
    assert gold_answer("Some reasoning.\n#### 1,234") == "1234"


def test_extract_prediction_prefers_final_answer_language():
    text = "First I get 10. Therefore, the answer is $12."
    assert extract_prediction(text) == "12"
