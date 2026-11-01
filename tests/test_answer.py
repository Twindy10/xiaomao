from xiaomao.generation.answer import refusal_text


def test_refusal_names_scope_in_chinese():
    text = refusal_text("temu", "DE")
    assert text.startswith("暂未找到与该问题相关的资料")
    assert "Temu · 德国站" in text
    assert "temu" not in text


def test_refusal_without_scope_says_all():
    assert "全部平台和站点" in refusal_text(None, None)


def test_refusal_keeps_unknown_codes():
    assert "AU" in refusal_text(None, "AU")
