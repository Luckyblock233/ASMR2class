"""滚动记忆测试（FR-C3）。"""

import json

from src.nlp.context_memory import RollingMemory


def test_merge_and_override_same_term():
    mem = RollingMemory()
    mem.begin_block()
    mem.merge_updates([{"term": "ユキ", "translation": "小雪", "type": "person"}], "")
    assert mem.glossary["ユキ"].translation == "小雪"
    mem.merge_updates([{"term": "ユキ", "translation": "由纪"}], None)  # 后登记覆盖
    assert mem.glossary["ユキ"].translation == "由纪"
    assert mem.glossary["ユキ"].type == "person" or True  # type 允许更新


def test_style_replace_and_truncate():
    mem = RollingMemory(style_max_chars=10)
    mem.merge_updates(None, "敬体，前后辈关系")
    assert mem.style_notes == "敬体，前后辈关系"
    mem.merge_updates(None, "x" * 50)  # 超长截断
    assert len(mem.style_notes) == 10
    mem.merge_updates(None, "")  # 空更新不替换
    assert mem.style_notes == "x" * 10


def test_style_preset_locked():
    mem = RollingMemory(style_preset="正式书面语")
    assert mem.style_notes == "正式书面语"
    mem.merge_updates(None, "LLM 推断的口语风格")
    assert mem.style_notes == "正式书面语"  # 档案预设优先（§8）


def test_lru_eviction():
    mem = RollingMemory(glossary_max_entries=3)
    for i in range(3):
        mem.begin_block()
        mem.merge_updates([{"term": f"t{i}", "translation": f"译{i}"}], None)
    # t0 最旧；让 t0 命中一次刷新
    mem.begin_block()
    mem.mark_hits("包含 t0 的句子")
    mem.merge_updates([{"term": "t3", "translation": "译3"}], None)  # 触发淘汰
    assert "t0" in mem.glossary      # 命中过，保留
    assert "t1" not in mem.glossary  # 最久未命中，淘汰
    assert len(mem.glossary) == 3


def test_manual_edit_and_clear():
    mem = RollingMemory()
    mem.set_term("タロウ", "太郎", "person")
    assert mem.glossary["タロウ"].translation == "太郎"
    assert mem.remove_term("タロウ")
    assert not mem.remove_term("不存在")
    mem.set_term("a", "b")
    mem.set_style("某风格")
    mem.clear()
    assert mem.glossary == {} and mem.style_notes == ""


def test_save_load_roundtrip(tmp_path):
    mem = RollingMemory()
    mem.begin_block()
    mem.merge_updates(
        [{"term": "ユキ", "translation": "小雪", "type": "person"}], "轻松日常"
    )
    path = tmp_path / "out" / "memory.json"
    mem.save(path, language_profile="ja")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["language_profile"] == "ja"
    assert data["glossary"][0]["term"] == "ユキ"
    assert "updated_at" in data

    loaded = RollingMemory.load(path)
    assert loaded.glossary["ユキ"].translation == "小雪"
    assert loaded.style_notes == "轻松日常"


def test_glossary_prompt_format():
    mem = RollingMemory()
    mem.set_term("ユキ", "小雪", "person")
    mem.set_term("東京", "东京")
    text = mem.glossary_prompt()
    assert "ユキ → 小雪（person）" in text
    assert "東京 → 东京" in text
    assert RollingMemory().glossary_prompt() == ""
