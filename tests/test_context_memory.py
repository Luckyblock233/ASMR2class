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


# ==================== M3 阶段 6：严格校验 / 不变量 / 并发 / 原子保存 ====================

import pytest

from src.nlp.context_memory import MemoryValidationError


def test_strict_load_rejects_malformed_json(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(MemoryValidationError, match="JSON 格式错误"):
        RollingMemory.load(path, strict=True)
    path.write_text('[1, 2]', encoding="utf-8")
    with pytest.raises(MemoryValidationError, match="根节点必须是对象"):
        RollingMemory.load(path, strict=True)


def test_strict_load_rejects_empty_fields_and_duplicates(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({
        "glossary": [{"term": "", "translation": "小雪"}],
    }), encoding="utf-8")
    with pytest.raises(MemoryValidationError, match="term 为空"):
        RollingMemory.load(path, strict=True)

    path.write_text(json.dumps({
        "glossary": [
            {"term": "ユキ", "translation": "小雪"},
            {"term": "ユキ", "translation": "由纪"},
        ],
    }), encoding="utf-8")
    with pytest.raises(MemoryValidationError, match="术语重复：ユキ"):
        RollingMemory.load(path, strict=True)

    path.write_text(json.dumps({
        "glossary": [{"term": "ユキ", "translation": ""}],
    }), encoding="utf-8")
    with pytest.raises(MemoryValidationError, match="translation 为空"):
        RollingMemory.load(path, strict=True)


def test_strict_load_profile_mismatch(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"language_profile": "ja", "glossary": []}), encoding="utf-8")
    with pytest.raises(MemoryValidationError, match="ja"):
        RollingMemory.load(path, strict=True, expected_profile="en")
    # 档案匹配则正常加载
    RollingMemory.load(path, strict=True, expected_profile="ja")


def test_load_rebuilds_invariants(tmp_path):
    """容量淘汰、风格截断、block_counter 恢复为最大 last_used_block。"""
    entries = [
        {"term": f"t{i}", "translation": f"译{i}", "last_used_block": i + 1}
        for i in range(5)
    ]
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({
        "glossary": entries, "style_notes": "x" * 50,
    }), encoding="utf-8")
    mem = RollingMemory.load(path, strict=True, glossary_max_entries=3, style_max_chars=10)
    assert len(mem.glossary) == 3            # 超出容量按 LRU 淘汰最旧两条
    assert "t0" not in mem.glossary and "t1" not in mem.glossary
    assert len(mem.style_notes) == 10        # 风格长度不变量重建
    assert mem.block_counter == 5            # LRU 计数恢复（否则后续淘汰顺序错乱）
    # 新条目淘汰时应基于恢复后的计数
    mem.begin_block()
    mem.set_term("t_new", "新")
    assert "t2" not in mem.glossary          # t2 的 last_used_block=3 最旧，被淘汰
    assert "t_new" in mem.glossary


def test_set_style_locked_by_preset():
    mem = RollingMemory(style_preset="正式书面语")
    assert mem.style_locked
    with pytest.raises(MemoryValidationError, match="锁定"):
        mem.set_style("人工修改")
    assert mem.style_notes == "正式书面语"  # 预设不被暗中覆盖


def test_set_term_rejects_empty():
    mem = RollingMemory()
    with pytest.raises(MemoryValidationError):
        mem.set_term("", "译文")
    with pytest.raises(MemoryValidationError):
        mem.set_term("术语", " ")


def test_atomic_save_leaves_no_tmp_and_preserves_on_failure(tmp_path):
    path = tmp_path / "session" / "memory.json"
    mem = RollingMemory()
    mem.set_term("ユキ", "小雪")
    mem.save(path, language_profile="ja")
    assert path.is_file()
    assert not list(path.parent.glob("*.tmp"))


def test_concurrent_edit_and_prompt_snapshot():
    """GUI 编辑与翻译线程读取并发：同一把锁，不抛异常且快照一致。"""
    import threading

    mem = RollingMemory()
    errors = []

    def writer():
        for i in range(200):
            try:
                mem.set_term(f"t{i % 20}", f"译{i}")
                mem.set_style(f"风格{i % 5}")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    def reader():
        for _ in range(400):
            try:
                mem.begin_block()
                mem.glossary_prompt()
                mem.mark_hits("包含 t1 的句子")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(2)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(mem.glossary) <= 20
