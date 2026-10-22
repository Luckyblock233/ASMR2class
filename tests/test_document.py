"""句段稳定 ID、分析落盘和会话重载。"""

import json
from pathlib import Path

from src.profiles.loader import get_profile
from src.subtitle.document import (
    ANALYSIS_READY,
    TRANSLATION_FAILED,
    TRANSLATION_NONE,
    TRANSLATION_PENDING,
    TRANSLATION_READY,
    load_session_document,
    parse_srt_timestamp,
)
from src.subtitle.session import SessionRecorder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JA = get_profile(PROJECT_ROOT / "config" / "languages", "ja")


def _segment(document, segment_id: int):
    segment = document.get(segment_id)
    assert segment is not None
    return segment


def test_analysis_roundtrip_preserves_ids_order_and_metadata(tmp_path):
    recorder = SessionRecorder(JA, prompt_template_version=11)
    first = recorder.add(0, 1.2, "第一句", "一")
    second = recorder.add(1.5, 2.8, "第二句", "二")
    # 模拟第二句先完成，保存顺序仍应按句段而非完成顺序。
    recorder.document.apply_analysis(
        second,
        {
            "structure": {"custom": {"nested": True}},
            "grammar_points": [{"custom_field": "任意语言字段"}],
            "vocabulary": [],
            "extra": {},
        },
    )
    recorder.document.update_grammar_metadata(
        template_version=22, provider_id="grammar-x", model="model-y"
    )

    out_dir = recorder.save(tmp_path, "roundtrip")
    assert out_dir is not None
    loaded = load_session_document(out_dir)
    assert [s.id for s in loaded.segments] == [first, second]
    assert [s.source for s in loaded.segments] == ["第一句", "第二句"]
    assert _segment(loaded,first).analysis_status != ANALYSIS_READY
    assert _segment(loaded,second).analysis_status == ANALYSIS_READY
    assert _segment(loaded,second).structure == {"custom": {"nested": True}}
    assert _segment(loaded,second).grammar_points == [{"custom_field": "任意语言字段"}]
    assert loaded.language_profile == "ja" and loaded.source_lang == "ja"
    assert loaded.prompt_template_version == 11
    assert loaded.grammar_prompt_template_version == 22
    assert loaded.grammar_provider == "grammar-x"
    assert loaded.grammar_model == "model-y"

    payload = json.loads((out_dir / "analysis.ja.zh.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in payload["segments"]] == [first, second]


def test_reload_falls_back_to_bilingual_srt(tmp_path):
    recorder = SessionRecorder(JA)
    recorder.add(0, 1, "原文", "译文")
    out_dir = recorder.save(tmp_path, "srt-only")
    assert out_dir is not None
    (out_dir / "analysis.ja.zh.json").unlink()

    loaded = load_session_document(out_dir)
    assert len(loaded) == 1
    assert _segment(loaded,1).source == "原文"
    assert _segment(loaded,1).translation == "译文"


def test_parse_srt_timestamp_supports_comma_and_dot():
    assert parse_srt_timestamp("01:02:03,250") == 3723.25
    assert parse_srt_timestamp("00:00:01.500") == 1.5


def test_translation_placeholder_then_update_by_id(tmp_path):
    """ASR 句先登记占位，翻译完成/失败按稳定 ID 回写（M3 后续阶段 1）。"""
    recorder = SessionRecorder(JA)
    first = recorder.add_pending(0.0, 1.2, "第一句", asr_ms=120.0)
    second = recorder.add_pending(1.5, 2.8, "第二句", asr_ms=130.0)

    pending = _segment(recorder.document, first)
    assert pending.translation is None
    assert pending.translation_status == TRANSLATION_PENDING
    assert pending.asr_ms == 120.0

    # 第二句先失败、第一句后成功：互不影响
    recorder.update_translation(
        second, None, status=TRANSLATION_FAILED, error="服务不可用", translate_ms=800.0
    )
    recorder.update_translation(first, "一", status=TRANSLATION_READY, translate_ms=800.0)
    ok = _segment(recorder.document, first)
    failed = _segment(recorder.document, second)
    assert ok.translation == "一" and ok.translation_status == TRANSLATION_READY
    assert ok.translate_ms == 800.0
    assert failed.translation is None
    assert failed.translation_status == TRANSLATION_FAILED
    assert failed.translation_error == "服务不可用"

    # 状态/耗时随分析 JSON 落盘并可重载
    out_dir = recorder.save(tmp_path, "placeholder")
    assert out_dir is not None
    loaded = load_session_document(out_dir)
    assert _segment(loaded, first).translation_status == TRANSLATION_READY
    assert _segment(loaded, first).asr_ms == 120.0
    assert _segment(loaded, second).translation_status == TRANSLATION_FAILED
    assert _segment(loaded, second).translation_error == "服务不可用"


def test_old_analysis_json_without_translation_fields_loads(tmp_path):
    """旧版分析文件缺 translation_status/asr_ms 等字段时按译文存在性推断。"""
    from src.subtitle.document import SessionDocument
    from src.subtitle.writer import write_analysis_json

    doc = SessionDocument(language_profile="ja", source_lang="ja", target_lang="zh")
    doc.add(0.0, 1.0, "有译文", "一")
    doc.add(1.5, 2.5, "无译文", None)
    path = tmp_path / "analysis.ja.zh.json"
    write_analysis_json(
        doc.segments, path, language_profile="ja", source_lang="ja", target_lang="zh"
    )
    # 模拟旧格式：删掉新增字段
    payload = json.loads(path.read_text(encoding="utf-8"))
    for seg in payload["segments"]:
        for key in ("translation_status", "translation_error", "asr_ms", "translate_ms"):
            seg.pop(key)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    loaded = load_session_document(tmp_path)
    assert _segment(loaded, 1).translation_status == TRANSLATION_READY
    assert _segment(loaded, 2).translation_status == TRANSLATION_NONE
    assert _segment(loaded, 1).asr_ms == 0.0


def test_segment_at_time_lookup():
    recorder = SessionRecorder(JA)
    recorder.add(0.0, 1.0, "一", "1")
    recorder.add(2.0, 3.0, "二", "2")
    recorder.add(5.0, 6.5, "三", "3")
    doc = recorder.document

    assert doc.segment_at(0.5).id == 1
    assert doc.segment_at(1.0).id == 1      # 句尾边界含在内
    assert doc.segment_at(2.0).id == 2      # 句首边界含在内
    assert doc.segment_at(1.5) is None      # 句间空白
    assert doc.segment_at(6.4).id == 3
    assert doc.segment_at(6.6) is None
    assert doc.segment_at(100.0) is None
    # 时间查询不改变稳定 ID 主键
    assert doc.get(2).source == "二"
