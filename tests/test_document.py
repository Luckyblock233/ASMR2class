"""句段稳定 ID、分析落盘和会话重载。"""

import json
from pathlib import Path

from src.profiles.loader import get_profile
from src.subtitle.document import ANALYSIS_READY, load_session_document, parse_srt_timestamp
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

    payload = json.loads((out_dir / "analysis.ja.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in payload["segments"]] == [first, second]


def test_reload_falls_back_to_bilingual_srt(tmp_path):
    recorder = SessionRecorder(JA)
    recorder.add(0, 1, "原文", "译文")
    out_dir = recorder.save(tmp_path, "srt-only")
    assert out_dir is not None
    (out_dir / "analysis.ja.json").unlink()

    loaded = load_session_document(out_dir)
    assert len(loaded) == 1
    assert _segment(loaded,1).source == "原文"
    assert _segment(loaded,1).translation == "译文"


def test_parse_srt_timestamp_supports_comma_and_dot():
    assert parse_srt_timestamp("01:02:03,250") == 3723.25
    assert parse_srt_timestamp("00:00:01.500") == 1.5
