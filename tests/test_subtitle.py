"""字幕文件生成与解析测试（FR-2）。"""

import json
from pathlib import Path

from src.nlp.context_memory import RollingMemory
from src.profiles.loader import get_profile
from src.subtitle.parser import load_srt
from src.subtitle.session import SessionRecorder
from src.subtitle.writer import (
    SubtitleEntry,
    make_session_dir,
    sanitize_filename,
    srt_timestamp,
    write_srt,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JA = get_profile(PROJECT_ROOT / "config" / "languages", "ja")


def test_srt_timestamp():
    assert srt_timestamp(1.2) == "00:00:01,200"
    assert srt_timestamp(3661.5) == "01:01:01,500"
    assert srt_timestamp(0) == "00:00:00,000"
    assert srt_timestamp(-1) == "00:00:00,000"


def test_write_original_and_bilingual_srt(tmp_path):
    entries = [
        SubtitleEntry(1.2, 4.5, "原文一", "译文一"),
        SubtitleEntry(4.6, 8.0, "原文二", None),  # 降级句无译文
    ]
    orig = tmp_path / "original.srt"
    bi = tmp_path / "bilingual.srt"
    write_srt(entries, orig, bilingual=False)
    write_srt(entries, bi, bilingual=True)

    orig_text = orig.read_text(encoding="utf-8")
    assert "1\n00:00:01,200 --> 00:00:04,500\n原文一" in orig_text
    assert "译文一" not in orig_text

    bi_text = bi.read_text(encoding="utf-8")
    assert "原文一\n译文一" in bi_text
    # 无译文句双语文件里只有原文行
    assert "原文二\n\n" in bi_text or bi_text.rstrip().endswith("原文二")


def test_srt_roundtrip_via_parser(tmp_path):
    entries = [SubtitleEntry(1.2, 4.5, "原文一", "译文一"), SubtitleEntry(4.6, 8.0, "原文二", "译文二")]
    path = tmp_path / "bilingual.srt"
    write_srt(entries, path, bilingual=True)
    loaded = load_srt(path)
    assert len(loaded) == 2
    assert loaded[0].start == 1.2 and loaded[0].end == 4.5
    assert loaded[0].source == "原文一" and loaded[0].translation == "译文一"


def test_session_recorder_saves_all_files(tmp_path):
    rec = SessionRecorder(JA, prompt_template_version=3)
    rec.add(0.0, 1.5, "おはよう", "早上好")
    rec.add(2.0, 3.0, "さようなら", None)
    mem = RollingMemory()
    mem.set_term("ユキ", "小雪", "person")

    out_dir = rec.save(tmp_path / "output", "测试 会话", memory=mem)
    assert out_dir is not None
    assert out_dir.parent == tmp_path / "output"
    # FR-2.4 文件清单 + FR-C3.6 memory.json
    assert (out_dir / "transcript.txt").is_file()
    assert (out_dir / "original.srt").is_file()
    assert (out_dir / "bilingual.srt").is_file()
    assert (out_dir / "analysis.ja.json").is_file()  # 分析文件名带语言标识
    assert (out_dir / "memory.json").is_file()

    txt = (out_dir / "transcript.txt").read_text(encoding="utf-8")
    assert txt == "おはよう\nさようなら\n"

    analysis = json.loads((out_dir / "analysis.ja.json").read_text(encoding="utf-8"))
    assert analysis["language_profile"] == "ja"
    assert analysis["prompt_template_version"] == 3
    assert len(analysis["segments"]) == 2
    seg = analysis["segments"][0]
    assert seg["source"] == "おはよう" and seg["translation"] == "早上好"
    assert seg["structure"] is None and seg["grammar_points"] == []

    memory = json.loads((out_dir / "memory.json").read_text(encoding="utf-8"))
    assert memory["glossary"][0]["term"] == "ユキ"


def test_empty_session_no_output(tmp_path):
    rec = SessionRecorder(JA)
    assert rec.save(tmp_path / "output", "x") is None
    assert not (tmp_path / "output").exists()


def test_sanitize_and_collision(tmp_path):
    assert sanitize_filename("a/b:c d") == "a-b-c-d"
    d1 = make_session_dir(tmp_path, "live")
    d2 = make_session_dir(tmp_path, "live")  # 重名自动加后缀
    assert d1 != d2 and d2.name.startswith(d1.name)
