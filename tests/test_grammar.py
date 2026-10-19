"""异步语法分析：契约、乱序回写、去重和降级。"""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.nlp.grammar import GrammarAnalyzer, GrammarContractError, validate_analysis_payload
from src.nlp.prompt_engine import PromptEngine
from src.profiles.loader import LanguageProfile
from src.subtitle.document import (
    ANALYSIS_FAILED,
    ANALYSIS_PENDING,
    ANALYSIS_READY,
    SessionDocument,
)


def _profile() -> LanguageProfile:
    return LanguageProfile(
        name="test",
        language="test",
        display_name="测试语言",
        source_lang="xx",
        target_lang="zh",
    )


def _prompts(tmp_path: Path) -> PromptEngine:
    default = tmp_path / "_default"
    default.mkdir()
    (default / "grammar.txt").write_text(
        "分析 {source_lang} → {target_lang}：{sentence}\n译文：{translation}",
        encoding="utf-8",
    )
    return PromptEngine(tmp_path)


class FakeProvider:
    def __init__(self, responder):
        self.cfg = SimpleNamespace(id="fake-grammar", model="fake-v1")
        self.responder = responder
        self.calls = 0

    def chat(self, prompt, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=self.responder(prompt))


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _segment(doc: SessionDocument, segment_id: int):
    segment = doc.get(segment_id)
    assert segment is not None
    return segment


def test_analysis_results_merge_by_id_when_out_of_order(tmp_path):
    doc = SessionDocument(language_profile="test", source_lang="xx", target_lang="zh")
    first = doc.add(0, 1, "第一句", "一")
    second = doc.add(1, 2, "第二句", "二")
    first_started = threading.Event()
    release_first = threading.Event()

    def respond(prompt):
        if "第一句" in prompt:
            first_started.set()
            assert release_first.wait(2)
            return json.dumps({"structure": {"custom": "第一句结果"}})
        assert first_started.wait(1)
        return json.dumps({"structure": {"custom": "第二句结果"}})

    analyzer = GrammarAnalyzer(
        FakeProvider(respond), _prompts(tmp_path), _profile(), doc, max_workers=2
    )
    assert analyzer.submit(first)
    assert analyzer.submit(second)
    assert _wait_until(lambda: _segment(doc,second).analysis_status == ANALYSIS_READY)
    assert _segment(doc,first).analysis_status == ANALYSIS_PENDING
    release_first.set()
    assert analyzer.drain()

    assert [s.id for s in doc.segments] == [first, second]
    assert _segment(doc,first).structure == {"custom": "第一句结果"}
    assert _segment(doc,second).structure == {"custom": "第二句结果"}
    assert _segment(doc,first).source == "第一句" and _segment(doc,first).translation == "一"
    assert doc.grammar_provider == "fake-grammar"
    assert doc.grammar_model == "fake-v1"
    assert doc.grammar_prompt_template_version > 0


def test_invalid_response_retries_once_then_only_marks_sentence_failed(tmp_path):
    doc = SessionDocument()
    segment_id = doc.add(0, 1, "坏句", None)
    provider = FakeProvider(lambda _prompt: "not-json")
    analyzer = GrammarAnalyzer(provider, _prompts(tmp_path), _profile(), doc, max_workers=1)

    assert analyzer.submit(segment_id)
    assert analyzer.drain()
    segment = _segment(doc,segment_id)
    assert provider.calls == 2
    assert segment.analysis_status == ANALYSIS_FAILED
    assert segment.structure is None and segment.translation is None


def test_submit_deduplicates_pending_and_ready_segments(tmp_path):
    doc = SessionDocument()
    segment_id = doc.add(0, 1, "句子", "译文")
    started = threading.Event()
    release = threading.Event()

    def respond(_prompt):
        started.set()
        release.wait(2)
        return '{"grammar_points": [{"custom_field": "值"}]}'

    analyzer = GrammarAnalyzer(FakeProvider(respond), _prompts(tmp_path), _profile(), doc)
    assert analyzer.submit(segment_id)
    assert started.wait(1)
    assert not analyzer.submit(segment_id)
    assert analyzer.pending_count == 1
    release.set()
    assert analyzer.drain()
    assert not analyzer.submit(segment_id)
    assert _segment(doc,segment_id).grammar_points == [{"custom_field": "值"}]


def test_bounded_queue_rejects_without_blocking_other_segments(tmp_path):
    doc = SessionDocument()
    first = doc.add(0, 1, "第一句", None)
    second = doc.add(1, 2, "第二句", None)
    started = threading.Event()
    release = threading.Event()

    def respond(_prompt):
        started.set()
        release.wait(2)
        return '{"structure": {}}'

    analyzer = GrammarAnalyzer(
        FakeProvider(respond),
        _prompts(tmp_path),
        _profile(),
        doc,
        max_workers=1,
        max_pending=1,
    )
    assert analyzer.submit(first)
    assert started.wait(1)
    assert not analyzer.submit(second)
    assert _segment(doc, second).analysis_status == ANALYSIS_FAILED
    release.set()
    assert analyzer.drain()
    assert _segment(doc, first).analysis_status == ANALYSIS_READY


def test_each_task_resolves_current_provider(tmp_path):
    doc = SessionDocument()
    first = doc.add(0, 1, "第一句", None)
    current = []

    def make_provider(provider_id):
        return FakeProvider(
            lambda _prompt: json.dumps({"extra": {"provider": provider_id}})
        )

    current.append(make_provider("provider-a"))
    analyzer = GrammarAnalyzer(
        lambda: current[0], _prompts(tmp_path), _profile(), doc, max_workers=1
    )
    analyzer.submit(first)
    assert _wait_until(lambda: analyzer.pending_count == 0)

    second = doc.add(1, 2, "第二句", None)
    current[0] = make_provider("provider-b")
    analyzer.submit(second)
    assert analyzer.drain()
    assert _segment(doc,first).extra == {"provider": "provider-a"}
    assert _segment(doc,second).extra == {"provider": "provider-b"}


def test_drain_timeout_cancels_queued_and_waits_for_running_task(tmp_path):
    doc = SessionDocument()
    first = doc.add(0, 1, "运行中", None)
    second = doc.add(1, 2, "排队中", None)
    started = threading.Event()
    release = threading.Event()

    def respond(_prompt):
        started.set()
        assert release.wait(1)
        return '{"structure": {"custom": "完成"}}'

    analyzer = GrammarAnalyzer(
        FakeProvider(respond),
        _prompts(tmp_path),
        _profile(),
        doc,
        max_workers=1,
    )
    assert analyzer.submit(first)
    assert analyzer.submit(second)
    assert started.wait(1)

    timer = threading.Timer(0.05, release.set)
    timer.start()
    assert not analyzer.drain(timeout=0.01)
    timer.join()

    assert _segment(doc, first).analysis_status == ANALYSIS_READY
    assert _segment(doc, second).analysis_status == ANALYSIS_FAILED
    assert analyzer.pending_count == 0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("grammar_points", {"not": "a list"}),
        ("grammar_points", {}),
        ("grammar_points", ""),
        ("vocabulary", {}),
        ("extra", []),
    ],
)
def test_validate_analysis_payload_rejects_wrong_container_types(key, value):
    with pytest.raises(GrammarContractError) as exc_info:
        validate_analysis_payload({key: value})
    assert key in str(exc_info.value)
