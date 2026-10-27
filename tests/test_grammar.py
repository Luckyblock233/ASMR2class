"""异步语法分析：契约、乱序回写、去重和降级。"""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.nlp.grammar import GrammarAnalyzer, GrammarContractError, validate_analysis_payload
from src.nlp.prompt_engine import PromptEngine
from src.profiles.loader import LanguageProfile, get_profile
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


# ==================== 阶段进度 / 错误分类 / --debug ====================


def _run_analyzer(provider, debug=None):
    """跑一次单句分析（同步等待完成），返回 (document, segment_id, debug输出)。"""
    import json as _json

    from src.nlp.llm_client import ChatResult
    from src.nlp.prompt_engine import PromptEngine
    from src.subtitle.document import SessionDocument

    prompts = PromptEngine(PROJECT_ROOT / "config" / "prompts")
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    doc = SessionDocument(language_profile="ja", source_lang="ja", target_lang="zh")
    sid = doc.add(0.0, 1.0, "彼は行っています", "他正要去")
    debug_out = []
    analyzer = GrammarAnalyzer(
        provider, prompts, profile, doc,
        debug=(lambda text: debug_out.append(text)) if debug is not None else None,
    )
    analyzer.submit(sid)
    analyzer.drain()
    return doc, sid, debug_out


def test_grammar_debug_logs_exchange():
    """--debug：语法分析的 prompt/响应/耗时进入调试输出。"""
    import json

    from src.nlp.llm_client import ChatResult

    class P:
        cfg = None
        online = True

        def chat(self, prompt, **kw):
            return ChatResult(json.dumps({
                "structure": {"type": "主谓"}, "grammar_points": [],
                "vocabulary": [], "extra": {},
            }, ensure_ascii=False), prompt_tokens=5, completion_tokens=10, latency_sec=0.3)

    doc, sid, debug_out = _run_analyzer(P(), debug=True)
    assert doc.get(sid).analysis_status == "ready"
    assert len(debug_out) == 1
    text = debug_out[0]
    assert "语法分析 句段#" in text
    assert "[请求 prompt]" in text and "彼は行っています" in text
    assert "[响应]" in text and "tokens 5+10=15" in text
    assert "主谓" in text


def test_grammar_stage_progression():
    """成功路径的阶段推进：queued → rendering → requesting → parsing →（完成清空）。"""
    import json

    from src.nlp.grammar import STAGE_PARSING, STAGE_RENDERING, STAGE_REQUESTING
    from src.nlp.llm_client import ChatResult
    from src.nlp.prompt_engine import PromptEngine
    from src.subtitle.document import SessionDocument

    stages = []

    class P:
        cfg = None
        online = True

        def chat(self, prompt, **kw):
            doc = stage_probe["doc"]
            sid = stage_probe["sid"]
            stages.append(doc.get(sid).analysis_stage)
            return ChatResult(json.dumps({"structure": None, "grammar_points": [],
                                          "vocabulary": [], "extra": {}}))

    prompts = PromptEngine(PROJECT_ROOT / "config" / "prompts")
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    doc = SessionDocument(language_profile="ja", source_lang="ja", target_lang="zh")
    sid = doc.add(0.0, 1.0, "テスト", "测试")
    stage_probe = {"doc": doc, "sid": sid}
    analyzer = GrammarAnalyzer(P(), prompts, profile, doc)
    analyzer.submit(sid)
    analyzer.drain()
    assert stages == [STAGE_REQUESTING]  # chat 时处于 requesting
    assert doc.get(sid).analysis_stage == ""  # 完成后清空


def test_grammar_error_classification():
    """失败分类：契约/网络/未绑定，原因记录在句段上。"""
    import json

    from src.nlp.llm_client import ChatResult
    from src.nlp.prompt_engine import PromptEngine
    from src.subtitle.document import ANALYSIS_FAILED, SessionDocument
    from src.nlp.grammar import GrammarAnalyzer
    from openai import APIConnectionError

    prompts = PromptEngine(PROJECT_ROOT / "config" / "prompts")
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")

    # 契约错误：模型返回非契约 JSON
    class BadJSON:
        cfg = None
        online = True

        def chat(self, prompt, **kw):
            return ChatResult("not json at all")

    doc, sid, _ = _run_analyzer(BadJSON())
    assert doc.get(sid).analysis_status == ANALYSIS_FAILED
    assert "解析错误" in doc.get(sid).analysis_error  # 纯文本非 JSON → JSONDecodeError → 解析错误

    class BadContract:
        cfg = None
        online = True

        def chat(self, prompt, **kw):
            return ChatResult(json.dumps({"grammar_points": "not-a-list"}))

    doc, sid, _ = _run_analyzer(BadContract())
    assert "契约" in doc.get(sid).analysis_error

    class NetErr:
        cfg = None
        online = True

        def chat(self, prompt, **kw):
            raise APIConnectionError(request=None)

    doc, sid, _ = _run_analyzer(NetErr())
    assert "模型调用错误" in doc.get(sid).analysis_error
    assert "网络" in doc.get(sid).analysis_error

    # 未绑定
    doc, sid, _ = _run_analyzer(None)
    assert "未绑定" in doc.get(sid).analysis_error


def test_extract_json_tolerates_trailing_junk():
    """raw_decode：合法 JSON 后尾随杂散文本（含花括号）都能取到首个对象。"""
    from src.nlp.translate import _extract_json

    good = '{"structure": {"type": "t"}, "extra": {}}'
    assert _extract_json(good)["structure"]["type"] == "t"
    assert _extract_json(good + "\n模型多说的解释……")["structure"]["type"] == "t"
    assert _extract_json(good + '\n另一个对象 {"a": 1}')["structure"]["type"] == "t"
    assert _extract_json(f"```json\n{good}\n```")["structure"]["type"] == "t"


def test_extra_string_wrapped_into_object():
    """模型把 extra 输出为字符串时包装为对象，不判整句失败。"""
    payload = validate_analysis_payload({
        "structure": None, "grammar_points": [], "vocabulary": [],
        "extra": "这句话是口语省略句",
    })
    assert payload["extra"] == {"note": "这句话是口语省略句"}
