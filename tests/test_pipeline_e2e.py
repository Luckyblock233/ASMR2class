"""端到端测试：FileSource → RingBuffer → ASRStreamer(真实 Silero VAD + whisper tiny) → Translator(stub)。

需要首次运行时下载 whisper tiny 模型（约 75MB，经 HF_ENDPOINT 镜像）。
设置环境变量 ASR2CLASS_E2E=1 才运行真实模型用例。
"""

import json
import os
import re
import time
from pathlib import Path

import pytest

from src.asr.engine import ASREngine
from src.asr.streamer import ASRStreamer, Sentence
from src.audio.base import RingBuffer
from src.audio.file_source import FileSource
from src.config import ASRConfig, SegmentationConfig
from src.nlp.llm_client import ChatResult
from src.nlp.prompt_engine import PromptEngine
from src.nlp.translate import Translator
from src.profiles.loader import get_profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = PROJECT_ROOT / "tests" / "data" / "sample_en.flac"

pytestmark = pytest.mark.skipif(
    os.environ.get("ASR2CLASS_E2E") != "1", reason="端到端用例需 ASR2CLASS_E2E=1（下载模型）"
)


class StubLLM:
    """回显式 stub：按块级 JSON 契约返回，每句译文固定。"""

    online = True

    def chat(self, prompt, **kw):
        m = re.search(r"\[\{.*\}\]", prompt, re.DOTALL)
        items = json.loads(m.group(0)) if m else []
        payload = {
            "sentences": [
                {"id": it["id"], "source": it["source"], "translation": "模拟译文"}
                for it in items
            ],
            "glossary_updates": [],
            "style_updates": "",
        }
        return ChatResult(json.dumps(payload, ensure_ascii=False), 5, 2)


def test_e2e_file_to_translated_sentences():
    engine = ASREngine(ASRConfig(device="cpu", model_size="tiny", compute_type="int8", beam_size=1))
    assert engine.info.device == "cpu"

    ring = RingBuffer(16000 * 120)
    sentences: list[Sentence] = []
    source = FileSource(SAMPLE, realtime=False)

    streamer = ASRStreamer(
        engine,
        SegmentationConfig(silence_ms=500, max_segment_sec=15),
        language_hint="en",
        on_sentence=sentences.append,
    )
    source.start(ring.write)
    streamer.start(ring)
    source.finished.wait(timeout=30)
    time.sleep(0.3)
    source.stop()
    streamer.stop()

    assert len(sentences) >= 1
    full_text = " ".join(s.text for s in sentences).lower()
    assert "stew" in full_text or "dinner" in full_text
    # 时间戳单调且在音频时长内
    assert sentences[0].start >= 0
    assert sentences[-1].end <= 12.0
    for a, b in zip(sentences, sentences[1:]):
        assert a.start <= b.start

    # 翻译层：stub LLM + 真实模板渲染，走完整块级路径（FR-C2）
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "en")
    translator = Translator(StubLLM(), PromptEngine(PROJECT_ROOT / "config" / "prompts"), profile)
    results = translator.translate_block([s.text for s in sentences])
    assert len(results) == len(sentences)
    assert all(r.translation == "模拟译文" for r in results)
    assert translator.tokens_used == 7
    # 块内多句一次请求：token 消耗与句数无关（AC-6 成本约束的结构保障）

