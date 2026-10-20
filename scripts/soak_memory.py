"""内存增长 soak 测试：模拟长会话高负载，定位 RAM 持续上升的来源。

用法：QT_QPA_PLATFORM=offscreen uv run --no-sync python scripts/soak_memory.py
"""

import gc
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def report(name: str, before: float) -> float:
    gc.collect()
    after = rss_mb()
    print(f"{name:44s} {after - before:+8.1f} MB   (当前 {after:.0f} MB)")
    return after


def main() -> None:
    from PyQt6.QtWidgets import QApplication

    from PyQt6.QtCore import QEventLoop, QTimer

    app = QApplication([])

    def pump():
        # 真实事件循环迭代：deleteLater 的延迟删除只在 exec 中处理
        loop = QEventLoop()
        QTimer.singleShot(0, loop.quit)
        loop.exec()

    base = rss_mb()
    print(f"基线 {base:.0f} MB")

    # ---- 1. SessionDocument：长会话句段全量保留（设计使然？）----
    from src.subtitle.document import SessionDocument

    doc = SessionDocument()
    N = 10000
    for i in range(N):
        sid = doc.add(i * 2.0, i * 2.0 + 1.0, f"これはテスト文です{i}", f"这是测试句{i}")
        doc.apply_analysis(sid, {
            "structure": {"pattern": "テスト", "parts": ["a", "b"]},
            "grammar_points": [{"name": "〜です", "explanation": "x" * 50}],
            "vocabulary": [{"word": "テスト", "meaning": "测试"}],
            "extra": {},
        })
    report(f"SessionDocument {N} 句+分析", base)

    # ---- 2. SubtitleView 容量淘汰：10000 次 upsert（新句+原位更新）----
    from src.subtitle.document import Segment, TRANSLATION_READY
    from src.ui.subtitle_view import SubtitleView

    view = SubtitleView(capacity=200)
    view.resize(720, 480)
    view.show()
    before = rss_mb()
    for i in range(N):
        view.upsert_segment(Segment(
            id=i + 1, start=i, end=i + 1, source=f"句{i}", translation=f"译{i}",
            translation_status=TRANSLATION_READY, asr_ms=100, translate_ms=500,
        ))
        if i % 200 == 0:
            pump()
    pump()
    assert len(view._blocks) <= 200
    report(f"SubtitleView {N} 次 upsert（容量 200）", before)

    # ---- 3. 监控行每秒重建 ×3600（模拟 1 小时）----
    from src.ui.overlay import create_overlay
    from src.telemetry import RuntimeMetrics, SystemSample

    metrics = RuntimeMetrics()
    overlay = create_overlay(metrics=metrics, sampler=None)
    overlay.show()
    overlay.metrics = metrics

    class _FixedSampler:
        def sample(self):
            return SystemSample(
                cpu_percent=10, ram_percent=50, gpu_percent=20, vram_free_mb=4000
            )

    overlay.sampler = _FixedSampler()
    before = rss_mb()
    for i in range(3600):
        metrics.record_audio_block(0.1)
        overlay._refresh_metrics()
        if i % 10 == 0:  # 及时处理 deleteLater，贴近真实事件循环
            pump()
    pump()
    report("监控行重建 ×3600（1 小时）", before)

    # ---- 4. 语法卡反复弹出 ×500 ----
    from src.profiles.loader import LanguageProfile
    from src.ui.grammar_card import GrammarCardPopup

    profile = LanguageProfile(
        name="t", language="t", display_name="t", source_lang="x", target_lang="zh",
        raw={"grammar_card": {"sections": [
            {"key": "structure", "title": "结构", "renderer": "tree"},
            {"key": "vocabulary", "title": "词汇", "renderer": "table",
             "columns": [{"key": "word", "label": "词"}]},
        ]}},
    )
    card = GrammarCardPopup()
    seg = doc.get(1)
    before = rss_mb()
    for _ in range(500):
        card.show_segment(seg, profile)
    pump()
    report("语法卡弹出 ×500", before)

    # ---- 5. WhisperAsmrVad 缓存（假 session，长跑不切句 → 检查缓存上界）----
    import numpy as np

    from src.asr.vad import WhisperAsmrVad

    class FakeSession:
        class _In:
            name = "input"

        def get_inputs(self):
            return [self._In()]

        def run(self, _out, feed):
            return [np.zeros((1, 3000), dtype=np.float32)]

    vad = WhisperAsmrVad("fake", session=FakeSession())
    before = rss_mb()
    # 模拟 pending 不断增长（静音段不切句，最多到 max_segment_sec 被 streamer 强切）
    audio = np.zeros(480000 * 3, dtype=np.float32)
    for i in range(100):
        vad(audio[: 16000 * (i % 15 + 1)])  # 长度周期性回退 → 缓存重置
    probs_bytes = sum(p.nbytes for p in vad._final_probs)
    report("whisper_asmr VAD 缓存 ×100 次调用", before)
    print(f"    _final_probs 驻留：{len(vad._final_probs)} 块 / {probs_bytes / 1024:.0f} KB")

    print("\n判读：单步增长 >50MB 需要关注；<10MB 属正常波动。")


if __name__ == "__main__":
    main()
