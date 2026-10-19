"""语义边界分块器测试（FR-C2）。"""

import time

from src.asr.streamer import Sentence
from src.config import BlockConfig
from src.nlp.blocker import Blocker


def s(start, end, text="x"):
    return Sentence(start=start, end=end, text=text)


def collect(cfg: BlockConfig, autoflush=False, clock=None):
    blocks: list[list[Sentence]] = []
    b = Blocker(cfg, blocks.append, clock=clock, autoflush=autoflush)
    return b, blocks


def test_split_by_silence_gap():
    b, blocks = collect(BlockConfig(max_sentences=8, silence_gap_sec=1.5))
    b.feed(s(0.0, 1.0))
    b.feed(s(1.2, 2.0))   # 间隙 0.2s → 同块
    b.feed(s(4.0, 5.0))   # 间隙 2.0s > 1.5s → 成块
    assert len(blocks) == 1
    assert [len(x) for x in blocks] == [2]
    b.flush()
    assert [len(x) for x in blocks] == [2, 1]


def test_split_by_max_sentences():
    b, blocks = collect(BlockConfig(max_sentences=3, silence_gap_sec=10))
    for i in range(3):
        b.feed(s(i * 0.5, i * 0.5 + 0.4))
    assert [len(x) for x in blocks] == [3]
    b.feed(s(2.0, 2.4))
    assert len(blocks) == 1  # 第四句进入新块，未 flush
    b.flush()
    assert [len(x) for x in blocks] == [3, 1]


def test_idle_autoflush():
    clock_val = [0.0]
    b, blocks = collect(
        BlockConfig(max_sentences=8, silence_gap_sec=1.5),
        autoflush=True,
        clock=lambda: clock_val[0],
    )
    b.feed(s(0.0, 1.0))
    assert blocks == []  # 未满块、无间隙，暂不翻译
    clock_val[0] = 3.0   # 推进墙上时钟超过 idle（1.5 + 0.5）
    deadline = time.monotonic() + 2
    while not blocks and time.monotonic() < deadline:
        time.sleep(0.05)
    assert [len(x) for x in blocks] == [1]
    b.flush()


def test_flush_idempotent_and_close():
    b, blocks = collect(BlockConfig())
    b.feed(s(0.0, 1.0))
    b.flush()
    b.flush()
    assert [len(x) for x in blocks] == [1]
    b.feed(s(2.0, 3.0))  # close 后忽略
    assert len(blocks) == 1


def test_overlap_sentences_not_duplicated():
    """FR-C2.3：间隙句只属于一个块，不重复出现。"""
    b, blocks = collect(BlockConfig(max_sentences=8, silence_gap_sec=1.0))
    b.feed(s(0.0, 1.0, "a"))
    b.feed(s(5.0, 6.0, "b"))  # 触发成块
    b.flush()
    all_texts = [s.text for blk in blocks for s in blk]
    assert all_texts == ["a", "b"]  # 无重复
