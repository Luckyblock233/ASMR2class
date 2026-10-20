"""运行时指标快照层测试（采集线程写、UI 读快照；系统采样可降级为 N/A）。"""

import threading

from src.telemetry import MetricsSnapshot, RuntimeMetrics, SystemSampler, SystemSample


def test_metrics_counters_and_averages():
    m = RuntimeMetrics()
    snap = m.snapshot()
    assert snap.asr_count == 0 and snap.asr_avg_ms == 0.0
    assert snap.translate_count == 0 and snap.translate_failed == 0

    m.record_audio_block(0.25)
    m.set_vad_active(True)
    m.record_asr(100.0)
    m.record_asr(300.0)
    m.record_translation(800.0, True)
    m.record_translation(1200.0, False)

    snap = m.snapshot()
    assert snap.rms == 0.25
    assert snap.vad_active is True
    assert snap.asr_count == 2
    assert snap.asr_last_ms == 300.0
    assert snap.asr_avg_ms == 200.0
    assert snap.translate_count == 2
    assert snap.translate_failed == 1
    assert snap.translate_last_ms == 1200.0
    assert snap.translate_avg_ms == 1000.0
    assert snap.system == SystemSample()  # 未附系统采样时全 N/A


def test_metrics_thread_safety():
    m = RuntimeMetrics()

    def writer():
        for _ in range(500):
            m.record_asr(1.0)
            m.record_translation(2.0, True)
            m.record_audio_block(0.1)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = m.snapshot()
    assert snap.asr_count == 2000
    assert snap.translate_count == 2000


def test_snapshot_carries_system_sample():
    m = RuntimeMetrics()
    sample = SystemSample(cpu_percent=12.5, vram_free_mb=4096.0)
    snap = m.snapshot(system=sample)
    assert snap.system.cpu_percent == 12.5
    assert snap.system.vram_free_mb == 4096.0
    assert snap.system.gpu_percent is None


def test_system_sampler_never_raises():
    """本机有/无 GPU、有/无 NVIDIA 驱动都必须返回合法快照。"""
    sampler = SystemSampler()
    try:
        sample = sampler.sample()
        assert isinstance(sample, SystemSample)
        if sampler.gpu_available:
            assert sample.vram_free_mb is not None and sample.vram_free_mb >= 0
            assert sample.gpu_percent is not None
        else:
            assert sample.gpu_percent is None
            assert sample.vram_free_mb is None
        if sample.cpu_percent is not None:
            assert 0.0 <= sample.cpu_percent <= 100.0
    finally:
        sampler.close()
    # 关闭后 GPU 采样安全降级为 N/A（CPU/RAM 若 psutil 可用仍正常）
    after = sampler.sample()
    assert after.gpu_percent is None and after.vram_free_mb is None


def test_snapshot_is_immutable():
    import dataclasses

    import pytest

    snap = MetricsSnapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.asr_count = 9
