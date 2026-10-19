import numpy as np

from src.audio.base import RingBuffer


def test_write_read_basic():
    rb = RingBuffer(16000)
    data = np.arange(1000, dtype=np.float32)
    rb.write(data)
    out, start = rb.read(500)
    assert start == 0
    np.testing.assert_array_equal(out, data[:500])
    out, start = rb.read(500)
    assert start == 500
    np.testing.assert_array_equal(out, data[500:])


def test_wraparound():
    rb = RingBuffer(100)
    rb.write(np.arange(80, dtype=np.float32))
    rb.read(80)
    rb.write(np.arange(80, 130, dtype=np.float32))  # 写入 50 个，环绕
    out, start = rb.read(50)
    assert start == 80
    np.testing.assert_array_equal(out, np.arange(80, 130, dtype=np.float32))


def test_overflow_drops_oldest():
    rb = RingBuffer(100)
    rb.write(np.arange(250, dtype=np.float32))
    out, start = rb.read(100)
    assert start == 150  # 最旧的 150 个被覆盖
    np.testing.assert_array_equal(out, np.arange(150, 250, dtype=np.float32))


def test_read_timeout_returns_partial():
    rb = RingBuffer(16000)
    rb.write(np.ones(100, dtype=np.float32))
    out, _ = rb.read(500, timeout=0.05)
    assert len(out) == 100


def test_read_timeout_empty():
    rb = RingBuffer(16000)
    out, _ = rb.read(500, timeout=0.05)
    assert len(out) == 0


def test_block_until_data():
    import threading, time

    rb = RingBuffer(16000)
    result = {}

    def reader():
        out, start = rb.read(200)
        result["out"] = out

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    rb.write(np.ones(200, dtype=np.float32))
    t.join(timeout=2)
    assert len(result["out"]) == 200
