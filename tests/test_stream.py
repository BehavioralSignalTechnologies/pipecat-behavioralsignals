import time
import threading
from types import SimpleNamespace

from pipecat_behavioralsignals import stream
from pipecat_behavioralsignals.stream import BehavioralSignalsStream


CHUNK = b"\0" * 320


def wait_until(condition) -> None:
    deadline = time.monotonic() + 2
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_a_stalled_stream_drops_its_audio_and_restarts(monkeypatch):
    monkeypatch.setattr(stream, "MAX_CHUNKS", 10)
    monkeypatch.setattr(stream, "FIRST_RETRY_S", 0.01)
    unstall = threading.Event()
    calls = []

    def stream_audio(audio, options):
        calls.append(1)
        next(audio)
        unstall.wait()  # the server stops reading
        for _ in audio:
            pass
        yield from ()

    client = SimpleNamespace(behavioral=SimpleNamespace(stream_audio=stream_audio))
    s = BehavioralSignalsStream(client, on_failing=lambda message: None)
    s.send(CHUNK, 16000, 0.0)
    wait_until(lambda: calls)
    for _ in range(25):
        s.send(CHUNK, 16000, 0.0)
    queued = s._chunks.qsize()
    unstall.set()
    wait_until(lambda: s._chunks.qsize() == 0)  # the new stream's queue
    s.send(CHUNK, 16000, 0.0)
    wait_until(lambda: len(calls) == 2)
    s.close()

    assert queued <= 10
    assert len(calls) == 2
