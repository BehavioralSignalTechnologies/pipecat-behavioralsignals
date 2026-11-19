import time
import threading
from types import SimpleNamespace

from behavioralsignals import END_OF_TURN

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

    client = SimpleNamespace(stream_audio=stream_audio)
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


def test_a_stream_that_fails_to_open_is_retried_and_reported(monkeypatch):
    monkeypatch.setattr(stream, "FIRST_RETRY_S", 0.01)
    monkeypatch.setattr(stream, "MAX_RETRY_S", 0.01)
    reports = []

    def stream_audio(audio, options):
        raise ConnectionError("network down")  # before any audio is read

    s = BehavioralSignalsStream(SimpleNamespace(stream_audio=stream_audio), reports.append)
    deadline = time.monotonic() + 2
    while not reports and time.monotonic() < deadline:
        s.send(CHUNK, 16000, 0.0)  # each failed stream needs new audio to retry
        time.sleep(0.01)
    s.close()

    assert len(reports) == 1


def recording_client(items: list, options: list, ended: threading.Event) -> SimpleNamespace:
    """A client whose stream records its options and every item of its audio, then ends."""

    def stream_audio(audio, stream_options):
        options.append(stream_options)
        items.extend(audio)
        ended.set()
        yield from ()

    return SimpleNamespace(stream_audio=stream_audio)


def started(client) -> BehavioralSignalsStream:
    """A stream whose thread has made the queue it reads, so no audio sent to it is lost."""
    s = BehavioralSignalsStream(client, on_failing=lambda message: None)
    created = s._chunks
    wait_until(lambda: s._chunks is not created)
    return s


def test_end_turn_queues_the_marker_between_the_audio():
    items, options, ended = [], [], threading.Event()
    s = started(recording_client(items, options, ended))

    s.send(CHUNK, 16000, 0.0)
    s.end_turn()
    s.send(CHUNK, 16000, 0.01)
    wait_until(lambda: len(items) == 3)
    s.close()
    ended.wait(2)

    assert items == [CHUNK, END_OF_TURN, CHUNK]


def test_end_turn_before_any_audio_is_dropped():
    items, options, ended = [], [], threading.Event()
    s = started(recording_client(items, options, ended))

    s.end_turn()
    s.send(CHUNK, 16000, 0.0)
    wait_until(lambda: items)
    s.close()
    ended.wait(2)

    assert items == [CHUNK]
    assert options[0].sample_rate == 16000


def test_end_turn_on_a_closed_stream_queues_nothing():
    items, options, ended = [], [], threading.Event()
    s = started(recording_client(items, options, ended))
    s.close()
    wait_until(lambda: s._chunks.qsize() == 0)  # the thread has read the end

    s.end_turn()

    assert s._chunks.qsize() == 0


def test_end_turn_on_a_full_queue_raises_nothing(monkeypatch):
    monkeypatch.setattr(stream, "MAX_CHUNKS", 10)
    unstall = threading.Event()
    calls = []

    def stream_audio(audio, options):
        calls.append(1)
        next(audio)
        unstall.wait()  # the server stops reading
        for _ in audio:
            pass
        yield from ()

    s = started(SimpleNamespace(stream_audio=stream_audio))
    s.send(CHUNK, 16000, 0.0)
    wait_until(lambda: calls)
    for _ in range(10):
        s.send(CHUNK, 16000, 0.0)

    s.end_turn()
    queued = s._chunks.qsize()
    unstall.set()
    s.close()

    assert queued == 10
