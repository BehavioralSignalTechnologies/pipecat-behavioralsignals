#
# Copyright (c) 2026, Behavioral Signal Technologies, Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""One Behavioral Signals stream per call, in a thread."""

import queue
import threading
from collections.abc import Callable, Iterator

from loguru import logger
from behavioralsignals import Behavioral
from behavioralsignals.models import StreamingOptions

from pipecat_behavioralsignals.signals import Reading, reading_of


FIRST_RETRY_S = 0.5
MAX_RETRY_S = 8.0
# After this many streams in a row with no results, it reports an error. It keeps retrying, since
# the cause may pass, like a network outage or a credit top-up.
FAILURES_TO_REPORT = 3
# The most audio chunks waiting to be sent: about 20 s of 20 ms chunks. When the stream stops
# reading audio and this fills, the queued audio is dropped and the stream restarts.
MAX_CHUNKS = 1000

# One chunk of audio: where it starts in seconds from the first audio, its sample rate, its bytes.
Chunk = tuple[float, int, bytes]


def _audio(first: Chunk, chunks: queue.Queue) -> Iterator[bytes]:
    """The audio of one stream, until None."""
    chunk = first
    while chunk is not None:
        yield chunk[2]
        chunk = chunks.get()


def _end(chunks: queue.Queue) -> None:
    """End the audio of a stream: drop the queued audio, then add None."""
    try:
        while True:
            chunks.get_nowait()
    except queue.Empty:
        chunks.put_nowait(None)


class BehavioralSignalsStream:
    """Audio in with `send`, readings out with `drain`.

    The stream opens on the first audio. When it fails, a new one opens after a pause, from the
    audio sent after it failed. When streams keep failing, it calls `on_failing` once.
    """

    def __init__(self, client: Behavioral, on_failing: Callable[[str], None]):
        """Start the thread that runs the stream.

        Args:
            client: A Behavioral Signals client that has logged in.
            on_failing: Called, from the stream's thread, when streams keep failing.
        """
        self._client = client
        self._on_failing = on_failing
        self._chunks: queue.Queue = queue.Queue(MAX_CHUNKS)
        self._readings: queue.Queue = queue.Queue()
        self._closed = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, audio: bytes, sample_rate: int, at: float) -> None:
        """Send 16-bit mono audio.

        Args:
            audio: The audio bytes.
            sample_rate: The audio's sample rate.
            at: Where the audio starts, in seconds from the first audio.
        """
        if self._closed.is_set():
            return
        try:
            self._chunks.put_nowait((at, sample_rate, audio))
        except queue.Full:
            logger.warning(
                "Behavioral Signals stream is not reading audio: dropping it and restarting"
            )
            _end(self._chunks)

    def drain(self) -> list[Reading]:
        """Take the readings that arrived since the last call.

        Returns:
            The readings, oldest first.
        """
        readings = []
        while not self._readings.empty():
            readings.append(self._readings.get_nowait())
        return readings

    def close(self) -> None:
        """Stop streaming: end the audio, and open no new stream."""
        self._closed.set()
        _end(self._chunks)

    def _run(self) -> None:
        retry, failures = FIRST_RETRY_S, 0
        while not self._closed.is_set():
            answered = self._stream_or_fail()
            if self._closed.is_set():
                return
            retry = FIRST_RETRY_S if answered else min(retry * 2, MAX_RETRY_S)
            failures = 0 if answered else failures + 1
            if failures == FAILURES_TO_REPORT:
                self._on_failing(
                    f"Behavioral Signals stream failed {failures} times in a row, still retrying"
                )
            logger.warning(f"Behavioral Signals stream ended, opening a new one in {retry:.1f}s")
            self._closed.wait(retry)

    def _stream_or_fail(self) -> bool:
        """Run one stream; a failure to open it counts as a stream with no results."""
        try:
            return self._stream()
        except Exception as error:  # noqa: BLE001 - reconnects on any failure
            logger.warning(f"Behavioral Signals stream failed: {error!r}")
            return False

    def _stream(self) -> bool:
        """Run one stream from a new queue, so audio sent while it was down is dropped.

        Returns:
            True if the stream sent any results.
        """
        chunks = self._chunks = queue.Queue(MAX_CHUNKS)
        if self._closed.is_set() or (first := chunks.get()) is None:
            return False
        options = StreamingOptions(sample_rate=first[1], encoding="LINEAR_PCM")
        responses = self._client.stream_audio(_audio(first, chunks), options)
        answered = False
        while (results := _next_results(responses)) is not None:
            answered = answered or bool(results)
            self._keep(results, first[0])
        _end(chunks)  # ends the audio if the stream is still reading it
        return answered

    def _keep(self, items: list, offset: float) -> None:
        for item in items:
            if reading := reading_of(item, offset):
                self._readings.put(reading)


def _next_results(responses: Iterator) -> list | None:
    """The results of the next response, or None when the stream ends or fails."""
    try:
        response = next(responses)
    except StopIteration:
        return None
    except Exception as error:  # noqa: BLE001 - reconnects on any failure
        logger.warning(f"Behavioral Signals stream failed: {error!r}")
        return None
    return response.results or []
