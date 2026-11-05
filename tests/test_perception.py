from types import SimpleNamespace

import pytest
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    EagerTranscriptionFrame,
    PerceptionMetadataFrame,
)
from behavioralsignals.models import ResultItem, ModelPredictions, StreamingResultResponse
from pipecat.expressive.perception import PerceptionSignal

from pipecat_behavioralsignals import stream, perception
from pipecat_behavioralsignals.signals import CAPABILITIES
from pipecat_behavioralsignals.perception import BehavioralSignalsPerceptionService


SAMPLE_RATE = 16000
ONE_SECOND = b"\0" * SAMPLE_RATE * 2  # 16-bit mono
WAIT = SleepFrame(sleep=0.2)  # time for the stream thread to read the audio

ANGRY = ResultItem(
    task="emotion",
    startTime="0.2",
    endTime="0.8",
    level="segment",
    prediction=[
        ModelPredictions(label="neutral", posterior="0.2"),
        ModelPredictions(label="angry", posterior="0.7"),
        ModelPredictions(label="sad", posterior="0.1"),
    ],
)
INTENSITY = ResultItem(
    task="intensity",
    startTime="0.2",
    endTime="0.8",
    level="segment",
    prediction=[ModelPredictions(score="0.6123")],
)
LATER_STRENGTH = ResultItem(
    task="strength",
    startTime="1.2",
    endTime="1.8",
    level="segment",
    prediction=[ModelPredictions(label="strong", posterior="0.9")],
)
AGE = ResultItem(
    task="age",
    startTime="0.2",
    endTime="0.8",
    level="segment",
    prediction=[ModelPredictions(label="23 - 30", posterior="0.9")],
)


def fake_client(results: list[ResultItem]):
    """A client whose stream reads the first audio, answers with `results`, then reads the rest."""

    def stream_audio(audio, options):
        next(audio)
        yield StreamingResultResponse(result=results)
        for _ in audio:
            pass

    return SimpleNamespace(
        behavioral=SimpleNamespace(stream_audio=stream_audio), close=lambda: None
    )


def failing_client(stream_audio):
    """A client whose every stream reads the first audio, then runs `stream_audio`."""

    def streams(audio, options):
        next(audio)
        yield from stream_audio()

    return SimpleNamespace(behavioral=SimpleNamespace(stream_audio=streams), close=lambda: None)


def use_client(monkeypatch, client) -> None:
    monkeypatch.setattr(perception, "Client", lambda cid, api_key: client)


def audio(channels: int = 1) -> InputAudioRawFrame:
    return InputAudioRawFrame(
        audio=ONE_SECOND * channels, sample_rate=SAMPLE_RATE, num_channels=channels
    )


def transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="user", timestamp="")


def of_type(frames, frame_type) -> list:
    return [frame for frame in frames if isinstance(frame, frame_type)]


async def test_a_reading_goes_to_the_transcription_it_overlaps(monkeypatch):
    use_client(monkeypatch, fake_client([ANGRY, INTENSITY, LATER_STRENGTH, AGE]))
    # Audio is a system frame, so it overtakes a queued transcription: wait after each one.
    frames = [audio(), WAIT, transcription("first"), WAIT, audio(), WAIT, transcription("second")]

    down, _ = await run_test(BehavioralSignalsPerceptionService(), frames_to_send=frames)

    first, second = of_type(down, TranscriptionFrame)
    assert first.perception == [  # its audio is 0-1 s
        PerceptionSignal("emotion", labels=[("angry", 0.7), ("neutral", 0.2), ("sad", 0.1)]),
        PerceptionSignal("intensity", value=0.61),
    ]
    assert second.perception == [  # its audio is 1-2 s
        PerceptionSignal("strength", labels=[("strong", 0.9)]),
    ]


async def test_a_transcription_with_no_new_audio_reads_the_previous_audio(monkeypatch):
    use_client(monkeypatch, fake_client([ANGRY]))
    frames = [audio(), WAIT, transcription("first"), WAIT, transcription("second")]

    down, _ = await run_test(BehavioralSignalsPerceptionService(), frames_to_send=frames)

    first, second = of_type(down, TranscriptionFrame)
    assert first.perception
    assert second.perception == first.perception


async def test_an_empty_transcription_is_not_read_and_does_not_end_the_span(monkeypatch):
    use_client(monkeypatch, fake_client([ANGRY]))
    frames = [audio(), WAIT, transcription(" "), WAIT, audio(), WAIT, transcription("first")]

    down, _ = await run_test(BehavioralSignalsPerceptionService(), frames_to_send=frames)

    empty, first = of_type(down, TranscriptionFrame)
    assert empty.perception is None
    assert first.perception  # its audio is 0-2 s


async def test_results_with_missing_values_are_skipped(monkeypatch):
    no_posterior = ResultItem(
        task="strength",
        startTime="0.2",
        endTime="0.8",
        prediction=[ModelPredictions(label="strong", posterior=None)],
    )
    no_label = ResultItem(
        task="positivity",
        startTime="0.2",
        endTime="0.8",
        prediction=[ModelPredictions(label=None, posterior="0.9")],
    )
    no_score = ResultItem(
        task="intensity", startTime="0.2", endTime="0.8", prediction=[ModelPredictions()]
    )
    use_client(monkeypatch, fake_client([no_posterior, no_label, no_score, ANGRY]))
    frames = [audio(), WAIT, transcription("first")]

    down, _ = await run_test(BehavioralSignalsPerceptionService(), frames_to_send=frames)

    assert [signal.name for signal in of_type(down, TranscriptionFrame)[0].perception] == [
        "emotion"
    ]


def raises():
    raise RuntimeError("UNAUTHENTICATED")
    yield


def answers_without_results():
    yield StreamingResultResponse(result=[])


@pytest.mark.parametrize("stream_audio", [raises, answers_without_results])
async def test_a_stream_that_keeps_failing_is_reported_once(monkeypatch, stream_audio):
    monkeypatch.setattr(stream, "FIRST_RETRY_S", 0.01)
    monkeypatch.setattr(stream, "MAX_RETRY_S", 0.01)
    use_client(monkeypatch, failing_client(stream_audio))
    frames = [audio(), SleepFrame(sleep=0.05)] * 10

    _, up = await run_test(BehavioralSignalsPerceptionService(), frames_to_send=frames)

    errors = of_type(up, ErrorFrame)
    assert len(errors) == 1
    assert "still retrying" in errors[0].error


async def test_audio_that_is_not_mono_is_reported_and_not_read(monkeypatch):
    use_client(monkeypatch, fake_client([ANGRY]))
    frames = [audio(channels=2), WAIT, transcription("first")]

    down, up = await run_test(BehavioralSignalsPerceptionService(), frames_to_send=frames)

    assert of_type(down, TranscriptionFrame)[0].perception is None
    assert "mono" in of_type(up, ErrorFrame)[0].error


async def test_an_eager_transcription_does_not_end_the_span(monkeypatch):
    use_client(monkeypatch, fake_client([ANGRY]))
    eager = EagerTranscriptionFrame(text="first", user_id="user", timestamp="")
    frames = [audio(), WAIT, eager, audio(), transcription("first second")]

    down, _ = await run_test(BehavioralSignalsPerceptionService(), frames_to_send=frames)

    assert of_type(down, EagerTranscriptionFrame)[0].perception
    assert of_type(down, TranscriptionFrame)[0].perception


async def test_the_signals_are_declared_once_at_start(monkeypatch):
    use_client(monkeypatch, fake_client([ANGRY]))
    frames = [audio(), WAIT, transcription("first")]

    down, _ = await run_test(BehavioralSignalsPerceptionService(), frames_to_send=frames)

    assert isinstance(down[0], PerceptionMetadataFrame)
    assert len(of_type(down, PerceptionMetadataFrame)) == 1
    assert down[0].perception == CAPABILITIES


async def test_a_failed_login_is_reported_and_frames_pass_unread(monkeypatch):
    def failing_client(cid, api_key):
        raise ValueError("Missing cid")

    monkeypatch.setattr(perception, "Client", failing_client)
    frames = [audio(), transcription("first")]

    down, up = await run_test(BehavioralSignalsPerceptionService(), frames_to_send=frames)

    assert not of_type(down, PerceptionMetadataFrame)
    assert of_type(down, TranscriptionFrame)[0].perception is None
    assert "Missing cid" in of_type(up, ErrorFrame)[0].error
