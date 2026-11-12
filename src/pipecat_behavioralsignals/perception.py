#
# Copyright (c) 2026, Behavioral Signal Technologies, Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The service that reads how the user spoke with Behavioral Signals."""

import asyncio

from behavioralsignals import Behavioral
from pipecat.frames.frames import (
    Frame,
    EndFrame,
    CancelFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    EagerTranscriptionFrame,
    PerceptionMetadataFrame,
)
from pipecat.services.settings import ServiceSettings
from pipecat.services.ai_service import AIService
from pipecat.expressive.perception import (
    PerceptionSignal,
    PerceptionCapabilities,
    merge_perception,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup

from pipecat_behavioralsignals.stream import BehavioralSignalsStream
from pipecat_behavioralsignals.signals import CAPABILITIES, Reading


class BehavioralSignalsPerceptionService(AIService):
    """Reads how the user spoke from their voice with Behavioral Signals.

    Goes between the STT service and the user aggregator. It streams the user's
    audio, which the STT service passes on, to the Behavioral Signals streaming
    API. On each final transcription it adds to ``perception`` the latest
    reading of each signal that overlaps the transcription's audio: the audio
    since the previous final transcription, or that of the previous one when
    no audio came in between. It also adds it on an eager transcription,
    without ending the span. Signals with no such reading are left out, and
    with none at all ``perception`` stays as it was. Transcriptions with no
    text pass through unread. A reading a producer before it set, such as
    ``PipecatToneReader``, is kept, merged with ``merge_perception``.

    It reads mono audio only. On audio with more channels it reports an error
    and stops reading. When the stream keeps failing, it reports an error
    once and keeps retrying.

    At start it broadcasts a ``PerceptionMetadataFrame`` that declares its
    signals. A ``PerceptionMetadataFrame`` from a producer before it is
    declared again with its signals, since the LLM keeps the last one only.
    It logs in to Behavioral Signals while the pipeline is set up; if
    that fails, it reports an error that leaves it unusable, and frames pass
    through unread.

    Part of expressive conversation (alpha): turn it on with
    ``PipelineParams(enable_expressive_conversation=True)`` for the LLM to read
    the turns.

    Example::

        pipeline = Pipeline([transport.input(), stt, BehavioralSignalsPerceptionService(),
                             user_aggregator, llm, ...])
    """

    def __init__(self, *, cid: str | None = None, api_key: str | None = None, **kwargs):
        """Initialize the service.

        Args:
            cid: Your Behavioral Signals client ID. Defaults to the
                ``BEHAVIORALSIGNALS_CID`` environment variable.
            api_key: Your Behavioral Signals API key. Defaults to the
                ``BEHAVIORALSIGNALS_API_KEY`` environment variable.
            **kwargs: Additional arguments passed to the parent.
        """
        super().__init__(settings=ServiceSettings(model=None), **kwargs)
        self._cid = cid
        self._api_key = api_key
        self._client: Behavioral | None = None
        self._stream: BehavioralSignalsStream | None = None
        self._seconds = 0.0  # the user's audio so far
        self._span_start = 0.0  # where the next transcription's audio starts
        self._span = (0.0, 0.0)  # the audio of the last final transcription
        self._latest: dict[str, Reading] = {}  # the latest reading of each signal
        self._declared_before: PerceptionCapabilities | None = None  # by producers before it

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and log in to Behavioral Signals.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        try:
            self._client = await asyncio.to_thread(Behavioral, self._cid, self._api_key)
        except Exception as e:  # noqa: BLE001 - any login failure makes it unusable
            await self.push_error(
                f"{self}: couldn't log in to Behavioral Signals: {e}",
                e,
                force_treat_as_permanent=True,
            )
            return
        self._stream = BehavioralSignalsStream(self._client, self._report)

    def service_metadata_frame(self) -> PerceptionMetadataFrame | None:
        """The signals this service reports, or None when it couldn't log in.

        Returns:
            The metadata frame to broadcast at start.
        """
        if not self._stream:
            return None
        return PerceptionMetadataFrame(service_name=self.name, perception=self._capabilities())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Stream the user's audio, and set the perception of transcriptions.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._send(frame)
        elif isinstance(frame, EagerTranscriptionFrame):
            self._add_reading(frame, self._perceive(self._span_start, self._seconds))
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._end_span()
            self._add_reading(frame, self._perceive(*self._span))
        elif (
            isinstance(frame, PerceptionMetadataFrame)
            and direction == FrameDirection.DOWNSTREAM
            and self._stream
        ):
            # Declared again with its signals: the LLM keeps the last declaration only.
            self._declared_before = frame.perception
            frame = self.service_metadata_frame()
        await self.push_frame(frame, direction)

    async def stop(self, frame: EndFrame):
        """Close the stream.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        self._close()

    async def cancel(self, frame: CancelFrame):
        """Close the stream.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        self._close()

    async def cleanup(self):
        """Close the stream and the client."""
        await super().cleanup()
        self._close()
        if self._client:
            self._client.close()
            self._client = None

    async def _send(self, frame: InputAudioRawFrame) -> None:
        if self._stream and frame.num_channels != 1:
            self._close()  # its readings would be mistimed
            await self.push_error(
                f"{self}: Behavioral Signals reads mono audio only, "
                f"but got {frame.num_channels} channels; stopped reading"
            )
        if self._stream:
            self._stream.send(frame.audio, frame.sample_rate, self._seconds)
        self._seconds += frame.num_frames / frame.sample_rate

    def _end_span(self) -> None:
        """Make the audio since the last final transcription its span, if there is any."""
        if self._seconds > self._span_start:
            self._span = (self._span_start, self._seconds)
            self._span_start = self._seconds

    def _capabilities(self) -> PerceptionCapabilities:
        """The signals declared before it, then its own."""
        if self._declared_before:
            return self._declared_before.merge(CAPABILITIES)
        return CAPABILITIES

    def _add_reading(
        self,
        frame: TranscriptionFrame | EagerTranscriptionFrame,
        reading: list[PerceptionSignal] | None,
    ) -> None:
        """Add the reading to the one the transcription carries, if any."""
        if reading:
            frame.perception = merge_perception(frame.perception, reading)

    def _perceive(self, start: float, end: float) -> list[PerceptionSignal] | None:
        if self._stream:
            self._latest.update({r.signal.name: r for r in self._stream.drain()})
        signals = [r.signal for r in self._latest.values() if r.overlaps(start, end)]
        return signals or None

    def _report(self, message: str) -> None:
        """Report a stream error upstream. Called from the stream's thread."""
        coroutine = self.push_error(f"{self}: {message}")
        asyncio.run_coroutine_threadsafe(coroutine, self.get_event_loop())

    def _close(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None
