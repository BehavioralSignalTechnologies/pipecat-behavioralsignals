#
# Copyright (c) 2026, Behavioral Signal Technologies, Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The signals sent to the LLM, and how a Behavioral Signals result becomes one."""

from dataclasses import dataclass

from behavioralsignals.models import ResultItem
from pipecat.expressive.perception import (
    PerceptionTerm,
    PerceptionSignal,
    PerceptionCapabilities,
)


_FROM_VOICE = "Read from the voice, not the words, over the latest part of the turn."

# Each signal has the name of its Behavioral Signals task, except emotion, which is
# voice_emotion: Pipecat Tone's emotion is read from the words, and merging would mix them.
_SIGNAL_NAMES = {"emotion": "voice_emotion"}

CAPABILITIES = PerceptionCapabilities(
    signals=[
        PerceptionTerm(
            "voice_emotion",
            f"The emotion in the voice. {_FROM_VOICE}",
            ["happy", "angry", "sad", "neutral"],
        ),
        PerceptionTerm(
            "strength",
            f"Arousal: the energy in the voice. {_FROM_VOICE}",
            ["strong", "weak", "neutral"],
        ),
        PerceptionTerm(
            "positivity",
            f"Valence: how positive the tone of voice is. {_FROM_VOICE}",
            ["positive", "negative", "neutral"],
        ),
        PerceptionTerm(
            "speaking_rate",
            f"How fast the user speaks, compared to speakers in general. {_FROM_VOICE}",
            ["fast", "slow", "normal"],
        ),
        PerceptionTerm(
            "hesitation",
            f"Whether the user sounds hesitant. {_FROM_VOICE}",
            ["yes", "no"],
        ),
        PerceptionTerm(
            "engagement",
            f"Whether the user sounds involved or detached. {_FROM_VOICE}",
            ["engaged", "withdrawn", "neutral"],
        ),
        PerceptionTerm(
            "intensity",
            f"How intense the emotion is, from 0 to 1, whichever emotion it is. {_FROM_VOICE}",
        ),
    ]
)

_TASK_NAMES = {signal: task for task, signal in _SIGNAL_NAMES.items()}
_TASKS = {_TASK_NAMES.get(term.name, term.name) for term in CAPABILITIES.signals}


@dataclass(frozen=True)
class Reading:
    """One signal read over a span of the user's audio, in seconds from the first audio.

    Parameters:
        start: Where the span starts.
        end: Where the span ends.
        signal: The signal read over the span.
    """

    start: float
    end: float
    signal: PerceptionSignal

    def overlaps(self, start: float, end: float) -> bool:
        """Whether this reading's span overlaps the span from `start` to `end`.

        Args:
            start: Where the other span starts.
            end: Where the other span ends.

        Returns:
            True if the spans share some audio.
        """
        return self.start < end and self.end > start


def reading_of(item: ResultItem, offset: float) -> Reading | None:
    """The reading of a result, moved by `offset` seconds.

    Args:
        item: A result of the stream.
        offset: Where the stream's audio starts, in seconds from the first audio.

    Returns:
        The reading, or None for a task that is not sent, or a result with a value missing.
    """
    if item.task not in _TASKS or item.st is None or item.et is None or not _complete(item):
        return None
    return Reading(offset + item.st, offset + item.et, _signal_of(item))


def _complete(item: ResultItem) -> bool:
    """Whether the result has every value its signal needs."""
    if not item.prediction:
        return False
    if item.task == "intensity":
        return item.prediction[0].score is not None
    return all(p.label is not None and p.posterior is not None for p in item.prediction)


def _signal_of(item: ResultItem) -> PerceptionSignal:
    if item.task == "intensity":
        return PerceptionSignal("intensity", value=round(float(item.prediction[0].score), 2))
    labels = [(p.label, float(p.posterior)) for p in item.prediction]
    name = _SIGNAL_NAMES.get(item.task, item.task)
    return PerceptionSignal(name, labels=sorted(labels, key=lambda label: -label[1]))
