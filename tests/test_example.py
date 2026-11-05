"""Builds the example's pipeline with made-up keys, without running it."""

import runpy
from types import SimpleNamespace
from pathlib import Path

from pipecat.processors.frame_processor import FrameProcessor

from pipecat_behavioralsignals import BehavioralSignalsPerceptionService


EXAMPLE = Path(__file__).parent.parent / "examples" / "bot.py"


class FakeTransport:
    def input(self) -> FrameProcessor:
        return FrameProcessor()

    def output(self) -> FrameProcessor:
        return FrameProcessor()

    def event_handler(self, name: str):
        return lambda handler: handler


WORKERS = []  # what the example gives the runner


class FakeRunner:
    def __init__(self, handle_sigint: bool):
        pass

    async def add_workers(self, worker) -> None:
        WORKERS.append(worker)

    async def run(self) -> None:
        pass


async def test_the_example_builds_its_pipeline(monkeypatch):
    for key in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "made-up")
    example = runpy.run_path(str(EXAMPLE))
    run_bot = example["run_bot"]
    monkeypatch.setitem(run_bot.__globals__, "WorkerRunner", FakeRunner)

    args = SimpleNamespace(pipeline_idle_timeout_secs=None, handle_sigint=False)
    await run_bot(FakeTransport(), args)

    (worker,) = WORKERS
    assert any(isinstance(p, BehavioralSignalsPerceptionService) for p in flat(worker.pipeline))


def flat(processor) -> list:
    """The processor and all those inside it."""
    inside = getattr(processor, "processors", [])
    return [processor] + [p for child in inside for p in flat(child)]
