#
# Copyright (c) 2026, Behavioral Signal Technologies, Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A voice agent that hears how the user speaks, with Behavioral Signals.

Run it with `uv run examples/bot.py` and open http://localhost:7860/client in a browser.
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.frames.frames import LLMRunFrame
from pipecat.workers.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.expressive.perception import format_perception
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    UserTurnStoppedMessage,
    LLMUserAggregatorParams,
    LLMContextAggregatorPair,
)

from pipecat_behavioralsignals import BehavioralSignalsPerceptionService


load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    # Reads how the user speaks from their voice, and adds it to each turn.
    # Uses BEHAVIORALSIGNALS_CID and BEHAVIORALSIGNALS_API_KEY.
    perception = BehavioralSignalsPerceptionService()

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction="You are a helpful assistant in a voice conversation. Your "
            "responses will be spoken aloud, so avoid emojis, bullet points, or other formatting "
            "that can't be spoken. Keep your responses brief."
        ),
    )

    tts = DeepgramTTSService(api_key=os.environ["DEEPGRAM_API_KEY"])

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            perception,  # after the STT service, which passes the audio on
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        # The LLM reads the perception only with expressive conversation on.
        params=PipelineParams(enable_expressive_conversation=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        if message.perception:
            logger.info(f"Voice: {format_perception(message.perception)}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message({"role": "developer", "content": "Please introduce yourself."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
