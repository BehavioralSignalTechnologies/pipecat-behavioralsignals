# pipecat-behavioralsignals

<p align="center">
  <img src="https://raw.githubusercontent.com/BehavioralSignalTechnologies/behavioralsignals-python/main/assets/logo.png" alt="Behavioral Signal Technologies"/>
</p>

<div align="center">

[![Discord](https://badgen.net/discord/members/fxjRrbMH3Q/?color=8978cc&icon=discord)](https://discord.com/invite/fxjRrbMH3Q)
[![readme.io](https://badgen.net/badge/readme.io/Documentation/?color=black)](https://behavioralsignals.readme.io/)
[![CI](https://github.com/BehavioralSignalTechnologies/pipecat-behavioralsignals/actions/workflows/ci.yml/badge.svg)](https://github.com/BehavioralSignalTechnologies/pipecat-behavioralsignals/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyPI](https://badgen.net/pypi/v/pipecat-behavioralsignals)](https://pypi.org/project/pipecat-behavioralsignals/)
[![Python](https://badgen.net/pypi/python/pipecat-behavioralsignals)](https://pypi.org/project/pipecat-behavioralsignals/)
[![Pipecat](https://badgen.net/badge/Pipecat/1.15/blue)](https://github.com/pipecat-ai/pipecat)
[![License](https://badgen.net/badge/license/BSD-2-Clause/blue)](https://github.com/BehavioralSignalTechnologies/pipecat-behavioralsignals/blob/main/LICENSE)
[![Contributor Covenant](https://badgen.net/badge/Contributor%20Covenant/2.1/4baaaa)](https://github.com/BehavioralSignalTechnologies/pipecat-behavioralsignals/blob/main/CODE_OF_CONDUCT.md)

</div>

Lets a [Pipecat](https://github.com/pipecat-ai/pipecat) voice agent hear how the user speaks, not
just what they say. It streams the user's voice to the
[Behavioral Signals](https://behavioralsignals.com/) API and adds what it reads, such as emotion,
energy and hesitation, to each user turn. The LLM reads it with the user's words.

It uses Pipecat's Real-Time Context Standard (RCS): it is a perception producer. Pipecat's own
`PipecatToneReader` reads the words; this reads the voice.

Maintained by Behavioral Signal Technologies, the company behind the Behavioral Signals API.

## Installation

```bash
uv add pipecat-behavioralsignals
```

Or with pip:

```bash
pip install pipecat-behavioralsignals
```

Requires Python 3.12 or later and Pipecat 1.15.

## Prerequisites

A Behavioral Signals account. Create one and an API key in the
[Behavioral Signals portal](https://portal.behavioralsignals.com/). Set your client ID (CID) and API key:

```bash
export BEHAVIORALSIGNALS_CID=...
export BEHAVIORALSIGNALS_API_KEY=...
```

You can also pass them to the service as `cid` and `api_key`.

## Usage

Put `BehavioralSignalsPerceptionService` right after your STT service, and turn on expressive
conversation:

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker

from pipecat_behavioralsignals import BehavioralSignalsPerceptionService

perception = BehavioralSignalsPerceptionService()

pipeline = Pipeline([
    transport.input(),
    stt,
    perception,
    user_aggregator,
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])

worker = PipelineWorker(pipeline, params=PipelineParams(enable_expressive_conversation=True))
```

Things to know:

- It reads the audio that the STT service passes on. STT services do that by default; do not set
  `audio_passthrough=False`.
- It expects 16-bit mono audio, which is what Pipecat transports give by default. On audio with
  more channels it reports an error and stops reading.
- A pipeline has one perception producer. Use this instead of `PipecatToneReader`, not with it.
- Without `enable_expressive_conversation=True`, the LLM does not see the readings.

## What the LLM sees

At start, the service declares its signals in a `PerceptionMetadataFrame`. Pipecat uses it to tell
the LLM how to read them. Then each user message starts with the reading, like this:

```
([emotion angry=0.71, neutral=0.20, sad=0.06, happy=0.03],[strength strong=0.64, neutral=0.30, weak=0.06],[positivity negative=0.72, neutral=0.20, positive=0.08],[speaking_rate fast=0.50, normal=0.40, slow=0.10],[hesitation no=0.80, yes=0.20],[engagement engaged=0.55, neutral=0.35, withdrawn=0.10],[intensity value=0.61]) I already told you, I was charged twice.
```

The signals have the names of the Behavioral Signals tasks:

| Signal | What it reads | Labels |
|---|---|---|
| `emotion` | The emotion in the voice | `happy`, `angry`, `sad`, `neutral` |
| `strength` | Arousal: the energy in the voice | `strong`, `weak`, `neutral` |
| `positivity` | Valence: how positive the tone of voice is | `positive`, `negative`, `neutral` |
| `speaking_rate` | How fast the user speaks, compared to speakers in general | `fast`, `slow`, `normal` |
| `hesitation` | Whether the user sounds hesitant | `yes`, `no` |
| `engagement` | Whether the user sounds involved or detached | `engaged`, `withdrawn`, `neutral` |
| `intensity` | How intense the emotion is, whichever emotion it is | a value from 0 to 1 |

Each label has its probability, likeliest first.
The API's speaker signals (age and gender) are not sent.

## How it works

Behavioral Signals reads the voice in short windows of about 2 seconds, and each reading arrives
about 1 to 2 seconds after its audio. On each final transcription, the service takes the latest
reading of each signal that overlaps the transcription's audio: the audio since the previous
final transcription. So a reading usually covers the start or middle of a turn rather than its
last words, and a very short turn may have no reading. Then the message has no reading at the start.

## Your users' audio

The service sends the user's audio to the Behavioral Signals API. The
[SDK README](https://github.com/BehavioralSignalTechnologies/behavioralsignals-python) says a
stream's results are saved there and can be fetched later. Tell your users, and check that this
fits your privacy terms.

## Known limits

- If the stream keeps failing, for example with wrong credentials or no credits left, the service
  reports one error upstream and keeps retrying, in case the cause passes. Turns get no readings
  meanwhile.
- If the API stops reading audio, the service keeps up to about 20 seconds of it. Beyond that it
  drops the audio and starts a new stream.
- When the pipeline ends, the service closes its stream. But it cannot stop a call to the API that
  hangs: the SDK has no way to cancel it. Then the stream's thread lives on until the call returns.
  It is a daemon thread, so it does not keep the process from exiting.

## Running the example

[examples/bot.py](https://github.com/BehavioralSignalTechnologies/pipecat-behavioralsignals/blob/main/examples/bot.py) is a voice agent you talk to in the browser. It uses Deepgram
for speech to text and speech, and OpenAI for the LLM.

```bash
uv sync
cp examples/env.example .env   # then fill in the keys
uv run examples/bot.py
```

Open http://localhost:7860/client and talk. The bot logs each reading it gets, as `Voice: ...`.

## Compatibility

Tested with Pipecat v1.15.0. RCS is an alpha in Pipecat, so its types may change between releases.

## Development

```bash
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
```

The tests need no credentials: the Behavioral Signals client is faked. See
[CONTRIBUTING.md](https://github.com/BehavioralSignalTechnologies/pipecat-behavioralsignals/blob/main/CONTRIBUTING.md) for more, and
[SECURITY.md](https://github.com/BehavioralSignalTechnologies/pipecat-behavioralsignals/blob/main/SECURITY.md) to report a security problem.

## License

BSD 2-Clause. See [LICENSE](https://github.com/BehavioralSignalTechnologies/pipecat-behavioralsignals/blob/main/LICENSE).
