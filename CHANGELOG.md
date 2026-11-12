# Changelog

## 0.2.0 (2026-11-12)

- Requires Pipecat 1.16 (`>=1.16.0,<1.17`). On Pipecat 1.15, use 0.1.0.
- Adds its reading to the one a transcription already carries, as Pipecat 1.16 producers do,
  instead of replacing it. Signals declared by a producer before it are declared again with its own.
- Renamed the `emotion` signal to `voice_emotion`, so it is not merged with Pipecat Tone's
  `emotion`, which is read from the words.
- An eager transcription with no text is no longer read, like a final one.
- The signal descriptions tell the LLM a reading covers a recent part of the turn, often not its
  end, instead of its latest part.
- README: names `PipecatVoiceReader`; removed the "Known limits" section, reordered the signals
  in the example reading, and fixed the example's wording.

## 0.1.0 (2026-11-11)

- `BehavioralSignalsPerceptionService`: streams the user's voice to Behavioral Signals and sets
  the `perception` of each transcription, for Pipecat 1.15's expressive conversation (RCS).
