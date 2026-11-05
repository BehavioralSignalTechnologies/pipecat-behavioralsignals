# Contributing

Thanks for helping improve pipecat-behavioralsignals. By taking part, you agree to follow our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Report a bug or ask for a feature

Open an [issue](https://github.com/BehavioralSignalTechnologies/pipecat-behavioralsignals/issues/new/choose).
Never paste your API key or CID in an issue.

To report a security problem, follow [SECURITY.md](SECURITY.md) instead.

## Set up

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/BehavioralSignalTechnologies/pipecat-behavioralsignals.git
cd pipecat-behavioralsignals
uv sync
```

CI uses the versions pinned in `uv.lock`. If you change `pyproject.toml`, run `uv lock` and commit
`uv.lock`, or CI may fail.

To run the example, get a CID and API key from the
[Behavioral Signals portal](https://portal.behavioralsignals.com/), and see
[Running the example](README.md#running-the-example).

## Code style

We use [ruff](https://docs.astral.sh/ruff/) with the settings in `ruff.toml` (line length 100).
Before you open a pull request, run:

```bash
uv run ruff check
uv run ruff format
```

## Tests

The tests fake the Behavioral Signals client, so they need no credentials. Run them with:

```bash
uv run pytest
```

CI runs ruff and the tests on Python 3.12 and 3.13 for every pull request. It also builds the
package and installs the built wheel with the newest versions allowed, to check it as users get it.

## Pull requests

1. Create a branch from `main`.
2. Keep each pull request to one change.
3. If you change how the service is used, update `README.md` and `examples/`.
4. Add a line to `CHANGELOG.md`.
5. If the change needs a new release, bump `version` in `pyproject.toml` and run `uv lock`.
6. Open the pull request and fill in the template.

## Releases

Maintainers release from `main`. Publish a GitHub release with a tag like `v0.1.0` that matches
`version` in `pyproject.toml`. The Release workflow then builds the package and uploads it to PyPI.
