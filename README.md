# frontier-challenge

<p align="center">
    <em>Challenge for Frontier AI</em>
</p>

[![build](https://github.com/codingmaster8/frontier-challenge/workflows/Build/badge.svg)](https://github.com/codingmaster8/frontier-challenge/actions)
[![codecov](https://codecov.io/gh/codingmaster8/frontier-challenge/branch/master/graph/badge.svg)](https://codecov.io/gh/codingmaster8/frontier-challenge)
[![PyPI version](https://badge.fury.io/py/frontier-challenge.svg)](https://badge.fury.io/py/frontier-challenge)

---

**Documentation**: <a href="https://codingmaster8.github.io/frontier-challenge/" target="_blank">https://codingmaster8.github.io/frontier-challenge/</a>

**Source Code**: <a href="https://github.com/codingmaster8/frontier-challenge" target="_blank">https://github.com/codingmaster8/frontier-challenge</a>

---

## Development

### Setup environment

We use [uv](https://docs.astral.sh/uv/) to manage the development environment and production build. Ensure it's installed on your system.

### Run unit tests

You can run all the tests with:

```bash
uv run pytest
```

### Format the code

Execute the following command to apply linting and check typing:

```bash
uv run ruff format .
uv run ruff --fix .
uv run mypy frontier_challenge/
```

### Publish a new version

You can bump the version, create a commit and associated tag with one command:

```bash
uv version patch
```

```bash
uv version minor
```

```bash
uv version major
```

Your default Git text editor will open so you can add information about the release.

When you push the tag on GitHub, the workflow will automatically publish it on PyPi and a GitHub release will be created as draft.

## Serve the documentation

You can serve the Mkdocs documentation with:

```bash
uv run mkdocs serve
```

It'll automatically watch for changes in your code.

## License

This project is licensed under the terms of the MIT license.
