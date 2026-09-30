# Releasing decisionlane

Publishing is manual. Nothing in CI uploads packages.

## 1. Prepare

1. Update `version` in `pyproject.toml` (the only place the version lives).
2. Add the release date to `CHANGELOG.md`.
3. Start from a clean tree with no local `dist/` directory:

```bash
rm -rf dist build
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

## 2. Build and check

```bash
python -m build            # writes dist/decisionlane-X.Y.Z.tar.gz and a wheel
python -m twine check --strict dist/*
```

Inspect the contents:

```bash
python -m zipfile -l dist/decisionlane-*.whl
tar -tzf dist/decisionlane-*.tar.gz
```

The wheel should contain only `decisionlane/**/*.py`, `decisionlane/py.typed`
and the `.dist-info` metadata (including `licenses/LICENSE`). The sdist should
contain `src/`, `tests/`, `README.md`, `CHANGELOG.md`, `RELEASING.md`,
`LICENSE`, `pyproject.toml` and `PKG-INFO`, and never `.env` files.

## 3. Test the built wheel in a clean environment

Run this from outside the checkout so the source tree cannot be imported:

```bash
python -m venv /tmp/dl-check
/tmp/dl-check/bin/pip install dist/decisionlane-*.whl
cd /tmp && /tmp/dl-check/bin/python -c "import decisionlane; print(decisionlane.__file__)"
```

Optionally, check that the sdist alone can build a wheel:

```bash
mkdir /tmp/dl-sdist && tar -xzf dist/decisionlane-*.tar.gz -C /tmp/dl-sdist
python -m build --wheel /tmp/dl-sdist/decisionlane-*/
```

## 4. Upload to TestPyPI, then PyPI

You need accounts and API tokens on https://test.pypi.org and https://pypi.org.
Twine reads the username `__token__` and the token as the password, either
interactively or from `TWINE_USERNAME` / `TWINE_PASSWORD`.

```bash
python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI (dependencies come from the real PyPI):

```bash
python -m venv /tmp/dl-testpypi
/tmp/dl-testpypi/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  decisionlane==X.Y.Z
```

When that works:

```bash
python -m twine upload dist/*
```

A version number can only be uploaded once per index. To fix a mistake, bump
the version and rebuild.
