# Bundled wheel

`track_certify-0.1.1-py3-none-any.whl` is this repository's source, built. It
is committed so the package can be installed, run and reviewed without reaching
a package index.

```bash
python -m pip install ./tools/track_certify-0.1.1-py3-none-any.whl
track-certify demo
```

SHA-256:

```
2ef5b74ec7f7d01126ac181e8ff53eb186bb35514f999cc4a718531085eb383a
```

Python 3.9 or newer; the only runtime dependencies are NumPy and SciPy. The
wheel carries no author, maintainer or contact metadata.

The build is reproducible. From a clean checkout at this commit:

```bash
SOURCE_DATE_EPOCH=1577836800 python -m build --wheel --outdir /tmp/w .
sha256sum /tmp/w/track_certify-0.1.1-py3-none-any.whl
```

Rebuild the wheel whenever the version in `pyproject.toml` changes, and update
the digest above in the same commit.
