# Bundled wheel

`track_certify-0.1.1-py3-none-any.whl` is this repository's source, built. It
is committed so the package can be installed, run and reviewed without reaching
a package index, and therefore without an index account page being part of the
install path. It is the only supported install route for this review copy.

```bash
python -m pip install ./tools/track_certify-0.1.1-py3-none-any.whl
track-certify demo
```

SHA-256:

```
9d5be886fc6d0f667e411f923ddd16eb688da417aadec116a0541070b0935a54
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
