# Adapters API

Each adapter module's public API (its `__all__`) is one function, `open()`, which returns a
`Reader`. Other names visible in the modules with `dir()`, `Reader` included, are not part
of the public API and may change; `Reader` is documented here because every `open()`
returns one. See [Adapters](../../data/adapters.md) for how they behave. In the
signatures, `PathArg` is `str | os.PathLike[str]`: a string or a `pathlib.Path`.

::: frames2py.adapters.evt.open

::: frames2py.adapters.aedat4.open

::: frames2py.adapters.hdf5.open

::: frames2py.adapters._reader.Reader
    options:
      show_root_full_path: false
