# Installation

Frames2Py needs CPython 3.11 or newer and NumPy 2.4 or newer. The supported combinations of
Python version, platform and free-threaded build are on
[Supported Python and platforms](../reference/support.md).

## From PyPI

=== "pip"

    ```sh
    pip install frames2py
    ```

=== "uv"

    ```sh
    uv pip install frames2py
    # or, in a uv project:
    uv add frames2py
    ```

The package is pure Python (a `py3-none-any` wheel); no compiler is needed.

!!! note "The release candidate"
    The current release is 1.0.0rc1, a release candidate for 1.0.0. pip and uv skip
    pre-releases when a stable release exists, and install one only when none does: while
    1.0.0rc1 is the only release, `pip install frames2py` installs it, and once 1.0.0 is
    published it installs 1.0.0 instead. To ask for the release candidate explicitly:

    ```sh
    pip install frames2py==1.0.0rc1
    pip install --pre frames2py   # the newest release, pre-releases included
    ```

## Optional extras

The core installs NumPy only. Everything that needs another library is an extra:

| extra | for | installs |
|---|---|---|
| `evt` | [EVT 2.0 / 3.0](../data/evt.md) RAW files | nothing beyond NumPy (the decoder is part of Frames2Py) |
| `aedat4` | [AEDAT 4.0](../data/aedat4.md) files | dv-processing >= 2.0.4 |
| `hdf5` | [HDF5](../data/hdf5.md) event files | h5py >= 3.16, hdf5plugin >= 7.1 |
| `recorder` | the [recorder](../data/recorder.md) | h5py >= 3.16, hdf5plugin >= 7.1 |
| `viewer` | the [viewer](../consumers/viewer.md) window | pyglet >= 2.1.16 |

Name the extras in brackets, quoted so the shell leaves the brackets alone:

```sh
pip install "frames2py[hdf5]"
pip install "frames2py[recorder,viewer]"
```

`import frames2py` never imports an extra's library. A function that needs one raises
`ImportError` naming the extra to install when it is called, not before.

On Linux, dv-processing's wheels load the system's `libatomic.so.1`, which minimal images
leave out: on Debian or Ubuntu, `apt install libatomic1`. dv-processing publishes macOS
wheels for macOS 15 and later only.

## Check the installation

```python
import frames2py

print(frames2py.__version__)
```

```text title="Output"
1.0.0rc1
```

Then run the [quickstart](quickstart.md).

## Development version and source

These are for trying unreleased changes and for working on Frames2Py; the releases above
are what to depend on.

- **The development version**, `main` from the Git repository:

    ```sh
    pip install "frames2py @ git+https://github.com/siddiquifaras/frames2py"
    pip install "frames2py[hdf5] @ git+https://github.com/siddiquifaras/frames2py"
    ```

- **A checkout**, set up for development with every extra and the test tools
  ([Testing](../development/testing.md), [Contributing](../development/contributing.md)):

    ```sh
    git clone https://github.com/siddiquifaras/frames2py.git
    cd frames2py
    uv sync --all-extras
    ```

- **A wheel built from a checkout**, to install elsewhere:

    ```sh
    uv build          # the wheel and the sdist, in dist/
    pip install dist/frames2py-*-py3-none-any.whl
    ```
