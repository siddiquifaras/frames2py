# Installation

!!! note "Not on PyPI yet"
    Frames2Py has not been published to PyPI. `pip install frames2py` does not work yet;
    publishing is part of the 1.0 release work. Until then, install from the Git repository
    or from a wheel you build.

Frames2Py needs CPython 3.11 or newer and NumPy 2.4 or newer. The supported combinations of
Python version, platform and free-threaded build are on
[Supported Python and platforms](../reference/support.md).

## From the Git repository

=== "pip"

    ```sh
    pip install "frames2py @ git+https://github.com/siddiquifaras/frames2py"
    ```

=== "uv"

    ```sh
    uv pip install "frames2py @ git+https://github.com/siddiquifaras/frames2py"
    # or, in a uv project:
    uv add "frames2py @ git+https://github.com/siddiquifaras/frames2py"
    ```

The package is pure Python, so this builds a `py3-none-any` wheel locally from the source;
no compiler is needed.

## From a built wheel

Build the wheel and source distribution from a checkout, then install the wheel anywhere:

```sh
git clone https://github.com/siddiquifaras/frames2py.git
cd frames2py
uv build                      # writes dist/frames2py-0.1.0-py3-none-any.whl and the sdist
pip install dist/frames2py-0.1.0-py3-none-any.whl
```

If someone hands you the wheel file, install it the same way:
`pip install ./frames2py-0.1.0-py3-none-any.whl`.

## Optional extras

The core installs NumPy only. Everything that needs another library is an extra:

| extra | for | installs |
|---|---|---|
| `evt` | [EVT 2.0 / 3.0](../data/evt.md) RAW files | nothing beyond NumPy (the decoder is part of Frames2Py) |
| `aedat4` | [AEDAT 4.0](../data/aedat4.md) files | dv-processing >= 2.0.4 |
| `hdf5` | [HDF5](../data/hdf5.md) event files | h5py >= 3.16, hdf5plugin >= 7.1 |
| `recorder` | the [recorder](../data/recorder.md) | h5py >= 3.16, hdf5plugin >= 7.1 |
| `viewer` | the [viewer](../consumers/viewer.md) window | pyglet >= 2.1.16 |

Name the extras in brackets, with either install method:

=== "Git"

    ```sh
    pip install "frames2py[hdf5] @ git+https://github.com/siddiquifaras/frames2py"
    pip install "frames2py[recorder,viewer] @ git+https://github.com/siddiquifaras/frames2py"
    ```

=== "Wheel"

    ```sh
    pip install "dist/frames2py-0.1.0-py3-none-any.whl[hdf5]"
    pip install "dist/frames2py-0.1.0-py3-none-any.whl[recorder,viewer]"
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
0.1.0
```

Then run the [quickstart](quickstart.md).
