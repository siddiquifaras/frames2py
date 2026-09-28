"""Look at an Engine's snapshots: ``render()`` turns one into an RGB image, ``run()`` shows
them in a window.

::

    import threading
    from frames2py import viewer

    producer = threading.Thread(target=feed, args=(engine,), daemon=True)  # calls engine.ingest()
    producer.start()
    viewer.run(engine.snapshot)  # on the main thread, until the window is closed

``render()`` needs only NumPy. ``run()`` needs the ``frames2py[viewer]`` extra (pyglet), must
be called on the main thread and starts no thread of its own. The viewer is a consumer like
any other: it reads published snapshots at its own cadence and never touches the ingest path.
"""

from frames2py.viewer._render import render
from frames2py.viewer._run import run

__all__ = ["render", "run"]
