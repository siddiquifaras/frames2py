"""File adapters: read recorded event data as ``EVENT_DTYPE`` arrays.

Each adapter is a module with its own ``open()`` and its own optional extra.
``import frames2py`` never imports an adapter or its dependencies.
"""
