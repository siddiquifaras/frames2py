"""Optional adapters for event camera data sources and formats.

Each adapter is a thin wrapper around a vendor SDK or file format parser.
All dependencies are **lazily imported** -- the core library has no hard
dependency on any vendor SDK.

Available adapters:

- :func:`from_h5` -- HDF5 files (requires ``h5py``)
- :func:`from_aedat4` -- AEDAT4 binary files (pure Python, no vendor SDK)
- :func:`from_prophesee` -- Prophesee cameras/RAW files
  (requires ``metavision_core``)
- :func:`from_inivation` -- iniVation cameras/AEDAT4 via dv-processing
  (requires ``dv_processing``)
- :func:`from_udp` -- UDP event streams (pure Python)

Conversion utilities:

- :func:`convert.convert` -- convert between any two formats
- :func:`convert.read_events` -- read events from any format
- :func:`convert.write_events` -- write events to any format
"""

__all__: list[str] = []
