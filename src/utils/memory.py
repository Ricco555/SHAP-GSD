"""Process memory telemetry helpers."""

import resource


def _maxrss_mb() -> float:
    """Return the process' peak resident set size so far, in MiB.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS; this project only
    targets Linux (HPC + dev), so we assume kilobytes. It is a monotonically
    increasing high-water mark, not an instantaneous RSS reading.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
