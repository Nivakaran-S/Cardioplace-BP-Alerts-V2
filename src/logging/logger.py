"""Logging setup.

Two behaviours here are deliberate and both were bugs before.

1. The log FILE is created lazily. `logging.FileHandler` opens its file the moment it is
   constructed, so importing anything under `src` used to leave a timestamped empty log
   behind -- every `python -c "import src..."`, every test run, every CI import check. The
   directory filled with zero-byte files that made a real run's log hard to find.

2. stdout/stderr are reconfigured to UTF-8. MLflow and several dependencies print non-ASCII;
   on a Windows console defaulting to cp1252 the write raises UnicodeEncodeError, and
   because that happens inside a try/except in the tracking layer the visible symptom was
   every candidate run being "skipped" for an encoding reason, with nothing pointing here.
"""

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")


class _LazyFileHandler(logging.FileHandler):
    """A FileHandler that does not touch the filesystem until something is logged."""

    def __init__(self, filename, encoding=None):
        # delay=True is the whole point: the stream (and the file) is opened on first emit.
        super().__init__(filename, encoding=encoding, delay=True)

    def _open(self):
        os.makedirs(os.path.dirname(self.baseFilename) or ".", exist_ok=True)
        return super()._open()


logging.basicConfig(
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[_LazyFileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)


@contextmanager
def timer(label: str):
    """Log wall time for a block."""
    t0 = perf_counter()
    yield
    logging.info("%s -- %.1fs", label, perf_counter() - t0)
