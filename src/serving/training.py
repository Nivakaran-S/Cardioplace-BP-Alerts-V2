"""Single-flight training subprocess.

The API never imports the training graph -- it launches `main.py` as a child process. That
keeps a broken component out of the API's import path and lets the API keep serving the
rule engine while a run is in flight.
"""

import collections
import os
import signal
import subprocess
import sys
import threading

import pandas as pd

from src.logging.logger import logging
from src.serving import settings as S


class TrainingManager:
    def __init__(self, on_exit=None):
        self._lock = threading.Lock()
        self._proc = None
        self._tail = collections.deque(maxlen=S.TRAIN_TAIL_LINES)
        self._state = {"running": False, "started_at": None, "finished_at": None,
                       "returncode": None, "pid": None}
        self._on_exit = on_exit

    def start(self):
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return False, (f"a training run is already in progress (started "
                               f"{self._state['started_at']}, pid {self._state['pid']})")
            env = dict(os.environ)
            env.update(
                PYTHONUNBUFFERED="1",
                # The pipeline logs non-ASCII (arrows, >=). On a cp1252 Windows console
                # that raises UnicodeEncodeError mid-run and kills an hours-long job over a
                # log line. errors="replace" on the read side finishes the job.
                PYTHONIOENCODING="utf-8",
                MPLBACKEND="Agg")
            kwargs = {}
            if sys.platform == "win32":
                # Windows has no killpg; a new process group is the only way to deliver a
                # clean interrupt to the child later.
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self._tail.clear()
            self._proc = subprocess.Popen(
                [sys.executable, "-u", "main.py"],
                cwd=S.REPO_ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1, **kwargs)
            self._state = {"running": True,
                           "started_at": pd.Timestamp.utcnow().isoformat(),
                           "finished_at": None, "returncode": None,
                           "pid": self._proc.pid}
            threading.Thread(target=self._reader, daemon=True).start()
            logging.info("training started, pid %s", self._proc.pid)
            return True, "training started"

    def _reader(self):
        proc = self._proc
        try:
            for line in proc.stdout:
                self._tail.append(line.rstrip())
        except Exception as exc:                                     # noqa: BLE001
            self._tail.append(f"[reader error] {exc}")
        rc = proc.wait()
        self._state.update(running=False, returncode=rc,
                           finished_at=pd.Timestamp.utcnow().isoformat())
        self._tail.append(f"--- pipeline exited with code {rc} ---")
        logging.info("training finished, rc=%s", rc)
        if self._on_exit:
            try:
                # A completed run may have just promoted a new model.
                self._on_exit()
            except Exception:                                        # noqa: BLE001
                logging.exception("post-training refresh failed")

    def cancel(self):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return False, "no training run is in progress"
            try:
                if sys.platform == "win32":
                    self._proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self._proc.terminate()
            except Exception as exc:                                 # noqa: BLE001
                return False, f"could not signal the training process: {exc}"
            return True, "cancellation signalled"

    def status(self) -> dict:
        if self._proc is not None and self._proc.poll() is None:
            self._state["running"] = True
        return {**self._state, "tail": list(self._tail), "warning": S.TRAIN_WARNING}
