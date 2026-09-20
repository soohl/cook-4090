"""Single-owner, disposable storage for a UI server lifetime."""

import fcntl
from pathlib import Path
import shutil
import tempfile


class Runtime:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("Another cook-4090 server owns this runtime directory") from None
        for stale in self.root.glob("session-*"):
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
        self.path = Path(tempfile.mkdtemp(prefix="session-", dir=self.root))

    def close(self):
        if self.lock.closed:
            return
        shutil.rmtree(self.path, ignore_errors=True)
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
