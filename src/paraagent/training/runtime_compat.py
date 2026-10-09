"""Runtime adaptations for Python 3.12.0 training."""

import importlib.util
import threading

class _TrackerRLock(threading._RLock):
    """Python RLock with the recursion query required by multiprocess."""

    def _recursion_count(self):
        return self._count if self._is_owned() else 0

def patch_multiprocess_tracker_lock():
    """Provide _recursion_count on the multiprocess tracker lock before workers start.
    Use the Python RLock only when the existing lock lacks this method;
    preserve reentrancy checks.
    """
    if hasattr(threading.RLock(), "_recursion_count"):
        return False
    if importlib.util.find_spec("multiprocess") is None:
        return False

    from multiprocess import resource_tracker

    tracker = resource_tracker._resource_tracker
    if hasattr(tracker._lock, "_recursion_count"):
        return False
    tracker._lock = _TrackerRLock()
    return True
