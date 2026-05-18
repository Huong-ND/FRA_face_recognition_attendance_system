"""
In-memory gallery cache.
Reload from DB whenever students are added/removed.
"""

import threading
import numpy as np
from models.student_model import load_gallery

_lock  = threading.Lock()
_names: list[str]   = []
_matrix: np.ndarray = np.empty((0, 512), dtype=np.float32)


def refresh():
    """Reload gallery from DB into memory."""
    global _names, _matrix
    names, mat = load_gallery()
    with _lock:
        _names  = names
        _matrix = mat
    print(f"[cache] Gallery refreshed: {len(_names)} students.")


def get_gallery() -> tuple[list[str], np.ndarray]:
    with _lock:
        return list(_names), _matrix.copy()


# Load once at import time
refresh()
