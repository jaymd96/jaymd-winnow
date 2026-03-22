"""Thin wrapper around joblib.Memory for phase-keyed caching."""

from typing import Optional

from joblib import Memory


class PipelineCache:
    def __init__(self, cache_dir: Optional[str]):
        self._memory = Memory(location=cache_dir, verbose=0)

    def cache(self, func):
        return self._memory.cache(func)

    def clear(self):
        self._memory.clear(warn=False)
