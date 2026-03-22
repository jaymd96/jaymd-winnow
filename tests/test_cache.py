"""Tests for caching wrapper."""

import tempfile

from jaymd_winnow.cache import PipelineCache


def test_noop_when_no_dir():
    cache = PipelineCache(cache_dir=None)
    call_count = 0

    def add(a, b):
        nonlocal call_count
        call_count += 1
        return a + b

    cached_add = cache.cache(add)
    assert cached_add(1, 2) == 3
    assert cached_add(1, 2) == 3
    # Without real caching, should still work (Memory with None is a passthrough)
    assert call_count == 2


def test_cache_hit_with_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = PipelineCache(cache_dir=tmpdir)
        call_count = 0

        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        cached_add = cache.cache(add)
        assert cached_add(1, 2) == 3
        assert cached_add(1, 2) == 3
        # Second call should be a cache hit
        assert call_count == 1


def test_cache_miss_different_args():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = PipelineCache(cache_dir=tmpdir)
        call_count = 0

        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        cached_add = cache.cache(add)
        assert cached_add(1, 2) == 3
        assert cached_add(3, 4) == 7
        assert call_count == 2
