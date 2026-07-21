import pytest


def _cache_or_skip():
    pytest.importorskip(
        "sglang.srt.speculative.sri_forest._sglang_sri_suffix_tree",
        reason="Build optional SRI suffix tree before running this test.",
    )
    from sglang.srt.speculative.sri_forest_cache import SRIForestDecodingCache

    return SRIForestDecodingCache


def test_dataset_tree_can_supply_best_suffix_path():
    cache_cls = _cache_or_skip()
    cache = cache_cls(
        max_tree_depth=16,
        global_cache_max_requests=0,
        dataset_cache_max_requests=1,
    )
    cache.start_request("dataset-request", [11, 12])
    cache.add_active_response("dataset-request", [13, 14, 15])
    cache.stop_request("dataset-request")

    cache.start_request("active-request", [99])
    draft = cache.speculate(
        "active-request",
        [11, 12, 13],
        max_spec_tokens=4,
        max_spec_factor=1.0,
        max_spec_offset=0.0,
        min_token_prob=0.1,
    )

    assert draft.source == "dataset"
    assert draft.match_len == 3
    assert draft.token_ids == [14, 15]
    assert draft.parents == [-1, 0]


def test_global_tree_is_fifo_bounded():
    cache_cls = _cache_or_skip()
    cache = cache_cls(
        max_tree_depth=8,
        global_cache_max_requests=1,
        dataset_cache_max_requests=0,
    )
    cache.start_request("first", [1, 2])
    cache.stop_request("first")
    cache.start_request("second", [3, 4])

    assert "first" not in cache.cached_requests
    assert "second" in cache.cached_requests
