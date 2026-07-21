"""SRI-style local/global/dataset suffix forest for Standalone suffix proposals.

This is deliberately a proposal backend only. It returns the best *linear*
path so the existing FA3 ragged verifier remains unchanged. The optional
dataset tree is populated by the first N requests and retained as a stable
corpus, while the global tree is FIFO-evicted independently.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable, Iterable, KeysView, Optional, Sequence


@dataclass
class SRIForestDraft:
    token_ids: list[int]
    parents: list[int]
    probs: list[float]
    score: float
    match_len: int
    source: str


class SRIForestDecodingCache:
    """Local request trees plus rolling global and optional stable dataset trees."""

    def __init__(
        self,
        *,
        max_tree_depth: int,
        global_cache_max_requests: int,
        dataset_cache_max_requests: int,
    ) -> None:
        if global_cache_max_requests < -1:
            raise ValueError("global_cache_max_requests must be -1 or non-negative")
        if dataset_cache_max_requests < 0:
            raise ValueError("dataset_cache_max_requests must be non-negative")
        try:
            from sglang.srt.speculative.sri_forest._sglang_sri_suffix_tree import (
                SuffixTree,
            )
        except ImportError as exc:  # pragma: no cover - exercised at deployment
            raise RuntimeError(
                "SRI forest suffix backend is not compiled. Run: "
                "python python/sglang/srt/speculative/sri_forest/setup.py "
                "build_ext --inplace"
            ) from exc

        self._tree_type = SuffixTree
        self._max_tree_depth = max_tree_depth
        self._global_limit = global_cache_max_requests
        self._dataset_limit = dataset_cache_max_requests
        self._local_trees: dict[Hashable, object] = {}
        self._global_tree = SuffixTree(max_tree_depth) if global_cache_max_requests else None
        self._dataset_tree = SuffixTree(max_tree_depth) if dataset_cache_max_requests else None
        self._global_ids: OrderedDict[Hashable, int] = OrderedDict()
        self._dataset_ids: OrderedDict[Hashable, int] = OrderedDict()
        self._next_sequence_id = 0

    @property
    def active_requests(self) -> KeysView[Hashable]:
        return self._local_trees.keys()

    @property
    def cached_requests(self) -> set[Hashable]:
        # The proposer only needs membership testing. A request exists in one
        # cross-request tree at a time, so a materialized union is sufficient.
        return set(self._global_ids) | set(self._dataset_ids)

    def start_request(self, req_id: Hashable, prompt_token_ids: Sequence[int]) -> None:
        if req_id in self._local_trees:
            raise ValueError(f"Request {req_id!r} is already active")
        if req_id in self._global_ids or req_id in self._dataset_ids:
            self.evict_cached_response(req_id)

        local_tree = self._tree_type(self._max_tree_depth)
        local_tree.extend(0, list(prompt_token_ids))
        self._local_trees[req_id] = local_tree

        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1
        if self._dataset_tree is not None and len(self._dataset_ids) < self._dataset_limit:
            self._dataset_tree.extend(sequence_id, list(prompt_token_ids))
            self._dataset_ids[req_id] = sequence_id
        elif self._global_tree is not None:
            self._global_tree.extend(sequence_id, list(prompt_token_ids))
            self._global_ids[req_id] = sequence_id
            self._evict_global_if_needed()

    def stop_request(self, req_id: Hashable) -> None:
        if req_id not in self._local_trees:
            raise ValueError(f"Request {req_id!r} is not active")
        del self._local_trees[req_id]

    def add_active_response(
        self, req_id: Hashable, token_ids: int | Sequence[int]
    ) -> None:
        if req_id not in self._local_trees:
            raise ValueError(f"Request {req_id!r} is not active")
        tokens = [int(token_ids)] if isinstance(token_ids, int) else list(token_ids)
        if not tokens:
            return
        self._local_trees[req_id].extend(0, tokens)
        if req_id in self._global_ids:
            self._global_tree.extend(self._global_ids[req_id], tokens)
        if req_id in self._dataset_ids:
            self._dataset_tree.extend(self._dataset_ids[req_id], tokens)

    def evict_cached_response(self, req_id: Hashable) -> None:
        sequence_id = self._global_ids.pop(req_id, None)
        if sequence_id is not None:
            self._global_tree.remove(sequence_id)
            return
        sequence_id = self._dataset_ids.pop(req_id, None)
        if sequence_id is not None:
            self._dataset_tree.remove(sequence_id)
            return
        raise ValueError(f"Request {req_id!r} is not cached")

    def speculate(
        self,
        req_id: Hashable,
        pattern: Sequence[int],
        max_spec_tokens: int,
        max_spec_factor: float,
        max_spec_offset: float,
        min_token_prob: float,
    ) -> SRIForestDraft:
        if req_id not in self._local_trees:
            raise ValueError(f"Request {req_id!r} is not active")
        candidates = [
            ("local", self._local_trees[req_id].speculate(
                list(pattern),
                max_spec_tokens,
                max_spec_factor,
                max_spec_offset,
                min_token_prob,
            ))
        ]
        if self._dataset_tree is not None:
            candidates.append(
                ("dataset", self._dataset_tree.speculate(
                    list(pattern),
                    max_spec_tokens,
                    max_spec_factor,
                    max_spec_offset,
                    min_token_prob,
                ))
            )
        if self._global_tree is not None:
            candidates.append(
                ("global", self._global_tree.speculate(
                    list(pattern),
                    max_spec_tokens,
                    max_spec_factor,
                    max_spec_offset,
                    min_token_prob,
                ))
            )
        source, best = max(candidates, key=lambda item: float(item[1].score))
        return SRIForestDraft(
            token_ids=list(best.token_ids),
            parents=list(best.parents),
            probs=list(best.probs),
            score=float(best.score),
            match_len=int(best.match_len),
            source=source,
        )

    def stop_inactive_requests(self, active_ids: Iterable[Hashable]) -> None:
        active = set(active_ids)
        for req_id in list(self._local_trees):
            if req_id not in active:
                self.stop_request(req_id)

    def _evict_global_if_needed(self) -> None:
        if self._global_limit < 0:
            return
        while len(self._global_ids) > self._global_limit:
            req_id, sequence_id = self._global_ids.popitem(last=False)
            self._global_tree.remove(sequence_id)
