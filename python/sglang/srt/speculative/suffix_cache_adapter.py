"""Adapter from ArcticInference suffix cache to NGRAMWorker's cache interface."""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SuffixCacheAdapter:
    """Wrap ArcticInference's suffix cache with NGRAMWorker-compatible APIs."""

    def __init__(
        self,
        *,
        draft_token_num: int,
        max_batch_size: int,
        max_tree_depth: int,
        max_cached_requests: int,
        max_spec_factor: float,
        max_spec_offset: float,
        min_token_prob: float,
        max_spec_tokens: Optional[int],
    ):
        try:
            from arctic_inference.suffix_decoding import SuffixDecodingCache
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Suffix speculative decoding requires `arctic_inference` "
                "(pip install arctic-inference)."
            ) from exc

        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            max_cached_requests=max_cached_requests,
        )
        self.draft_token_num = draft_token_num
        self.max_batch_size = max_batch_size
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.max_spec_offset = max_spec_offset
        self.min_token_prob = min_token_prob
        self.max_spec_tokens_override = max_spec_tokens
        self.req_state: dict[str, list[object]] = {}

        max_total_drafts = max_batch_size * draft_token_num
        self.draft_buffer = np.empty((max_total_drafts,), dtype=np.int64)
        self.mask_buffer = np.empty(
            (max_batch_size, draft_token_num, draft_token_num),
            dtype=bool,
        )

    def batch_get(
        self,
        batch_req_ids: List[str],
        batch_prompts: List[List[int]],
        batch_tokens: List[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = len(batch_req_ids)
        if batch_size == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=bool)
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size={self.max_batch_size}"
            )

        total_draft_tokens = batch_size * self.draft_token_num
        draft_view = self.draft_buffer[:total_draft_tokens]
        mask_view = self.mask_buffer[:batch_size]
        draft_view.fill(0)
        mask_view.fill(False)

        self._cleanup_inactive_requests(set(batch_req_ids))

        for idx, (req_id, prompt, tokens) in enumerate(
            zip(batch_req_ids, batch_prompts, batch_tokens)
        ):
            cache_req_id, last_length = self._get_or_create_cache_req_id(
                req_id, prompt
            )
            last_length = self._sync_new_tokens(
                cache_req_id, req_id, tokens, last_length
            )

            context_token = tokens[-1] if tokens else 0
            pattern = tokens[max(0, len(tokens) - self.max_tree_depth) :]
            max_spec_tokens = max(self.draft_token_num - 1, 0)
            if self.max_spec_tokens_override is not None:
                max_spec_tokens = min(max_spec_tokens, self.max_spec_tokens_override)
            draft_ids, draft_parents = self._speculate(
                cache_req_id, pattern, max_spec_tokens
            )
            draft_ids, draft_parents = self._reorder_tree_bfs(draft_ids, draft_parents)
            draft_ids, draft_parents = self._inject_root_node(
                draft_ids, draft_parents, context_token
            )
            draft_ids, draft_parents, valid_len = self._pad_or_truncate(
                draft_ids, draft_parents
            )

            start = idx * self.draft_token_num
            end = start + self.draft_token_num
            draft_view[start:end] = draft_ids
            self._fill_tree_mask(mask_view[idx], draft_parents, valid_len)

        return (
            draft_view,
            mask_view.reshape(-1)[: total_draft_tokens * self.draft_token_num],
        )

    def batch_put(self, batch_req_ids: List[str], batch_tokens: List[List[int]]) -> None:
        # Cache deltas are synchronized in batch_get immediately before speculation.
        _ = batch_req_ids, batch_tokens

    def synchronize(self) -> None:
        pass

    def reset(self) -> None:
        for cache_req_id in list(getattr(self.suffix_cache, "active_requests", set())):
            self.suffix_cache.stop_request(cache_req_id)
        self.req_state.clear()

    def _cleanup_inactive_requests(self, active_req_ids: set[str]) -> None:
        for req_id in list(self.req_state):
            if req_id in active_req_ids:
                continue
            cache_req_id, _ = self.req_state.pop(req_id)
            if cache_req_id in getattr(self.suffix_cache, "active_requests", set()):
                self.suffix_cache.stop_request(cache_req_id)

    def _get_or_create_cache_req_id(
        self, req_id: str, prompt: List[int]
    ) -> tuple[str, int]:
        if req_id not in self.req_state:
            cache_req_id = req_id
            if cache_req_id in getattr(self.suffix_cache, "cached_requests", set()):
                self.suffix_cache.evict_cached_response(cache_req_id)
            self.suffix_cache.start_request(cache_req_id, list(prompt))
            self.req_state[req_id] = [cache_req_id, len(prompt)]

        cache_req_id, last_length = self.req_state[req_id]
        return str(cache_req_id), int(last_length)

    def _sync_new_tokens(
        self,
        cache_req_id: str,
        req_id: str,
        tokens: List[int],
        last_length: int,
    ) -> int:
        current_length = len(tokens)
        if current_length <= last_length:
            return last_length

        new_tokens = tokens[last_length:current_length]
        if cache_req_id not in getattr(self.suffix_cache, "active_requests", set()):
            raise RuntimeError(f"Suffix cache request {cache_req_id} is not active")
        self.suffix_cache.add_active_response(cache_req_id, list(new_tokens))
        self.req_state[req_id][1] = current_length
        return current_length

    def _speculate(
        self,
        cache_req_id: str,
        pattern: List[int],
        max_spec_tokens: int,
    ) -> tuple[List[int], List[int]]:
        if max_spec_tokens <= 0 or not pattern:
            return [], []

        draft = self.suffix_cache.speculate(
            cache_req_id,
            pattern,
            max_spec_tokens,
            self.max_spec_factor,
            self.max_spec_offset,
            self.min_token_prob,
        )
        token_ids = list(getattr(draft, "token_ids", []))
        parents = getattr(draft, "parents", None)
        if parents is None or len(parents) != len(token_ids):
            parents = [i - 1 for i in range(len(token_ids))]
        else:
            parents = [self._normalize_parent(parent) for parent in parents]
        return token_ids, parents

    @staticmethod
    def _normalize_parent(parent: Optional[int]) -> int:
        if parent is None:
            return -1
        parent = int(parent)
        return parent if parent >= 0 else -1

    def _pad_or_truncate(
        self, token_ids: List[int], parents: List[int]
    ) -> tuple[List[int], List[int], int]:
        valid_len = min(len(token_ids), self.draft_token_num)
        token_ids = token_ids[: self.draft_token_num]
        parents = parents[: self.draft_token_num]
        if len(token_ids) < self.draft_token_num:
            pad_len = self.draft_token_num - len(token_ids)
            token_ids.extend([0] * pad_len)
            parents.extend([0] * pad_len)
        return token_ids, parents, valid_len

    def _fill_tree_mask(
        self, mask: np.ndarray, parents: List[int], valid_len: int
    ) -> None:
        for idx in range(valid_len):
            mask[idx, idx] = True
            parent = parents[idx]
            while 0 <= parent < valid_len:
                mask[idx, parent] = True
                parent = parents[parent]

    @staticmethod
    def _reorder_tree_bfs(
        token_ids: List[int], parents: List[int]
    ) -> Tuple[List[int], List[int]]:
        n = len(token_ids)
        if n <= 1:
            return token_ids, parents

        children: List[List[int]] = [[] for _ in range(n)]
        roots: List[int] = []
        for idx, parent in enumerate(parents):
            if parent < 0 or parent >= n:
                roots.append(idx)
            else:
                children[parent].append(idx)
        if not roots:
            roots = [0]

        order: List[int] = []
        visited = [False] * n
        for root in roots:
            queue = deque([root])
            while queue:
                node = queue.popleft()
                if visited[node]:
                    continue
                visited[node] = True
                order.append(node)
                queue.extend(child for child in children[node] if not visited[child])
        order.extend(idx for idx in range(n) if not visited[idx])

        if order == list(range(n)):
            return token_ids, parents

        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
        reordered_ids = [token_ids[old_idx] for old_idx in order]
        reordered_parents = []
        for old_idx in order:
            parent = parents[old_idx]
            reordered_parents.append(remap[parent] if 0 <= parent < n else -1)
        return reordered_ids, reordered_parents

    @staticmethod
    def _inject_root_node(
        token_ids: List[int], parents: List[int], context_token: int
    ) -> Tuple[List[int], List[int]]:
        rooted_ids = [context_token]
        rooted_parents = [-1]
        rooted_ids.extend(token_ids)
        for parent in parents:
            rooted_parents.append(0 if parent < 0 else parent + 1)
        return rooted_ids, rooted_parents
