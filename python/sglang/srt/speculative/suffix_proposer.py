"""Suffix decoding proposer that wraps ArcticInference's suffix cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.server_args import ServerArgs

try:
    from arctic_inference.suffix_decoding import (
        SuffixDecodingCache,
        SuffixDecodingDraft,
    )
except Exception:  # pragma: no cover - optional dependency may be missing
    SuffixDecodingCache = None  # type: ignore
    SuffixDecodingDraft = None  # type: ignore


@dataclass
class SuffixProposal:
    """Lightweight wrapper around `SuffixDecodingDraft`."""

    token_ids: List[int]
    score: float
    match_len: int
    source: str = "arctic"

    @classmethod
    def from_draft(cls, draft: SuffixDecodingDraft) -> "SuffixProposal":  # type: ignore[valid-type]
        return cls(
            token_ids=list(draft.token_ids),
            score=float(draft.score),
            match_len=int(draft.match_len),
            source=str(getattr(draft, "source", "arctic")),
        )


class SuffixDecodingProposer:
    """Manages suffix cache lifecycle and speculation."""

    def __init__(
        self,
        server_args: ServerArgs,
        *,
        max_model_len: int,
    ):
        if (
            server_args.speculative_suffix_backend == "arctic"
            and SuffixDecodingCache is None
        ):
            raise RuntimeError(
                "Suffix decoding requires `arctic_inference` (pip install arctic-inference)."
            )

        self._max_model_len = max_model_len
        self._max_tree_depth = server_args.speculative_suffix_cache_max_depth
        self._max_spec_factor = server_args.speculative_suffix_max_spec_factor
        self._max_spec_offset = server_args.speculative_suffix_max_spec_offset
        self._min_token_prob = server_args.speculative_suffix_min_token_prob
        self._max_cached_requests = (
            server_args.speculative_suffix_cache_max_requests
        )
        self._max_spec_tokens_override = (
            server_args.speculative_suffix_max_spec_tokens
        )
        if server_args.speculative_suffix_backend == "sri_forest":
            from sglang.srt.speculative.sri_forest_cache import SRIForestDecodingCache

            self._cache = SRIForestDecodingCache(
                max_tree_depth=self._max_tree_depth,
                global_cache_max_requests=self._max_cached_requests,
                dataset_cache_max_requests=(
                    server_args.speculative_suffix_dataset_cache_max_requests
                ),
            )
        else:
            self._cache = SuffixDecodingCache(
                max_tree_depth=self._max_tree_depth,
                max_cached_requests=self._max_cached_requests,
            )

    # ------------------------------------------------------------------ #
    # Cache lifecycle helpers
    # ------------------------------------------------------------------ #
    def ensure_prompts_started(self, batch: ScheduleBatch) -> None:
        """Make sure each request has a prompt tree in the cache."""
        for req in batch.reqs:
            if req.rid in self._cache.active_requests:
                continue
            # SuffixDecodingCache expects a Python sequence of ints, not a Tensor
            prompt = list(req.origin_input_ids)
            if req.rid in self._cache.cached_requests:
                self._cache.evict_cached_response(req.rid)
            self._cache.start_request(req.rid, prompt)

    def stop_inactive_requests(self, active_ids: Iterable[str]) -> None:
        active = set(active_ids)
        for req_id in list(self._cache.active_requests):
            if req_id not in active:
                self._cache.stop_request(req_id)

    def add_accepted_tokens(
        self,
        req_id: str,
        token_ids: Sequence[int],
    ) -> None:
        if not token_ids:
            return
        # Pass a plain Python list to the cache to match the C++ binding
        tokens = list(token_ids)
        if req_id not in self._cache.active_requests:
            raise RuntimeError(f"Request {req_id} not active in suffix cache")
        self._cache.add_active_response(req_id, tokens)

    # ------------------------------------------------------------------ #
    # Proposal
    # ------------------------------------------------------------------ #
    def propose(self, batch: ScheduleBatch) -> List[Optional[SuffixProposal]]:
        proposals: List[Optional[SuffixProposal]] = []
        for req in batch.reqs:
            seq_len = req.origin_input_ids.__len__() + len(req.output_ids)
            if seq_len >= self._max_model_len:
                proposals.append(None)
                continue

            start = max(0, seq_len - self._max_tree_depth)
            context = (req.origin_input_ids + req.output_ids)[start:seq_len]
            max_spec_tokens = self._max_spec_tokens(seq_len)
            # 使用位置参数以兼容 arctic_inference 的 C++ 绑定（可能为“仅位置”参数）
            draft = self._cache.speculate(
                req.rid,
                context,
                max_spec_tokens,
                self._max_spec_factor,
                self._max_spec_offset,
                self._min_token_prob,
            )
            proposals.append(SuffixProposal.from_draft(draft))
        self.stop_inactive_requests(req.rid for req in batch.reqs)
        return proposals

    def _max_spec_tokens(self, seq_len: int) -> int:
        """
        Determine the maximum number of suffix draft tokens to propose.

        Priority:
        1) If an explicit suffix cap is provided via
           --speculative-suffix-max-spec-tokens, honor it.
        2) Otherwise default to the suffix cache tree depth, mimicking
           Arctic's behavior. This avoids coupling with the global
           --speculative-num-draft-tokens intended for EAGLE drafts.
        """
        headroom = max(self._max_model_len - seq_len - 1, 0)
        if self._max_spec_tokens_override is not None:
            limit = min(self._max_spec_tokens_override, headroom)
        else:
            # Default to limiting by the suffix tree depth
            limit = min(self._max_tree_depth, headroom)
        return limit
