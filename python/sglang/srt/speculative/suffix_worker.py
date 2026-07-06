"""Standalone suffix speculative decoding worker."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.suffix_cache_adapter import SuffixCacheAdapter


class SuffixWorker(NGRAMWorker):
    """Suffix decoding worker that reuses NGRAMWorker's verify path."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        self.spec_algorithm = SpeculativeAlgorithm.SUFFIX

    def _init_ngram_cache(self, server_args: ServerArgs):
        return SuffixCacheAdapter(
            draft_token_num=server_args.speculative_num_draft_tokens,
            max_batch_size=self.max_batch_size,
            max_tree_depth=server_args.speculative_suffix_cache_max_depth,
            max_cached_requests=server_args.speculative_suffix_cache_max_requests,
            max_spec_factor=server_args.speculative_suffix_max_spec_factor,
            max_spec_offset=server_args.speculative_suffix_max_spec_offset,
            min_token_prob=server_args.speculative_suffix_min_token_prob,
            max_spec_tokens=server_args.speculative_suffix_max_spec_tokens,
        )

    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_req_ids: List[str] = []
        batch_prompts: List[List[int]] = []
        batch_tokens: List[List[int]] = []

        self.ngram_cache.synchronize()
        for req in batch.reqs:
            batch_req_ids.append(req.rid)
            batch_prompts.append(list(req.origin_input_ids))
            batch_tokens.append(req.origin_input_ids + req.output_ids)

        req_drafts, mask = self.ngram_cache.batch_get(
            batch_req_ids,
            batch_prompts,
            batch_tokens,
        )
        total_draft_token_num = len(req_drafts)
        bs = batch.batch_size()
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _update_ngram_cache(self, batch: ScheduleBatch):
        # SuffixCacheAdapter synchronizes accepted-token deltas in batch_get,
        # immediately before the next speculation.
        pass
