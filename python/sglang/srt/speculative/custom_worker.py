import logging
import os
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import dynamic_import

logger = logging.getLogger(__name__)

USE_FULL_MASK = True


class CustomWorker:
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
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size

        self.max_batch_size = target_worker.max_running_requests
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self.draft_tokens_func = dynamic_import(
            server_args.speculative_draft_tokens_func
        )
        self.draft_extend_func = dynamic_import(
            server_args.speculative_draft_extend_func
        )
        self.draft_extend_after_decode_func = dynamic_import(
            server_args.speculative_draft_extend_after_decode_func
        )
        self.verify_func = dynamic_import(server_args.speculative_verify_func)

        self._init_preallocated_tensors()

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        if batch.forward_mode.is_extend():
            self.draft_extend_func(batch)
        else:
            batch.spec_algorithm = SpeculativeAlgorithm.CUSTOM
            self.draft_tokens_func(batch)
            batch.forward_mode = ForwardMode.TARGET_VERIFY

    def forward_batch_speculative_generation(self, batch: ScheduleBatch):
        self._prepare_for_speculative_decoding(batch)
        model_worker_batch = batch.get_model_worker_batch()
        bid = model_worker_batch.bid
        num_accepted_tokens = 0

        if model_worker_batch.forward_mode.is_target_verify():
            logits_output, _, can_run_cuda_graph = (
                self.target_worker.forward_batch_generation(
                    model_worker_batch, skip_sample=True
                )
            )
            logits_output, next_token_ids, num_accepted_tokens = self.verify_func(
                batch, logits_output
            )
            self.draft_extend_after_decode_func(batch)
            batch.forward_mode = ForwardMode.DECODE

        else:
            logits_output, next_token_ids, can_run_cuda_graph = (
                self.target_worker.forward_batch_generation(model_worker_batch)
            )

        return (
            logits_output,
            next_token_ids,
            bid,
            num_accepted_tokens,
            can_run_cuda_graph,
        )
