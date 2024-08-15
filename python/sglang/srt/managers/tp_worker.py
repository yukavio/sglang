"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""A tensor parallel worker."""

import logging
import multiprocessing
import os
import pickle
import time
import warnings
from multiprocessing import shared_memory
from typing import Any, List, Optional, Union

import torch
import torch.distributed
import torch.distributed as dist

from sglang.global_config import global_config
from sglang.srt.constrained.fsm_cache import FSMCache
from sglang.srt.constrained.jump_forward import JumpForwardCache
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.layers.logits_processor import LogitProcessorOutput
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchTokenIDOut,
    ControllerInfo,
    FlushCacheReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.policy_scheduler import PolicyScheduler, PrefillAdder
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    BaseFinishReason,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_config import ModelConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    get_int_token_logit_bias,
    is_multimodal_model,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


# TODO: Rename "CI" to "SGLANG_IS_IN_CI".
crash_on_warning = os.getenv("CI", "false") == "true"


class ModelTpServer:
    def __init__(
        self,
        gpu_id: int,
        tp_rank: int,
        server_args: ServerArgs,
        nccl_port: int,
        model_overide_args: dict,
        controller_info: ControllerInfo,
        dp_worker_id: int,
    ):
        suppress_other_loggers()

        # Copy arguments
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_worker_id
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.schedule_policy = server_args.schedule_policy
        self.disable_regex_jump_forward = server_args.disable_regex_jump_forward

        # Chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        self.current_inflight_req = None

        # Init model and tokenizer
        self.model_config = ModelConfig(
            server_args.model_path,
            server_args.trust_remote_code,
            context_length=server_args.context_length,
            model_overide_args=model_overide_args,
        )
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )

        # Flex DP inference
        if controller_info:
            self.controller_info = controller_info
            shm = shared_memory.SharedMemory(self.controller_info.cpu_kv_cache)
            self.swap_cache = torch.frombuffer(
                buffer=shm.buf, dtype=self.model_runner.dtype
            ).reshape(self.controller_info.cache_shape)
        else:
            self.controller_info = None

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if is_multimodal_model(server_args.model_path):
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
            ),
            self.model_runner.req_to_token_pool.size - 1,
        )
        self.int_token_logit_bias = torch.tensor(
            get_int_token_logit_bias(self.tokenizer, self.model_config.vocab_size)
        )
        self.max_req_input_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        set_random_seed(server_args.random_seed)

        # Print info
        logger.info(
            f"[gpu={self.gpu_id}] "
            f"max_total_num_tokens={self.max_total_num_tokens}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"context_len={self.model_config.context_len}"
        )

        # Init cache
        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
            )
        else:
            self.tree_cache = RadixCache(
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                disable=server_args.disable_radix_cache,
            )
        self.tree_cache_metrics = {"total": 0, "hit": 0}
        self.scheduler = PolicyScheduler(self.schedule_policy, self.tree_cache)
        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.token_to_kv_pool = self.model_runner.token_to_kv_pool

        # Init running status
        self.waiting_queue: List[Req] = []
        self.running_batch: ScheduleBatch = None
        self.out_pyobjs = []
        self.decode_forward_ct = 0
        self.stream_interval = server_args.stream_interval
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()

        # Init the FSM cache for constrained generation
        if not server_args.skip_tokenizer_init:
            self.regex_fsm_cache = FSMCache(
                server_args.tokenizer_path,
                {
                    "tokenizer_mode": server_args.tokenizer_mode,
                    "trust_remote_code": server_args.trust_remote_code,
                },
                skip_tokenizer_init=server_args.skip_tokenizer_init,
            )
        self.jump_forward_cache = JumpForwardCache()

        # Init new token estimation
        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"
        self.min_new_token_ratio = min(
            global_config.base_min_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.new_token_ratio = self.min_new_token_ratio
        self.new_token_ratio_decay = global_config.new_token_ratio_decay

    def exposed_step(self, recv_reqs: List):
        try:
            # Recv requests
            for recv_req in recv_reqs:
                if isinstance(
                    recv_req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                ):
                    self.handle_generate_request(recv_req)
                elif isinstance(recv_req, FlushCacheReq):
                    self.flush_cache()
                elif isinstance(recv_req, AbortReq):
                    self.abort_request(recv_req)
                else:
                    raise ValueError(f"Invalid request: {recv_req}")

            # Forward
            self.forward_step()
        except Exception:
            logger.error("Exception in ModelTpServer:\n" + get_exception_traceback())
            raise

        # Return results
        ret = self.out_pyobjs
        self.out_pyobjs = []
        return ret

    @torch.inference_mode()
    def forward_step(self):
        new_batch = self.get_new_prefill_batch()

        if new_batch is not None:
            # Run a new prefill batch
            self.forward_prefill_batch(new_batch)

            if not new_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge(new_batch)
        else:
            # Run a decode batch
            if self.running_batch is not None:
                # Run a few decode batches continuously for reducing overhead
                for _ in range(global_config.num_continue_decode_steps):
                    self.num_generated_tokens += len(self.running_batch.reqs)
                    self.forward_decode_batch(self.running_batch)

                    # Print stats
                    if self.tp_rank == 0 and self.decode_forward_ct % 40 == 0:
                        self.print_decode_stats()

                    if self.running_batch.is_empty():
                        self.running_batch = None
                        break

                    if self.out_pyobjs and self.running_batch.has_stream():
                        break
            else:
                self.check_memory()
                self.new_token_ratio = global_config.init_new_token_ratio

    def print_decode_stats(self):
        num_used = self.max_total_num_tokens - (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        throughput = self.num_generated_tokens / (time.time() - self.last_stats_tic)
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()
        logger.info(
            f"[gpu={self.gpu_id}] Decode batch. "
            f"#running-req: {len(self.running_batch.reqs)}, "
            f"#token: {num_used}, "
            f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
            f"gen throughput (token/s): {throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}"
        )

    def check_memory(self):
        available_size = (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        if available_size != self.max_total_num_tokens:
            warnings.warn(
                "Warning: "
                f"available_size={available_size}, max_total_num_tokens={self.max_total_num_tokens}\n"
                "KV cache pool leak detected!"
            )
            exit(1) if crash_on_warning else None

        if len(self.req_to_token_pool.free_slots) != self.req_to_token_pool.size:
            warnings.warn(
                "Warning: "
                f"available req slots={len(self.req_to_token_pool.free_slots)}, "
                f"total slots={self.req_to_token_pool.size}\n"
                "Memory pool leak detected!"
            )
            exit(1) if crash_on_warning else None

    def handle_generate_request(
        self,
        recv_req: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
    ):
        req = Req(recv_req.rid, recv_req.input_text, recv_req.input_ids)
        req.tokenizer = self.tokenizer
        req.sampling_params = recv_req.sampling_params
        if self.model_runner.is_generation:
            req.pixel_values = recv_req.pixel_values
            if req.pixel_values is not None:
                req.pad_value = [
                    (recv_req.image_hash) % self.model_config.vocab_size,
                    (recv_req.image_hash >> 16) % self.model_config.vocab_size,
                    (recv_req.image_hash >> 32) % self.model_config.vocab_size,
                    (recv_req.image_hash >> 64) % self.model_config.vocab_size,
                ]
                req.image_size = recv_req.image_size
                (
                    req.origin_input_ids,
                    req.image_offset,
                ) = self.model_runner.model.pad_input_ids(
                    req.origin_input_ids_unpadded,
                    req.pad_value,
                    req.pixel_values.shape,
                    req.image_size,
                )
            req.return_logprob = recv_req.return_logprob
            req.logprob_start_len = recv_req.logprob_start_len
            req.top_logprobs_num = recv_req.top_logprobs_num
            req.stream = recv_req.stream

            # Init regex fsm
            if req.sampling_params.regex is not None:
                req.regex_fsm = self.regex_fsm_cache.query(req.sampling_params.regex)
                if not self.disable_regex_jump_forward:
                    req.jump_forward_map = self.jump_forward_cache.query(
                        req.sampling_params.regex
                    )

        # Truncate prompts that are too long
        if len(req.origin_input_ids) >= self.max_req_input_len:
            logger.warn(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated!!!"
            )
            req.origin_input_ids = req.origin_input_ids[: self.max_req_input_len]

        if self.model_runner.is_generation:
            req.sampling_params.max_new_tokens = min(
                (
                    req.sampling_params.max_new_tokens
                    if req.sampling_params.max_new_tokens is not None
                    else 1 << 30
                ),
                self.max_req_input_len - 1 - len(req.origin_input_ids),
            )
        if self.controller_info:
            self.controller_info.available_kv_cache[self.dp_rank] = (
                self.token_to_kv_pool.available_size()
            )
            self.controller_info.current_bs[self.dp_rank].value += len(
                req.origin_input_ids
            )

        self.waiting_queue.append(req)

    def get_new_prefill_batch(self) -> Optional[ScheduleBatch]:
        running_bs = (
            len(self.running_batch.reqs) if self.running_batch is not None else 0
        )
        if running_bs >= self.max_running_requests:
            return None

        # Get priority queue
        prefix_computed = self.scheduler.calc_priority(self.waiting_queue)

        adder = PrefillAdder(
            self.tree_cache,
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size(),
            self.max_prefill_tokens,
            self.chunked_prefill_size,
        )

        if self.running_batch is not None:
            adder.remove_running_tokens(self.running_batch, self.new_token_ratio)

        has_inflight = self.current_inflight_req is not None
        if self.current_inflight_req is not None:
            self.current_inflight_req.init_next_round_input(
                None if prefix_computed else self.tree_cache
            )
            self.current_inflight_req = adder.add_inflight_req(
                self.current_inflight_req
            )

        for req in self.waiting_queue:
            req.init_next_round_input(None if prefix_computed else self.tree_cache)
            res = adder.add_one_req(req)
            if (
                not res
                or adder.no_remaining_tokens()
                or running_bs + len(adder.can_run_list) >= self.max_running_requests
            ):
                break

        can_run_list = adder.can_run_list

        if adder.new_inflight_req is not None:
            assert self.current_inflight_req is None
            self.current_inflight_req = adder.new_inflight_req

        if len(can_run_list) == 0:
            return None

        # Print stats
        if self.tp_rank == 0:
            self.tree_cache_metrics["total"] += (
                adder.log_input_tokens + adder.log_hit_tokens
            ) / 10**9
            self.tree_cache_metrics["hit"] += (adder.log_hit_tokens) / 10**9
            tree_cache_hit_rate = (
                self.tree_cache_metrics["hit"] / self.tree_cache_metrics["total"]
            )
            logger.info(
                f"[gpu={self.gpu_id}] Prefill batch. "
                f"#new-seq: {len(can_run_list)}, "
                f"#new-token: {adder.log_input_tokens}, "
                f"#cached-token: {adder.log_hit_tokens}, "
                f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
                f"#running-req: {running_bs}, "
                f"#queue-req: {len(self.waiting_queue) - len(can_run_list) + has_inflight}"
            )

        # Return the new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
        )
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_list]
        return new_batch

    def forward_prefill_batch(self, batch: ScheduleBatch):
        # Build batch tensors
        batch.prepare_for_extend(
            self.model_config.vocab_size, self.int_token_logit_bias
        )

        if self.controller_info:
            self.controller_info.available_kv_cache[self.dp_rank] = (
                self.token_to_kv_pool.available_size()
            )
            self.controller_info.current_bs[
                self.dp_rank
            ].value -= batch.input_ids.numel()

        if self.model_runner.is_generation:
            # Forward and sample the next tokens
            if batch.extend_num_tokens != 0:
                output = self.model_runner.forward(batch, ForwardMode.EXTEND)
                next_token_ids = batch.sample(output.next_token_logits)

                # Move logprobs to cpu
                if output.next_token_logprobs is not None:
                    output.next_token_logprobs = output.next_token_logprobs[
                        torch.arange(len(next_token_ids), device=next_token_ids.device),
                        next_token_ids,
                    ].tolist()
                    output.input_token_logprobs = output.input_token_logprobs.tolist()
                    output.normalized_prompt_logprobs = (
                        output.normalized_prompt_logprobs.tolist()
                    )

                next_token_ids = next_token_ids.tolist()
            else:
                if self.tokenizer is None:
                    next_token_ids = []
                    for req in batch.reqs:
                        next_token_ids.append(
                            next(iter(req.sampling_params.stop_token_ids))
                        )
                else:
                    next_token_ids = [self.tokenizer.eos_token_id] * len(batch.reqs)

            # Check finish conditions
            pt = 0
            for i, req in enumerate(batch.reqs):
                if req is not self.current_inflight_req:
                    # Inflight reqs' prefill is not finished
                    req.completion_tokens_wo_jump_forward += 1
                    req.output_ids.append(next_token_ids[i])
                    req.check_finished()

                if req.finished():
                    self.tree_cache.cache_finished_req(req)
                else:
                    self.tree_cache.cache_unfinished_req(req)

                if req is self.current_inflight_req:
                    # Inflight request would get a new req idx
                    self.req_to_token_pool.free(req.req_pool_idx)

                if req.return_logprob:
                    self.add_logprob_return_values(i, req, pt, next_token_ids, output)
                    pt += req.extend_input_len
        else:
            assert batch.extend_num_tokens != 0
            output = self.model_runner.forward(batch, ForwardMode.EXTEND)
            embeddings = output.embeddings.tolist()

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                req.embedding = embeddings[i]
                if req is not self.current_inflight_req:
                    # Inflight reqs' prefill is not finished
                    # dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                if req.finished():
                    self.tree_cache.cache_finished_req(req)
                else:
                    self.tree_cache.cache_unfinished_req(req)

                if req is self.current_inflight_req:
                    # Inflight request would get a new req idx
                    self.req_to_token_pool.free(req.req_pool_idx)

        self.handle_finished_requests(batch)

    def add_logprob_return_values(
        self,
        i,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        output: LogitProcessorOutput,
    ):
        if req.normalized_prompt_logprob is None:
            req.normalized_prompt_logprob = output.normalized_prompt_logprobs[i]

        if req.input_token_logprobs is None:
            # If logprob_start_len > 0, then first logprob_start_len prompt tokens will be ignored.
            req.input_token_logprobs = list(
                zip(
                    output.input_token_logprobs[pt : pt + req.extend_input_len - 1],
                    req.fill_ids[-req.extend_input_len + 1 :],
                )
            )
            if req.logprob_start_len == 0:
                req.input_token_logprobs = [
                    (None, req.fill_ids[0])
                ] + req.input_token_logprobs

        if req.last_update_decode_tokens != 0:
            req.output_token_logprobs.extend(
                list(
                    zip(
                        output.input_token_logprobs[
                            pt
                            + req.extend_input_len
                            - req.last_update_decode_tokens : pt
                            + req.extend_input_len
                            - 1
                        ],
                        req.fill_ids[-req.last_update_decode_tokens + 1 :],
                    )
                )
            )

        req.output_token_logprobs.append(
            (output.next_token_logprobs[i], next_token_ids[i])
        )

        if req.top_logprobs_num > 0:
            if req.input_top_logprobs is None:
                req.input_top_logprobs = output.input_top_logprobs[i]
                if req.logprob_start_len == 0:
                    req.input_top_logprobs = [None] + req.input_top_logprobs

            if req.last_update_decode_tokens != 0:
                req.output_top_logprobs.extend(
                    output.input_top_logprobs[i][-req.last_update_decode_tokens + 1 :]
                )
            req.output_top_logprobs.append(output.output_top_logprobs[i])

    def forward_decode_batch(self, batch: ScheduleBatch):
        # Check if decode out of memory
        if not batch.check_decode_mem():
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode()
            self.new_token_ratio = new_token_ratio

            logger.info(
                "decode out of memory happened, "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )
            self.waiting_queue.extend(retracted_reqs)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if not self.disable_regex_jump_forward:
            # Check for jump-forward
            jump_forward_reqs = batch.check_for_jump_forward(self.model_runner)
            self.waiting_queue.extend(jump_forward_reqs)
            if batch.is_empty():
                return

        # Update batch tensors
        self.decode_forward_ct = (self.decode_forward_ct + 1) % (1 << 30)
        batch.prepare_for_decode()

        # Forward and sample the next tokens
        output = self.model_runner.forward(batch, ForwardMode.DECODE)
        next_token_ids = batch.sample(output.next_token_logits)

        # Move logprobs to cpu
        if output.next_token_logprobs is not None:
            next_token_logprobs = output.next_token_logprobs[
                torch.arange(len(next_token_ids), device=next_token_ids.device),
                next_token_ids,
            ].tolist()

        next_token_ids = next_token_ids.tolist()

        # Check finish condition
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            req.completion_tokens_wo_jump_forward += 1
            req.output_ids.append(next_token_id)
            req.check_finished()

            if req.finished():
                self.tree_cache.cache_finished_req(req)

            if req.return_logprob:
                req.output_token_logprobs.append(
                    (next_token_logprobs[i], next_token_id)
                )
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs.append(output.output_top_logprobs[i])

        self.handle_finished_requests(batch)

    def swap_in_decode_request(self, req: Req):
        pass

    def swap_out_decode_request(self, req: Req):
        pass

    def handle_finished_requests(self, batch: ScheduleBatch):
        output_rids = []
        output_meta_info = []
        output_finished_reason: List[BaseFinishReason] = []
        if self.model_runner.is_generation:
            output_vids = []
            decoded_texts = []
            output_read_ids = []
            output_read_offsets = []
            output_skip_special_tokens = []
            output_spaces_between_special_tokens = []
        else:  # for embedding model
            output_embeddings = []
        unfinished_indices = []

        for i, req in enumerate(batch.reqs):
            if not req.finished() and req is not self.current_inflight_req:
                unfinished_indices.append(i)

            if req.finished() or (
                (
                    req.stream
                    and (
                        self.decode_forward_ct % self.stream_interval == 0
                        or len(req.output_ids) == 1
                    )
                )
            ):
                output_rids.append(req.rid)
                output_finished_reason.append(req.finished_reason)
                if self.model_runner.is_generation:
                    output_vids.append(req.vid)
                    decoded_texts.append(req.decoded_text)
                    read_ids, read_offset = req.init_incremental_detokenize()
                    output_read_ids.append(read_ids)
                    output_read_offsets.append(read_offset)
                    output_skip_special_tokens.append(
                        req.sampling_params.skip_special_tokens
                    )
                    output_spaces_between_special_tokens.append(
                        req.sampling_params.spaces_between_special_tokens
                    )

                    meta_info = {
                        "prompt_tokens": len(req.origin_input_ids),
                        "completion_tokens": len(req.output_ids),
                        "completion_tokens_wo_jump_forward": req.completion_tokens_wo_jump_forward,
                        "finish_reason": str(req.finished_reason),
                    }
                    if req.return_logprob:
                        (
                            meta_info["input_token_logprobs"],
                            meta_info["output_token_logprobs"],
                            meta_info["input_top_logprobs"],
                            meta_info["output_top_logprobs"],
                            meta_info["normalized_prompt_logprob"],
                        ) = (
                            req.input_token_logprobs,
                            req.output_token_logprobs,
                            req.input_top_logprobs,
                            req.output_top_logprobs,
                            req.normalized_prompt_logprob,
                        )
                    output_meta_info.append(meta_info)
                else:  # for embedding model
                    output_embeddings.append(req.embedding)
                    meta_info = {
                        "prompt_tokens": len(req.origin_input_ids),
                    }
                    output_meta_info.append(meta_info)

        # Send to detokenizer
        if output_rids:
            if self.model_runner.is_generation:
                self.out_pyobjs.append(
                    BatchTokenIDOut(
                        output_rids,
                        output_vids,
                        decoded_texts,
                        output_read_ids,
                        output_read_offsets,
                        output_skip_special_tokens,
                        output_spaces_between_special_tokens,
                        output_meta_info,
                        output_finished_reason,
                    )
                )
            else:  # for embedding model
                self.out_pyobjs.append(
                    BatchEmbeddingOut(
                        output_rids,
                        output_embeddings,
                        output_meta_info,
                        output_finished_reason,
                    )
                )

        # Remove finished reqs: update batch tensors
        batch.filter_batch(unfinished_indices)

    def flush_cache(self):
        if len(self.waiting_queue) == 0 and (
            self.running_batch is None or len(self.running_batch.reqs) == 0
        ):
            self.tree_cache.reset()
            self.tree_cache_metrics = {"total": 0, "hit": 0}
            self.regex_fsm_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool.clear()
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
        else:
            warnings.warn(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {0 if self.running_batch is None else len(self.running_batch.reqs)}"
            )

    def abort_request(self, recv_req):
        # Delete requests in the waiting queue
        to_del = None
        for i, req in enumerate(self.waiting_queue):
            if req.rid == recv_req.rid:
                to_del = i
                break

        if to_del is not None:
            del self.waiting_queue[to_del]

        # Delete requests in the running batch
        if self.running_batch:
            for req in self.running_batch.reqs:
                if req.rid == recv_req.rid:
                    req.finished_reason = FINISH_ABORT()
                    break


def run_tp_server(
    gpu_id: int,
    tp_rank: int,
    server_args: ServerArgs,
    nccl_port: int,
    model_overide_args: dict,
):
    """Run a tensor parallel server."""
    try:
        model_server = ModelTpServer(
            gpu_id,
            tp_rank,
            server_args,
            nccl_port,
            model_overide_args,
        )
        tp_cpu_group = model_server.model_runner.tp_group.cpu_group

        while True:
            recv_reqs = broadcast_recv_input(None, tp_rank, tp_cpu_group)
            model_server.exposed_step(recv_reqs)
    except Exception:
        logger.error("Exception in run_tp_server:\n" + get_exception_traceback())
        raise


def launch_tp_servers(
    gpu_ids: List[int],
    tp_rank_range: List[int],
    server_args: ServerArgs,
    nccl_port: int,
    model_overide_args: dict,
):
    """Launch multiple tensor parallel servers."""
    procs = []
    for i in tp_rank_range:
        proc = multiprocessing.Process(
            target=run_tp_server,
            args=(gpu_ids[i], i, server_args, nccl_port, model_overide_args),
        )
        proc.start()
        procs.append(proc)

    return procs


def broadcast_recv_input(
    data: Any, rank: int, dist_group: torch.distributed.ProcessGroup
):
    """Broadcast inputs from rank=0 to all other ranks with torch.dist backend."""

    if rank == 0:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long)
            dist.broadcast(tensor_size, src=0, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)
            tensor_data = torch.ByteTensor(list(serialized_data))
            tensor_size = torch.tensor([size], dtype=torch.long)

            dist.broadcast(tensor_size, src=0, group=dist_group)
            dist.broadcast(tensor_data, src=0, group=dist_group)
    else:
        tensor_size = torch.tensor([0], dtype=torch.long)
        dist.broadcast(tensor_size, src=0, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8)
        dist.broadcast(tensor_data, src=0, group=dist_group)

        serialized_data = bytes(tensor_data.tolist())
        data = pickle.loads(serialized_data)
        return data
