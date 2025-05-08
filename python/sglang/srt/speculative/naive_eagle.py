import logging
import os
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
from huggingface_hub import snapshot_download

from sglang.srt.distributed import GroupCoordinator, patch_tensor_parallel_group
from sglang.srt.layers.dp_attention import disable_dp_size
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.managers.schedule_batch import (
    ScheduleBatch,
    get_last_loc,
    global_server_args_dict,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
    assign_draft_cache_locs,
    select_top_k_tokens,
    assign_req_to_token_pool,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import empty_context, fast_topk, get_available_gpu_memory, is_cuda, next_power_of_2

if is_cuda():
    from sgl_kernel import segment_packbits

logger = logging.getLogger(__name__)


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    with disable_dp_size(), patch_tensor_parallel_group(tp_group):
        yield
        
import triton
import triton.language as tl

@triton.jit
def create_extend_spec_info(
    seq_len,
    accept_len,
    accept_len_cum,
    positions,
    accept_len_upper: tl.constexpr, # 1
):
    pid = tl.program_id(axis=0)
    offset = 0 if pid == 0 else tl.load(accept_len_cum + pid - 1)
    seq_length = tl.load(seq_len + pid)
    accept_length = tl.load(accept_len + pid) # 1
    positions_ptr = positions + offset
    data = tl.arange(0, accept_len_upper)
    mask = data < accept_length
    tl.store(positions_ptr + data, seq_length - accept_length + data, mask)
class NaiveEagleWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.padded_static_len = self.speculative_num_steps + 1
        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override context length with target model's context length
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        with empty_context():
            super().__init__(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                server_args=server_args,
                nccl_port=nccl_port,
                dp_rank=dp_rank,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # EAGLE3 models don't share lm_head
            self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            self.hot_token_id = self.draft_model_runner.model.get_hot_token_id().to(
                embed.device
            )
        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(self.draft_model_runner.tp_group):
            self.init_cuda_graphs()
            
        self.exit_cnt = 1
        self.run_cnt = 0
        
    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        # Capture draft
        tic = time.time()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.cuda_graph_runner = EAGLEDraftCudaGraphRunner(self)
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture draft cuda graph end. Time elapsed: {time.time() - tic:.2f} s. avail mem={after_mem:.2f} GB. mem usage={(before_mem - after_mem):.2f} GB."
        )

        # Capture extend
        if self.draft_extend_attn_backend:
            raise NotImplementedError()

    @property
    def draft_model_runner(self):
        return self.model_runner

    def forward_batch_speculative_generation(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, List[int], int, int]:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepeted,
            the batch id (used for overlap schedule), and number of accepeted tokens.
        """
        if batch.forward_mode.is_decode():
            return self.draft(batch)
        elif batch.forward_mode.is_idle():
            model_worker_batch = batch.get_model_worker_batch()
            logits_output, next_token_ids = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            return logits_output, next_token_ids, model_worker_batch.bid, 0
        else:
            logits_output, next_token_ids, bid = self.forward_target_extend(batch)
            with self.draft_tp_context(self.draft_model_runner.tp_group):
                self.forward_draft_extend(
                    batch, logits_output.hidden_states, next_token_ids
                )
            return logits_output, next_token_ids, bid, 0

    def check_kv_cache(self, msg):
        logger.info(f"######################[check_kv_cache:{msg}]######################")
        logger.info(f'{self.req_to_token_pool.req_to_token.shape=}')
        # logger.info(f'{self.token_to_kv_pool_allocator.get_kvcache().k_buffer[0].shape=}')
        logger.info(f'{self.req_to_token_pool.req_to_token[:2, :50].tolist()=}')
        # k_buffer_sum = [k.sum().item() for k in self.token_to_kv_pool_allocator.get_kvcache().k_buffer[0]]
        # logger.info(f'{k_buffer_sum=}')
        logger.info(f"#############################################################")
    
    
    def check_exit(self, exit_cnt=1):
        if self.run_cnt == exit_cnt:
            exit(-1)
    
    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, List[int], int]:
        # logger.info(f"check kv: {self.req_to_token_pool=}, {self.token_to_kv_pool_allocator=},{self.token_to_kv_pool_allocator.get_kvcache().get_key_buffer(0)=}")
        # logger.info(f"check kv: {self.target_worker.get_memory_pool()=},{self.target_worker.get_memory_pool()[1].get_kvcache().get_key_buffer(0)=}")
        
        
        # self.check_kv_cache("forward_target_extend start")
        # logger.info(f"[forward_target_extend]{batch.input_ids=}")
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
            bid: The model batch ID. Used for overlap schedule.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        model_worker_batch = batch.get_model_worker_batch()
        # logger.info(f"[forward_target_extend]{model_worker_batch=}")
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, next_token_ids = self.target_worker.forward_batch_generation(
            model_worker_batch
        )
        # logger.info(f"[forward_target_extend]{logits_output=}")
        # print(f"[forward_target_extend]{logits_output.hidden_states=}")
        # print(f"[forward_target_extend]{logits_output.hidden_states.shape=}")
        # self.check_kv_cache("forward_target_extend end")
        return logits_output, next_token_ids, model_worker_batch.bid

    def draft(self, batch: ScheduleBatch):
        self.run_cnt += 1
        # logger.info(f"[before draft]turns={self.run_cnt},\n{batch=}")
        # logger.info(f"[draft batch all]{batch=}")
        # logger.info(f"[draft batch]{batch.input_ids=}, {batch.spec_info.topk_index=}")
        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        
        # logger.info(f"[before draft]{batch.input_ids=}")
        # logger.info(f"check kv: {self.req_to_token_pool=}, {self.token_to_kv_pool_allocator=},{self.token_to_kv_pool_allocator.get_kvcache().get_key_buffer(0)=}")
        # logger.info(f"check kv: {self.target_worker.get_memory_pool()=},{self.target_worker.get_memory_pool()[1].get_kvcache().get_key_buffer(0)=}")
        if self.page_size == 1:
            batch.out_cache_loc = batch.alloc_token_slots(
                num_seqs * 2, backup_state=False # hard code, 1 for target preill, 1 for draft
            )
            # logger.info(f"[draft batch]{batch.out_cache_loc=},{batch.seq_lens=}")
            end_offset = batch.seq_lens + 2 # assign 2 tokens
            # self.check_kv_cache("[before assign_req_to_token_pool]")
            assign_req_to_token_pool[(num_seqs,)](
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                batch.seq_lens,
                end_offset,
                batch.out_cache_loc,
                batch.req_to_token_pool.req_to_token.shape[1],
                next_power_of_2(num_seqs),
            )
            # self.check_kv_cache("[after assign_req_to_token_pool]")
        else:
            raise NotImplementedError("josephyou: Page size > 1 not supported yet")
        
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        # logger.info(f"[draft's batch]{batch=}")
        # for 1 req
        # batch.input_ids = torch.cat((batch.input_ids, spec_info.topk_index[0]))
        # for multi reqs:
        if spec_info.accept_length is None:
            batch.input_ids = batch.output_ids # first decode.
        else:
            input_idx = spec_info.accept_length.cumsum(dim=-1) - 1
            batch.input_ids = batch.input_ids[input_idx]
            
        batch.input_ids = torch.stack((batch.input_ids, spec_info.topk_index.squeeze(1)), dim=1).reshape(-1)
        # logger.info(f"self.draft_token={batch.input_ids}")
        positions = torch.stack([batch.seq_lens,  batch.seq_lens + 1], dim=1).reshape(-1)
        # logger.info(f"[draft positions process]{positions=}")
        
        # NOTE: In naive speculative algorithm, we do not to set tree mask, insetead, we set casual mask is True directly.
        
        # tree_mask = torch.full((batch.seq_lens_sum * 2 + 2  * 2 * num_seqs,),
        #                     True,
        #                     device=self.device,)
        # 4 * idx + 2 * seq_lens[idx - 1] * idx + seq_lens[idx] + 1
        # if idx = 0, then 4 is zero
        # if idx = 1, then 4 * 1 + 2 * 3 * 1 + 4 + 1 = 4 + 6 + 4 + 1 = 15
        # tree_mask[batch.seq_lens_sum + 1] = False
        
        batch.spec_info = EagleVerifyInput(
            draft_token=batch.input_ids,
            custom_mask=None, 
            positions=positions,
            retrive_index=None,
            retrive_next_token=None,
            retrive_next_sibling=None,
            retrive_cum_len=None,
            draft_token_num=2,
            spec_steps=1,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        # logger.info(f"[after draft, before verify],{batch.input_ids=},{batch=}")
        
        model_worker_batch = batch.get_model_worker_batch()
        # logger.info(f'[before cat]{model_worker_batch.input_ids=}, {spec_info.topk_index[0]=}')
        # model_worker_batch.input_ids = torch.cat((model_worker_batch.input_ids, spec_info.topk_index[0])) #TODO: Adapt for multi requests.
        
        
        # logger.info(f'[target forward batch]{model_worker_batch=}')
        logits_output, next_token_ids = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=False
        )
        batch.spec_info.hidden_states = logits_output.hidden_states
        
        next_token_ids_cpu = next_token_ids.tolist()
        # logger.info(f"[verify]{logits_output=}, {next_token_ids=}")
        accept_index = torch.full((num_seqs, 2), -1, dtype=torch.int32, device="cuda")
        accept_length = torch.zeros((num_seqs,), dtype=torch.int32, device="cuda")
        
        # verify
        for i in range(num_seqs):
            accept_index[i, 0] = i * 2 # at least accept first token
            
            draft_token = batch.input_ids[2 * i + 1]
            target_token = next_token_ids[2 * i]
            if draft_token == target_token:
                accept_index[i, 1] = 2 * i + 1
                accept_length[i] += 1
                
        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        new_accept_index = []
        unfinished_index = []
        has_finished = False
        accept_index_cpu = accept_index.tolist()     

        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            new_accept_index_ = []
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                id = next_token_ids_cpu[idx]
                # if not found_finished:
                req.output_ids.append(id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    accept_index[i, j + 1 :] = -1
                    break
                else:
                    new_accept_index_.append(idx)
            if not req.finished():
                new_accept_index.extend(new_accept_index_)
                unfinished_index.append(i)
            req.spec_verify_ct += 1

        if has_finished:
            accept_length = (accept_index != -1).sum(dim=1) - 1
        
        # verify done, prepare for free
        # logger.info(f'[verify-accept_index]={accept_index}')
        accept_index = accept_index[accept_index != -1]
        evict_mask = torch.full((num_seqs * 2,), True, dtype=torch.bool)
        evict_mask[accept_index] = False
        # logger.info(f'[verify-evict_mask]={evict_mask}')
        if self.page_size != 1:
            # TODO: align_evict_mask_to_page_size, see eagle_utils.py/align_evict_mask_to_page_size 
            pass
        
        
        
        self.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
        
        verified_id = next_token_ids[accept_index]
        # logger.info(f'[verify-verified_id]={verified_id},{has_finished=},{new_accept_index=}')
        
        if not has_finished:
            batch.out_cache_loc = batch.out_cache_loc[accept_index]
            assign_req_to_token_pool[(num_seqs,)](
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc,
                    batch.req_to_token_pool.req_to_token.shape[1],
                    next_power_of_2(num_seqs),
                )
            batch.seq_lens.add_(accept_length + 1)
            accept_length_cpu = accept_length.tolist()
            
            draft_input = EagleDraftInput()
            draft_input.hidden_states = batch.spec_info.hidden_states[accept_index]
            draft_input.accept_length = accept_length
            draft_input.accept_length_cpu = accept_length_cpu
            draft_input.verified_id = verified_id
            draft_input.seq_lens_for_draft_extend = batch.seq_lens
            draft_input.req_pool_indices_for_draft_extend = batch.req_pool_indices
        else:
            assign_req_to_token_pool[(num_seqs,)](
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                batch.seq_lens,
                batch.seq_lens + accept_length + 1,
                batch.out_cache_loc[accept_index],
                batch.req_to_token_pool.req_to_token.shape[1],
                next_power_of_2(num_seqs),
            )
            batch.seq_lens.add_(accept_length + 1)
            accept_length_cpu = accept_length.tolist()
            draft_input = EagleDraftInput()
            if len(new_accept_index) > 0:
                new_accept_index = torch.tensor(new_accept_index, device="cuda")
                unfinished_index_device = torch.tensor(unfinished_index, device="cuda")
                # logger.info(f"new accept index: {new_accept_index}, unfinished index: {unfinished_index_device}")
                draft_input.hidden_states = batch.spec_info.hidden_states[new_accept_index]
                draft_input.verified_id = next_token_ids[new_accept_index]
                draft_input.accept_length_cpu = [
                    accept_length_cpu[i] for i in unfinished_index
                ]
                draft_input.accept_length = accept_length[unfinished_index_device]
                if has_finished:
                    draft_input.seq_lens_for_draft_extend = batch.seq_lens[
                        unfinished_index_device
                    ]
                    draft_input.req_pool_indices_for_draft_extend = (
                        batch.req_pool_indices[unfinished_index_device]
                    )
            batch.out_cache_loc = batch.out_cache_loc[new_accept_index]

        logits_output.next_token_logits = logits_output.next_token_logits[accept_index]
        logits_output.hidden_states = logits_output.hidden_states[accept_index]
        batch.spec_info = draft_input
        
        
        # All reqs done here.
        if batch.spec_info.verified_id is None:
            # logger.info(f"[return:!!]{verified_id=},{sum(accept_length_cpu)=},{model_worker_batch.bid=},{logits_output=}")
            return (
                logits_output,
                verified_id,
                model_worker_batch.bid,
                sum(accept_length_cpu),
            )

        
        # Backup fileds that will be modified in-place
        seq_lens_backup = batch.seq_lens.clone()
        req_pool_indices_backup = batch.req_pool_indices
        accept_length_backup = batch.spec_info.accept_length
        return_logprob_backup = batch.return_logprob
        
        # Prepare metadata
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        batch.spec_info.prepare_extend_after_decode(
            batch,
            1,
        )
        # logger.info(f"[verify-prepare_extend_after_decode]{batch.spec_info=}")
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        batch.return_logprob = False
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        # print(f"[forward_draft_extend_after_decode]{forward_batch=}")
        logits_output = self.draft_model_runner.forward(forward_batch)

        self._detect_nan_if_needed(logits_output)
        self.capture_for_decode(logits_output, forward_batch.spec_info)
        # logger.info(f"[draft decode done!]{logits_output=}, {forward_batch=}")

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = ForwardMode.DECODE
        batch.seq_lens = seq_lens_backup
        batch.req_pool_indices = req_pool_indices_backup
        batch.spec_info.accept_length = accept_length_backup
        batch.return_logprob = return_logprob_backup
        
        # return output ids for next decode.
        # offsets = torch.arange(num_seqs, dtype=torch.int32, device="cuda") * 2 #0, 2, 4, 6, 8
        # output_indices = offsets + (batch.spec_info.accept_length - 1)
        # logger.info(f"[return:!!]{batch.spec_info.accept_length=},{verified_id=},{next_token_ids=},{sum(accept_length_cpu)=},{model_worker_batch.bid=},{logits_output=}")
        return (
                logits_output,
                verified_id,
                model_worker_batch.bid,
                sum(accept_length_cpu),
            )

    def add_logprob_values(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
    ):
        # Extract args
        logits_output = res.logits_output
        top_logprobs_nums = batch.top_logprobs_nums
        token_ids_logprobs = batch.token_ids_logprobs
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits, dim=-1
        )
        batch_next_token_ids = res.verified_id
        num_tokens_per_req = [accept + 1 for accept in res.accept_length_per_req_cpu]

        # We should repeat top_logprobs_nums to match num_tokens_per_req.
        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)

        # Extract logprobs
        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums_repeat_interleaved)

        if any(x is not None for x in token_ids_logprobs):
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(logprobs, token_ids_logprobs_repeat_interleaved)

        logits_output.next_token_logprobs = logprobs[
            torch.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
            batch_next_token_ids,
        ]

        # Add output logprobs to the request
        pt = 0
        next_token_logprobs = logits_output.next_token_logprobs.tolist()
        verified_ids = batch_next_token_ids.tolist()
        for req, num_tokens in zip(batch.reqs, num_tokens_per_req):
            for _ in range(num_tokens):
                if req.return_logprob:
                    req.output_token_logprobs_val.append(next_token_logprobs[pt])
                    req.output_token_logprobs_idx.append(verified_ids[pt])
                    if req.top_logprobs_num > 0:
                        req.output_top_logprobs_val.append(
                            res.logits_output.next_token_top_logprobs_val[pt]
                        )
                        req.output_top_logprobs_idx.append(
                            res.logits_output.next_token_top_logprobs_idx[pt]
                        )
                pt += 1

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: List[int],
    ):
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # logger.info(f"check kv: {self.req_to_token_pool=}, {self.token_to_kv_pool_allocator=},{self.token_to_kv_pool_allocator.get_kvcache().get_key_buffer(0)=}")
        # logger.info(f"check kv: {self.target_worker.get_memory_pool()=},{self.target_worker.get_memory_pool()[1].get_kvcache().get_key_buffer(0)=}")
        # self.check_kv_cache("forward_draft_extend before")
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids,
        )
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.return_logprob = False
        # logger.info(f"[forward_draft_extend batch]{forward_batch=}")
        logits_output = self.draft_model_runner.forward(forward_batch)
        self._detect_nan_if_needed(logits_output)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        # print(f'{logits_output.hidden_states.shape=}')
        self.capture_for_decode(logits_output, forward_batch.spec_info)
        # self.check_kv_cache("forward_draft_extend after")

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        
        # logger.info(f"[capture_for_decode]{draft_input.topk_index}")
        draft_input.hidden_states = logits_output.hidden_states

    def _detect_nan_if_needed(self, logits_output: LogitsProcessorOutput):
        if self.enable_nan_detection:
            logits = logits_output.next_token_logits
            if torch.any(torch.isnan(logits)):
                logger.error("Detected errors during sampling! NaN in the logits.")
                raise ValueError("Detected errors during sampling! NaN in the logits.")


def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        cache_dir = snapshot_download(
            os.path.dirname(token_map_path),
            ignore_patterns=["*.bin", "*.safetensors"],
        )
        token_map_path = os.path.join(cache_dir, os.path.basename(token_map_path))
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int32)
