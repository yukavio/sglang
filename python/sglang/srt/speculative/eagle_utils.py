from __future__ import annotations

from typing import TYPE_CHECKING, List, Type

import torch
import triton
import triton.language as tl

from .build_eagle_tree import build_tree_kernel
from sglang.srt.model_executor.forward_batch_info import ForwardMode, ForwardBatch
from sglang.srt.speculative.speculative_utils import SpecDraftInput, SpecVerifyInput, DraftInfoFactory

if TYPE_CHECKING:
    from python.sglang.srt.layers.sampler import SampleOutput
    from python.sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs


# Copy from sglang.srt.layers.flashinfer_utils.create_flashinfer_kv_indices_triton due to import error
@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    max_context_len: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    req_to_token_ptr += req_pool_index * max_context_len
    kv_indices_ptr += kv_indices_offset

    ld_offset = kv_start + tl.arange(0, BLOCK_SIZE)
    st_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = ld_offset < kv_end
        data = tl.load(req_to_token_ptr + ld_offset, mask=mask)
        tl.store(kv_indices_ptr + st_offset, data, mask=mask)
        ld_offset += BLOCK_SIZE
        st_offset += BLOCK_SIZE


@triton.jit
def eagle_verify_retrive(
    retrive_index,
    accept_mask,
    retrive_cum_len,
    accept_index,
    accept_length,
    extract_index,
    max_len: tl.constexpr,
    draft_token_num: tl.constexpr,
    max_len_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    retrive_end = tl.load(retrive_cum_len + pid + 1)
    retrive_start = tl.load(retrive_cum_len + pid)
    retrive_len = retrive_end - retrive_start
    accept_ptr = accept_mask + retrive_start
    accept_offset = tl.arange(0, draft_token_num)
    accept_load_mask = accept_offset < retrive_len
    accept_len_list = tl.load(
        accept_ptr + accept_offset, mask=accept_load_mask, other=-1
    )

    accept_len = tl.max(accept_len_list)
    max_index = tl.argmax(accept_len_list, axis=0, tie_break_left=True)
    # triton is not support argmax with tie_break_right, so I need implement it by some way
    mask_max = accept_len_list == accept_len
    
    count_mask = tl.full(shape=[draft_token_num], value=0, dtype=tl.int32)
    count = tl.sum(tl.where(mask_max, 1, count_mask))
    if count>1:
        index = tl.arange(0, draft_token_num)
        mask_left = index != max_index
        remained_index = tl.where(mask_max and mask_left, index, 0)
        max_index = tl.max(remained_index)

    tl.store(accept_length + pid, accept_len)
    retrive_index_ptr = retrive_index + (retrive_start + max_index) * max_len
    retrive_offset = tl.arange(0, max_len_upper)
    retrive_load_mask = retrive_offset < accept_len + 1
    data = tl.load(retrive_index_ptr + retrive_offset, mask=retrive_load_mask)
    
    tl.store(
        accept_index + pid * max_len + retrive_offset, data, mask=retrive_load_mask
    )

    extract_load_ptr = accept_index + pid * max_len + accept_len
    if accept_len == max_len - 1:
        extract_data = tl.load(extract_load_ptr - 1)
        tl.store(extract_index + pid * 2, extract_data)
        extract_data = tl.load(extract_load_ptr)
        tl.store(extract_index + pid * 2 + 1, extract_data)

    else:
        extract_data = tl.load(extract_load_ptr)
        tl.store(extract_index + pid * 2, extract_data)

@triton.jit
def create_extend_spec_info(verified_id, seq_len, accept_len, accept_len_cum, positions, new_verified_id, accept_len_upper: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = 0 if pid ==0 else tl.load(accept_len_cum+pid-1)
    seq_length = tl.load(seq_len+pid)
    accept_length = tl.load(accept_len+pid)
    positions_ptr = positions+offset
    data = tl.arange(0, accept_len_upper)
    mask = data < accept_length
    tl.store(positions_ptr+data, seq_length-accept_length+data, mask)
    
    offset = tl.load(accept_len_cum+pid)-1
    verified_id_data = tl.load(verified_id+offset)
    tl.store(new_verified_id+pid, verified_id_data)
    

@triton.jit
def assign_req_to_token_pool(req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, pool_len: tl.constexpr, bs_upper: tl.constexpr):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset+pid)
    kv_end = tl.load(end_offset+pid)
    token_pool = req_to_token + tl.load(req_pool_indices+pid) * pool_len
    
    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset+length_offset, mask=length_offset<pid)
    end = tl.load(end_offset+length_offset, mask=length_offset<pid)
    out_offset = tl.sum(end-start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset
    
    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)
    
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr+load_offset, mask=mask)
        tl.store(token_pool+save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE
        
        
@triton.jit
def generate_draft_decode_kv_indices(req_pool_indices, req_to_token, paged_kernel_lens, kv_indices,
                                     iters: tl.constexpr, topk: tl.constexpr, pool_len: tl.constexpr, 
                                     bs_upper: tl.constexpr, iter_upper: tl.constexpr):
    BLOCK_SIZE: tl.constexpr = 128
    bid = tl.program_id(axis=0)
    topk_id = tl.program_id(axis=1)
    
    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(paged_kernel_lens+load_offset, mask=load_offset<bid)
    seq_len = tl.load(paged_kernel_lens+bid)
    cum_seq_len = tl.sum(seq_lens)
    
    kv_offset = cum_seq_len*topk+bid*iters*topk+ topk_id*(seq_len+iters)
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices+bid) * pool_len
    
    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < seq_len
        data = tl.load(token_pool_ptr+kv_offset, mask=mask)
        tl.store(kv_ptr+kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE
        
    extend_offset = tl.arange(0, iter_upper)
    extend_data = tl.load(token_pool_ptr+seq_len+tl.arange(0, iter_upper)*topk+topk_id, 
                          mask=extend_offset<iters)
    tl.store(kv_ptr+seq_len+extend_offset, extend_data, mask=extend_offset<iters)


@DraftInfoFactory.register("EAGLE", "DraftInput")
class EAGLEDraftInput(SpecDraftInput):
    hidden_states: torch.Tensor = None
    verified_id: torch.Tensor = None
    positions: torch.Tensor = None
    accept_length: torch.Tensor = None
    has_finished: bool = False
    unfinished_index: List[int] = None

    def init(self, server_args: ServerArgs):
        self.prev_mode = ForwardMode.DECODE
        self.sample_output = None
        self.topk: int = server_args.eagle_topk
        self.num_verify_token: int = server_args.num_draft_tokens
        self.spec_steps = server_args.num_speculative_steps

        self.scores: torch.Tensor = None
        self.score_list: List[torch.Tensor] = []
        self.token_list: List[torch.Tensor] = []
        self.origin_score_list: List[torch.Tensor] = [] #used for sampling
        self.parents_list: List[torch.Tensor] = []
        self.cache_list: List[torch.Tenor] = []
        self.iter = 0
        self.root_token: int = None
        
        assert self.topk <= 10, "topk should <= 10"

    def prepare_for_extend(self, batch: ForwardBatch):
        req_pool_indices = batch.alloc_req_slots(len(batch.reqs))
        out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
        batch.out_cache_loc = out_cache_loc
        
        pt=0
        for i, req in enumerate(batch.reqs):
            req.req_pool_idx = req_pool_indices[i]
            pre_len, seq_len = len(req.prefix_indices), len(req.fill_ids)
            assert seq_len - pre_len == req.extend_input_len

            if pre_len > 0:
                batch.req_to_token_pool.req_to_token[req.req_pool_idx][
                    :pre_len
                ] = req.prefix_indices

            batch.req_to_token_pool.req_to_token[req.req_pool_idx][pre_len:seq_len] = (
                out_cache_loc[pt : pt + req.extend_input_len]
            )

            pt += req.extend_input_len
        
        seq_lens = [0] + batch.extend_lens
        input_ids = batch.input_ids.tolist()
        verified_id = batch.spec_info.verified_id.tolist()
        model_input_ids = []
        for i in range(len(seq_lens) - 1):
            model_input_ids.extend(
                input_ids[seq_lens[i] + 1 : seq_lens[i + 1]] + [verified_id[i]]
            )
        batch.input_ids = torch.tensor(
            model_input_ids, dtype=torch.int32, device="cuda"
        )

    def capture_for_decode(self, sample_output: SampleOutput, hidden_states: torch.Tensor,
                           prev_mode: ForwardMode):
        self.sample_output = sample_output
        self.prev_mode = prev_mode
        self.hidden_states = hidden_states

    def prepare_for_decode(self, batch: ScheduleBatch):
        prob = self.sample_output  # b * (1/topk), vocab
        top = torch.topk(prob, self.topk, dim=-1)
        topk_index, topk_p = top.indices, top.values  # b * (1/topk), topk
        if self.prev_mode == ForwardMode.DECODE:
            scores = torch.mul(
                self.scores.unsqueeze(2), topk_p.reshape(-1, self.topk, self.topk)
            )  # (b, topk) mul (b * topk ,topk) -> b, topk, topk
            topk_cs = torch.topk(
                scores.flatten(start_dim=1), self.topk, dim=-1
            )  # (b, topk)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            self.scores = topk_cs_p

            selected_input_index = topk_cs_index.flatten() // self.topk  # b* topk

            batch.spec_info.hidden_states = batch.spec_info.hidden_states[
                selected_input_index, :
            ]
            topk_index = topk_index.reshape(-1, self.topk**2)
            batch.input_ids = torch.gather(
                topk_index, index=topk_cs_index, dim=1
            ).flatten()
            batch.out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
            self.score_list.append(scores) # b, topk, topk
            self.token_list.append(topk_index) # b, topk*topk
            self.origin_score_list.append(topk_p.reshape(topk_index.shape))
            self.parents_list.append(
                topk_cs_index + (self.topk**2 * (self.iter - 1) + self.topk)
            ) # b, topk

        elif self.prev_mode in (ForwardMode.EXTEND, ForwardMode.SPECEXTEND) :
            self.scores = topk_p  # b, top_k
            self.score_list.append(topk_p.unsqueeze(1))
            self.token_list.append(topk_index)
            self.origin_score_list.append(topk_p)
            batch.spec_info.hidden_states = (
                batch.spec_info.hidden_states.repeat_interleave(self.topk, 0)
            )
            batch.input_ids = topk_index.flatten()
            batch.out_cache_loc = batch.alloc_token_slots(topk_index.numel())
            self.parents_list.append(
                torch.arange(-1, self.topk, dtype=torch.long, device="cuda").unsqueeze(0).repeat(self.scores.shape[0], 1)
            ) # b, topk+1
        self.cache_list.append(batch.out_cache_loc)
        self.positions = (
            batch.seq_lens[:, None]
            + torch.ones([1, self.topk], device="cuda", dtype=torch.long) * self.iter
        ).flatten()
        
        bs = batch.seq_lens.numel()
        assign_req_to_token_pool[(bs, )](batch.req_pool_indices,
                                 batch.req_to_token_pool.req_to_token,
                                 batch.seq_lens+self.topk*self.iter, 
                                 batch.seq_lens+self.topk*(self.iter+1), 
                                 batch.out_cache_loc,
                                 batch.req_to_token_pool.req_to_token.shape[1],
                                 triton.next_power_of_2(bs)
                                )
        self.iter += 1
        
    def prepare_extend_after_decode(self, batch: ScheduleBatch):
        batch.out_cache_loc = batch.alloc_token_slots(self.verified_id.numel())
        batch.extend_lens = (self.accept_length+1).tolist()
        
        pt=0
        seq_lens = batch.seq_lens.tolist()
        
        i = 0
        # TODO: Chage it to triton kernel @kavioyu
        for req in batch.reqs:
            if req.finished():
                continue
            #assert seq_len - pre_len == req.extend_input_len
            input_len = self.accept_length[i] + 1
            seq_len = seq_lens[i]
            batch.req_to_token_pool.req_to_token[req.req_pool_idx][seq_len-input_len:seq_len] = (
                batch.out_cache_loc[pt : pt + input_len]
            )
            pt += input_len
            i += 1

        self.positions = torch.empty_like(self.verified_id)
        new_verified_id = torch.empty_like(self.accept_length, dtype=torch.long)
        self.accept_length.add_(1)

        create_extend_spec_info[(self.accept_length.numel(),)](self.verified_id, batch.seq_lens, 
                self.accept_length, torch.cumsum(self.accept_length, axis=0, dtype=torch.int),
                self.positions, new_verified_id, triton.next_power_of_2(self.spec_steps+1))

        batch.input_ids = self.verified_id
        self.verified_id = new_verified_id
    

    def prepare_for_verify(self, batch: ScheduleBatch):
        score_list = torch.cat(self.score_list, dim=1).flatten(1)  # b, n, topk; n= 1+(self.iter-1)*self.topk
        ss_token_list = torch.cat(self.token_list, dim=1)  # b, (self.topk+(self.iter-1)*self.topk)
        origin_token_list = torch.cat(self.origin_score_list, dim=1)
        top_scores = torch.topk(score_list, self.num_verify_token - 1, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)
        scores = torch.gather(origin_token_list, index=top_scores_index, dim=1)
        draft_tokens = torch.cat((self.verified_id.unsqueeze(1), draft_tokens), dim=1)
        parent_list = torch.cat(self.parents_list[:-1], dim=1)

        tree_mask, position, retrive_index, retrive_cum_len = build_tree_kernel(
            parent_list,
            top_scores_index,
            batch.seq_lens,
            self.topk,
            self.iter - 1,
            self.num_verify_token,
        )

        return EagleVerifyInput(
            draft_tokens.flatten(),
            scores.flatten(),
            tree_mask,
            position,
            retrive_index,
            retrive_cum_len,
            self.num_verify_token,
        )

    def generate_attn_arg(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        req_to_token: torch.Tensor,
    ):
        seq_num = req_pool_indices.numel()
        bs = self.topk * req_pool_indices.numel()
        seq_len = self.positions.reshape(-1).contiguous()
        
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        cum_kv_seq_len[1:] = torch.cumsum(seq_len + 1, dim=0)
        kv_last_page_len = torch.ones((bs,), dtype=torch.int32, device="cuda")
        total_len = torch.sum(paged_kernel_lens).item()
        
        kv_indices = torch.empty((total_len * self.topk + seq_num*self.iter*self.topk, ), 
                                 dtype=torch.int32, device='cuda')
        
        generate_draft_decode_kv_indices[(req_pool_indices.numel(), self.topk)](req_pool_indices, req_to_token,
                                         paged_kernel_lens, kv_indices, self.iter, self.topk, 
                                         req_to_token.shape[1], triton.next_power_of_2(seq_num),
                                         triton.next_power_of_2(self.spec_steps))
        return kv_indices, cum_kv_seq_len, kv_last_page_len, None

    def clear(self):
        self.iter = 0
        self.score_list.clear()
        self.positions = None
    
    def clear_draft_cache(self, batch):
        draft_cache = torch.cat(self.cache_list, dim=0)
        batch.token_to_kv_pool.free(draft_cache)

    def generate_attn_arg_spec_extend(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        req_to_token: torch.Tensor,
    ):
        bs = self.accept_length.numel()
        qo_indptr = torch.zeros(
            (bs + 1,), dtype=torch.int32, device="cuda"
        )
        qo_indptr[1:] = torch.cumsum(self.accept_length, dim=0)
        
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)
        
        kv_last_page_len = torch.ones((bs,), dtype=torch.int32, device="cuda")
        
        kv_indices = torch.empty(cum_kv_seq_len[-1], dtype=torch.int32, device="cuda")
        
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )

        return kv_indices, cum_kv_seq_len, kv_last_page_len, qo_indptr
    
    def merge_batch(self, spec_info: EAGLEDraftInput):
        
        self.hidden_states = torch.cat([self.hidden_states, spec_info.hidden_states], axis=0)
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], axis=0)
        #self.positions = torch.cat([self.positions, spec_info.positions], axis=0)
        self.sample_output = torch.cat([self.sample_output, spec_info.sample_output])

@DraftInfoFactory.register("EAGLE", "VerifyInput")
class EagleVerifyInput(SpecVerifyInput):
    def __init__(
        self,
        draft_token: torch.Tensor,
        draft_score: torch.Tensor,
        tree_mask: torch.Tensor,
        positions: torch.Tensor,
        retrive_index: torch.Tensor,
        retrive_cum_len: torch.Tensor,
        draft_token_num: int,
    ):
        self.draft_token = draft_token
        self.draft_score = draft_score
        self.custom_mask = tree_mask
        self.positions = positions
        self.retrive_index = retrive_index
        self.retrive_cum_len = retrive_cum_len
        self.draft_token_num = draft_token_num

    def prepare_for_verify(self, batch: ScheduleBatch):
        batch.input_ids = self.draft_token
        batch.out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
        bs = batch.seq_lens.numel()
        assign_req_to_token_pool[(bs, )](batch.req_pool_indices,
                            batch.req_to_token_pool.req_to_token,
                            batch.seq_lens, 
                            batch.seq_lens+self.draft_token_num, 
                            batch.out_cache_loc,
                            batch.req_to_token_pool.req_to_token.shape[1],
                            triton.next_power_of_2(bs)
                        )

    def generate_attn_arg(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        req_to_token: torch.Tensor,
    ):
        batch_size = len(req_pool_indices)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device="cuda",
        )

        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")

        kv_indices = torch.empty(cum_kv_seq_len[-1], dtype=torch.int32, device="cuda")
        
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, kv_last_page_len, qo_indptr

    def verify(self, batch: ScheduleBatch, logits_output: torch.Tensor) -> torch.Tensor:
        predict = torch.argmax(logits_output.next_token_logits_bak, dim=-1)
        predict = torch.cat([predict, torch.full([1], -1, dtype=torch.long, device='cuda')], dim=-1)
        draft_token = torch.cat([self.draft_token, torch.full([1], -1, dtype=torch.long, device='cuda')], dim=-1)
        target_predict = predict[self.retrive_index]
        candidates = draft_token[self.retrive_index]
        # logits = logits_output.next_token_logits[self.retrive_index]
        # target_predict = torch.argmax(logits[:, :-1], dim=-1)
        accept_mask = candidates[:, 1:] == target_predict[:, :-1]
        accept_mask = (torch.cumprod(accept_mask, dim=1)).sum(dim=1)
        bs = self.retrive_cum_len.numel() - 1

        max_draft_len = self.retrive_index.shape[-1]
        accept_index = torch.full(
            (bs, max_draft_len), -1, dtype=torch.long, device="cuda"
        )
        accept_length = torch.empty((bs,), dtype=torch.int, device="cuda")
        extract_index = torch.full((bs * 2,), 0, dtype=torch.int, device="cuda")
        eagle_verify_retrive[(bs,)](
            self.retrive_index.contiguous(),
            accept_mask.contiguous(),
            self.retrive_cum_len,
            accept_index,
            accept_length,
            extract_index,
            max_draft_len,
            self.draft_token_num,
            triton.next_power_of_2(max_draft_len),
        )

        accept_index = accept_index[accept_index != -1]
        #extract_index = extract_index[extract_index != 0]
        
        draft_input = EAGLEDraftInput()

        accept_length_cpu = accept_length.tolist()
        verified_id = predict[accept_index]
        verified_id_cpu = verified_id.tolist()
        
        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False
        mem_need_free_idx = batch.out_cache_loc[evict_mask]
        batch.token_to_kv_pool.free(mem_need_free_idx)
        batch.seq_lens.add_(accept_length+1)

        assign_req_to_token_pool[(bs, )](batch.req_pool_indices,
                            batch.req_to_token_pool.req_to_token,
                            batch.seq_lens, 
                            batch.seq_lens+accept_length+1, 
                            batch.out_cache_loc[accept_index],
                            batch.req_to_token_pool.req_to_token.shape[1],
                            triton.next_power_of_2(bs)
                        )
        
        new_accept_index = []
        unfinished_index = []
        finished_extend_len = {} # {rid:accept_length + 1}
        #retracted_reqs, new_token_ratio = batch.retract_decode()

        low = 0
        for i, (req, verified_len) in enumerate(zip(batch.reqs, accept_length_cpu)):
            req.output_ids.extend(verified_id_cpu[low : low + verified_len + 1])
            req.check_finished()
            if req.finished():
                draft_input.has_finished = True
                finished_extend_len[req.rid] = verified_len+1
            else:
                new_accept_index.append(accept_index[low: low+verified_len+1])
                unfinished_index.append(i)
            low += verified_len + 1
        
        if len(new_accept_index)>0:
            new_accept_index = torch.cat(new_accept_index, dim=0)
            draft_input.verified_id = predict[new_accept_index]
            draft_input.hidden_states = batch.spec_info.hidden_states[
                new_accept_index
            ]
            draft_input.accept_length = accept_length[unfinished_index]
            draft_input.unfinished_index = unfinished_index

        logits_output.next_token_logits = logits_output.next_token_logits_bak[accept_index]
        return draft_input, logits_output, verified_id, finished_extend_len

