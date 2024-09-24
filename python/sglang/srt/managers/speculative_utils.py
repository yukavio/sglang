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

from __future__ import annotations

from typing import TYPE_CHECKING, List, Type

import torch

from python.sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    InputMetadata,
)

if TYPE_CHECKING:
    from python.sglang.srt.layers.sampler import SampleOutput
    from python.sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class SpecDraftInput:
    def prepare_for_extend(self, batch):
        raise NotImplementedError()

    def prepare_for_decode(self, batch):
        raise NotImplementedError()

    def generate_attn_arg(
        self,
        req_pool_indices: List,
        paged_kernel_lens: List,
        req_to_token_pool: ReqToTokenPool,
    ):
        raise NotImplementedError()

    def clear():
        pass


class SpecVerifyInput:
    pass


class SpecDraftInfoFactory:
    def __init__(self):
        self.factory = {}

    def register(self, name: str) -> SpecDraftInput:
        def wrapper(info: Type[SpecDraftInput]) -> Type[SpecDraftInput]:
            self.factory[name] = info
            return info

        return wrapper

    def get(self, name):
        return self.factory[name]


DraftInfoFactory = SpecDraftInfoFactory()


@DraftInfoFactory.register("EAGLE")
class EAGLEDraftInput(SpecDraftInput):
    hidden_states: torch.Tensor = None
    verified_id: torch.Tensor = None

    prev_mode = None
    sample_output = None
    topk: int = 10
    num_verify_token: int = 60

    scores: torch.Tensor = None
    score_list: List[torch.Tensor] = []
    token_list: List[torch.Tensor] = []
    iter = 0
    root_token: int = None

    positions: torch.Tensor = None
    tree_mask: torch.Tensor = None

    def __init__(self):
        self.tree_mask_init = torch.eye(self.topk).to("cuda").unsqueeze(0)
        self.tree_mask = self.tree_mask_init.clone()

    def prepare_for_extend(self, batch):
        seq_lens = [0] + batch.seq_lens.tolist()
        input_ids = batch.input_ids.tolist()
        verified_id = self.verified_id.tolist()
        model_input_ids = []
        for i in range(len(seq_lens) - 1):
            model_input_ids.extend(
                input_ids[seq_lens[i] + 1 : seq_lens[i + 1]] + [verified_id[i]]
            )
        batch.input_ids = torch.tensor(
            model_input_ids, dtype=torch.int32, device="cuda"
        )
        self.verified_id = self.verified_id.clone()

    def capture_for_decode(self, sample_output: SampleOutput, prev_mode: ForwardMode):
        self.sample_output = sample_output
        self.prev_mode = prev_mode

    def prepare_for_decode(self, batch: ScheduleBatch):
        prob = self.sample_output  # b * (1/topk), vocab
        top = torch.topk(prob, self.topk, dim=-1)
        topk_index, topk_p = top.indices, top.values  # b * (1/topk), topk
        if self.prev_mode == ForwardMode.DECODE:
            scores = torch.mul(
                self.scores.unsqueeze(1), topk_p
            )  # (b, topk) mul (b * topk ,topk) -> b, topk, topk
            topk_cs = torch.topk(
                scores.flatten(start_dim=1), self.topk, dim=-1
            )  # (b, topk)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            self.scores = topk_cs_p

            selected_input_index = topk_cs_index.flatten() // self.topk  # b, topk

            batch.spec_draft_input.hidden_states = batch.spec_draft_input.hidden_states[
                selected_input_index, :
            ]
            batch.input_ids = torch.gather(
                topk_index.reshape(-1, self.topk**2), index=topk_cs_index, dim=1
            ).flatten()
            batch.out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
            self.score_list.append(scores)
            self.token_list.append(topk_index)

        elif self.prev_mode == ForwardMode.SPECEXTEND:
            self.scores = topk_p  # b, top_k
            self.score_list.append(topk_p.unsqueeze(1))
            self.token_list.append(topk_index)
            batch.spec_draft_input.hidden_states = (
                batch.spec_draft_input.hidden_states.repeat(self.topk, 1)
            )
            batch.input_ids = topk_index.flatten()
            batch.out_cache_loc = batch.alloc_token_slots(topk_index.numel())

        self.positions = (
            batch.seq_lens[:, None]
            + torch.ones([1, self.topk], device="cuda") * self.iter
        )
        print("allocate ", batch.out_cache_loc)
        print("next pos ", self.positions)
        # TODO: Check it @kavioyu
        batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices,
            batch.seq_lens
            + self.topk * self.iter
            - 1 : batch.seq_lens
            + self.topk * (self.iter + 1)
            - 1,
        ] = batch.out_cache_loc

        self.iter += 1

    def prepare_for_verify(self):
        score_list = torch.cat(self.score_list, dim=1).view(-1)  # b, 1/topk, topk
        ss_token_list = torch.cat(self.token_list, dim=0).view(-1)
        top_scores = torch.topk(score_list, self.num_verify_token, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((self.verified_id, draft_tokens), dim=0)

        print(draft_tokens)

    def generate_attn_arg(
        self,
        req_pool_indices: List,
        paged_kernel_lens: List,
        req_to_token_pool: ReqToTokenPool,
    ):
        bs = self.topk * len(req_pool_indices)
        seq_len = self.positions.reshape(-1).contiguous()
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        cum_kv_seq_len[1:] = torch.cumsum(seq_len, dim=0)
        kv_last_page_len = torch.ones((bs,), dtype=torch.int32, device="cuda")
        kv_indices_list = []
        for i in range(len(req_pool_indices)):
            for k in range(self.topk):
                index = torch.arange(self.iter) * self.topk + k
                kv_indices_list.append(
                    req_to_token_pool.req_to_token[
                        req_pool_indices[i], : paged_kernel_lens[i]
                    ]
                )
                kv_indices_list.append(
                    req_to_token_pool.req_to_token[
                        req_pool_indices[i], paged_kernel_lens[i] + index
                    ]
                )

        kv_indices = torch.cat(kv_indices_list, dim=0).contiguous()

        return kv_indices, cum_kv_seq_len, kv_last_page_len

    def clear(self):
        self.iter = 0
        self.score_list.clear()
        self.positions = None
        self.tree_mask = self.tree_mask_init.clone()


class SpecInfoPipline:
    def __init__(self):
        ctx = torch.multiprocessing.get_context("forkserver")
        self.draft_input_queue = ctx.Queue()
        self.draft_output_queue = ctx.Queue()
        self.max_total_num_tokens = ctx.Value("i", -1)
