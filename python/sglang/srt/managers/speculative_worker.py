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

"""A speculative draft worker."""

from sglang.srt.managers.speculative_utils import SpecInfoPipline
from sglang.srt.managers.tp_worker import ModelTpServer
from sglang.srt.server_args import ServerArgs


class SpecDraftServer(ModelTpServer):
    def __init__(
        self,
        gpu_id: int,
        tp_rank: int,
        server_args: ServerArgs,
        nccl_port: int,
        model_overide_args: dict,
        spec_queue: SpecInfoPipline,
    ):
        super().__init__(
            gpu_id, tp_rank, server_args, nccl_port, model_overide_args, spec_queue
        )
