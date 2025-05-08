import json
import multiprocessing as mp
import os
import random
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import requests
import torch

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.runners import DEFAULT_PROMPTS, SRTRunner
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_logprob_check,
)

torch_dtype = torch.float16
prefill_tolerance = 5e-2
decode_tolerance: float = 5e-2


class TestEAGLEEngine(CustomTestCase):
    BASE_CONFIG = {
        "model_path": DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
        "speculative_draft_model_path": DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
        # "speculative_algorithm": "EAGLE",
        "speculative_algorithm": "NAIVE_EAGLE",
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 1,
        "mem_fraction_static": 0.7,
        "cuda_graph_max_bs": 2,
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
    }
    NUM_CONFIGS = 1

    def setUp(self):
        pass
        # self.prompt = "Today is"
        # self.sampling_params = {"temperature": 0, "max_new_tokens": 8}

        # ref_engine = sgl.Engine(
        #     model_path=self.BASE_CONFIG["model_path"], cuda_graph_max_bs=1
        # )
        # self.ref_output = ref_engine.generate(self.prompt, self.sampling_params)["text"]
        # ref_engine.shutdown()

    def test_correctness(self):
        configs = [
            # Basic config
            self.BASE_CONFIG,
            # Chunked prefill
            {**self.BASE_CONFIG, "chunked_prefill_size": 4},
        ]

        for i, config in enumerate(configs[: self.NUM_CONFIGS]):
            with self.subTest(i=i):
                print(f"{config=}")
                engine = sgl.Engine(**config, log_level="info", decode_log_interval=10)
                try:
                    pass
                    # self._test_single_generation(engine)
                    self._test_batch_generation(engine)
                    # self._test_eos_token(engine)
                    # self._test_acc_length(engine)
                finally:
                    engine.shutdown()
                    pass
                print("=" * 100)

    def _test_single_generation(self, engine):
        output = engine.generate(self.prompt, self.sampling_params)["text"]
        print(f"{output=}, {self.ref_output=}")
        self.assertEqual(output, self.ref_output)

    def _test_batch_generation(self, engine):
        prompts = [
            "Hello The",
            "One hamster", 
            # "The president of the United States is",
            # "The capital of France is",
            # "The future of come is A",
            # "The future of come is A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
            # "I am testing Eagle!!!! NoW!!",
            # "How old are you?",
            # "近年来，随着中国农业现代化进程的不断推进，智能温室作为现代设施农业的重要组成部分，受到政策和产业界的高度重视。针对当前温室管理中存在的效率低、资源利用率不高等问题，本文提出并设计了一种基于物联网与云计算技术的智能温室控制系统。该系统集成了环境数据实时监测、多因子协调控制算法、远程设备管理以及数据分析与智能预警等功能模块，能够实现对温室内外多项环境参数（如温湿度、光照、CO₂浓度、pH值等）的精准采集与智能调控。系统采用软硬件协同设计，硬件部分基于ARM架构的嵌入式平台，软件部分采用SpringBoot与Vue框架开发，并结合MQTT协议实现高效的数据通信。实验结果表明，该系统能够有效提升温室环境管理的自动化与智能化水平，显著提高农业生产效率和资源利用率。该研究为现代农业实体提供了一种高效、智能的温室管理解决方案，对推动农业生产的可持续发展具有重要意义。",
            # "In recent years, with the continuous advancement of agricultural modernization in China, intelligent greenhouses have become a crucial component of modern facility agriculture and have attracted significant attention from both policymakers and industry. To address issues such as low management efficiency and suboptimal resource utilization in traditional greenhouse operations, this paper proposes and designs an intelligent greenhouse control system based on Internet of Things (IoT) and cloud computing technologies. The system integrates real-time environmental data monitoring, multi-factor coordinated control algorithms, remote device management, as well as data analysis and intelligent warning modules, enabling precise acquisition and intelligent regulation of various environmental parameters (such as temperature, humidity, light intensity, CO₂ concentration, and pH value) inside and outside the greenhouse. The system adopts a collaborative design of hardware and software, with the hardware based on an ARM-architecture embedded platform, and the software developed using SpringBoot and Vue frameworks, combined with the MQTT protocol for efficient data communication. Experimental results demonstrate that the system effectively enhances the automation and intelligence level of greenhouse environmental management, significantly improving agricultural production efficiency and resource utilization. This research provides an efficient and intelligent greenhouse management solution for modern agricultural entities and holds significant implications for promoting the sustainable development of agricultural production.",
        ]
        params = {"temperature": 0, "max_new_tokens": 50}

        outputs = engine.generate(prompts, params)
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Generated: {output['text']}")
            print("-" * 40)

        print(f"{engine.get_server_info()=}")

        avg_spec_accept_length = engine.get_server_info()["avg_spec_accept_length"]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 1.9)

    def _test_eos_token(self, engine):
        prompt = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\nToday is a sunny day and I like [/INST]"
        params = {
            "temperature": 0.1,
            "max_new_tokens": 1024,
            "skip_special_tokens": False,
        }

        tokenizer = get_tokenizer(DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST)
        output = engine.generate(prompt, params)["text"]
        print(f"{output=}")

        tokens = tokenizer.encode(output, truncation=False)
        self.assertNotIn(tokenizer.eos_token_id, tokens)

    def _test_acc_length(self, engine):
        prompt = [
            "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:",
        ] * 5  # test batched generation
        sampling_params = {"temperature": 0, "max_new_tokens": 512}
        output = engine.generate(prompt, sampling_params)
        output = output[0]

        if "spec_verify_ct" in output["meta_info"]:
            acc_length = (
                output["meta_info"]["completion_tokens"]
                / output["meta_info"]["spec_verify_ct"]
            )
        else:
            acc_length = 1.0

        speed = (
            output["meta_info"]["completion_tokens"]
            / output["meta_info"]["e2e_latency"]
        )
        print(f"{acc_length=}")

        if engine.server_args.model_path == DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST:
            self.assertGreater(acc_length, 3.6)
        else:
            self.assertGreater(acc_length, 2.5)







if __name__ == "__main__":
    unittest.main()
