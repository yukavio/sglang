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

class TestEAGLEEngine2(CustomTestCase):
    BASE_CONFIG = {
        "model_path": DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
        "speculative_draft_model_path": DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
        "speculative_algorithm": "EAGLE",
        "speculative_algorithm": "NAIVE_EAGLE",
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 2,
        "mem_fraction_static": 0.7,
        "cuda_graph_max_bs": 8,
        "disable_cuda_graph": False,
        "disable_cuda_graph": True, 
        "disable_overlap_schedule": True,
        "requests_all_greedy": False,
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
            "The president of the United States is",
            "The capital of France is",
            "The future of come is A",
            "The future of come is A B C /D E F G H I J K L M N O P Q R S T U V W X Y Z",
            "I am testing Eagle!!!! NoW!!",
            "How old are you?",
            "How is the wheather like today?",
            "近年来，随着中国农业现代化进程的不断推进，智能温室作为现代设施农业的重要组成部分，受到政策和产业界的高度重视。针对当前温室管理中存在的效率低、资源利用率不高等问题，本文提出并设计了一种基于物联网与云计算技术的智能温室控制系统。该系统集成了环境数据实时监测、多因子协调控制算法、远程设备管理以及数据分析与智能预警等功能模块，能够实现对温室内外多项环境参数（如温湿度、光照、CO₂浓度、pH值等）的精准采集与智能调控。系统采用软硬件协同设计，硬件部分基于ARM架构的嵌入式平台，软件部分采用SpringBoot与Vue框架开发，并结合MQTT协议实现高效的数据通信。实验结果表明，该系统能够有效提升温室环境管理的自动化与智能化水平，显著提高农业生产效率和资源利用率。该研究为现代农业实体提供了一种高效、智能的温室管理解决方案，对推动农业生产的可持续发展具有重要意义。",
            "In recent years, with the continuous advancement of agricultural modernization in China, intelligent greenhouses have become a crucial component of modern facility agriculture and have attracted significant attention from both policymakers and industry. To address issues such as low management efficiency and suboptimal resource utilization in traditional greenhouse operations, this paper proposes and designs an intelligent greenhouse control system based on Internet of Things (IoT) and cloud computing technologies. The system integrates real-time environmental data monitoring, multi-factor coordinated control algorithms, remote device management, as well as data analysis and intelligent warning modules, enabling precise acquisition and intelligent regulation of various environmental parameters (such as temperature, humidity, light intensity, CO₂ concentration, and pH value) inside and outside the greenhouse. The system adopts a collaborative design of hardware and software, with the hardware based on an ARM-architecture embedded platform, and the software developed using SpringBoot and Vue frameworks, combined with the MQTT protocol for efficient data communication. Experimental results demonstrate that the system effectively enhances the automation and intelligence level of greenhouse environmental management, significantly improving agricultural production efficiency and resource utilization. This research provides an efficient and intelligent greenhouse management solution for modern agricultural entities and holds significant implications for promoting the sustainable development of agricultural production.",
        ]
        params = {"temperature": 0, "max_new_tokens": 500}
        # params = {"temperature": 1, "max_new_tokens": 50, "top_p": 0.8, "top_k": 10000}

        outputs = engine.generate(prompts, params)
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Generated: {output['text']}")
            print("-" * 40)

        print(f"{engine.get_server_info()=}")

        avg_spec_accept_length = engine.get_server_info()["avg_spec_accept_length"]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 1.5)

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

class TestNaiveEAGLEEngine(CustomTestCase):
    BASE_CONFIG = {
        "model_path": DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
        "speculative_draft_model_path": DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
        "speculative_algorithm": "NAIVE_EAGLE",
        "mem_fraction_static": 0.7,
        "cuda_graph_max_bs": 4,
        # "disable_cuda_graph": True,
        "requests_all_greedy": True,
    }
    NUM_CONFIGS = 2

    def setUp(self):
        self.prompt = "Today is a sunny day and I like"
        self.sampling_params = {"temperature": 0, "max_new_tokens": 8}

        ref_engine = sgl.Engine(
            model_path=self.BASE_CONFIG["model_path"], cuda_graph_max_bs=1
        )
        self.ref_output = ref_engine.generate(self.prompt, self.sampling_params)["text"]
        ref_engine.shutdown()

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
                    self._test_single_generation(engine)
                    self._test_batch_generation(engine)
                    self._test_eos_token(engine)
                    self._test_acc_length(engine)
                finally:
                    engine.shutdown()
                print("=" * 100)

    def _test_single_generation(self, engine):
        output = engine.generate(self.prompt, self.sampling_params)["text"]
        print(f"{output=}, {self.ref_output=}")
        self.assertEqual(output, self.ref_output)

    def _test_batch_generation(self, engine):
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
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
        self.assertGreater(avg_spec_accept_length, 1.4)

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


class TestNaiveEAGLEServer(CustomTestCase):
    PROMPTS = [
        "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nToday is a sunny day and I like[/INST]"
        '[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nWhat are the mental triggers in Jeff Walker\'s Product Launch Formula and "Launch" book?[/INST]',
        "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nSummarize Russell Brunson's Perfect Webinar Script...[/INST]",
        "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwho are you?[/INST]",
        "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwhere are you from?[/INST]",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "NAIVE_EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def send_request(self):
        time.sleep(random.uniform(0, 2))
        for prompt in self.PROMPTS:
            url = self.base_url + "/generate"
            data = {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1024,
                },
            }
            response = requests.post(url, json=data)
            # print(response)
            assert response.status_code == 200

    def send_requests_abort(self):
        for prompt in self.PROMPTS:
            try:
                time.sleep(random.uniform(0, 2))
                url = self.base_url + "/generate"
                data = {
                    "model": "base",
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1024,
                    },
                }
                # set timeout = 1s, mock disconnected
                requests.post(url, json=data, timeout=1)
            except Exception as e:
                print(e)
                pass

    def test_request_abort(self):
        concurrency = 4
        threads = [
            threading.Thread(target=self.send_request) for _ in range(concurrency)
        ] + [
            threading.Thread(target=self.send_requests_abort)
            for _ in range(concurrency)
        ]
        for worker in threads:
            worker.start()
        for p in threads:
            p.join()

    # def test_max_token_one(self):
    #     requests.get(self.base_url + "/flush_cache")

    #     args = SimpleNamespace(
    #         num_shots=5,
    #         data_path=None,
    #         num_questions=200,
    #         max_new_tokens=1,
    #         parallel=128,
    #         host="http://127.0.0.1",
    #         port=int(self.base_url.split(":")[-1]),
    #     )

    #     # Just run and check it does not hang
    #     metrics = run_eval(args)
    #     self.assertGreater(metrics["output_throughput"], 30)

    # TODO: fix this
    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.20)

        server_info = requests.get(self.base_url + "/get_server_info").json()
        avg_spec_accept_length = server_info["avg_spec_accept_length"]
        print(f"{avg_spec_accept_length=}")

        speculative_eagle_topk = server_info["speculative_eagle_topk"]

        if speculative_eagle_topk == 1:
            self.assertGreater(avg_spec_accept_length, 2.5)
        else:
            self.assertGreater(avg_spec_accept_length, 3.5)

        # Wait a little bit so that the memory check happens.
        time.sleep(4)

    # def test_logprob_start_len(self):
    #     logprob_start_len = 4
    #     new_tokens = 4
    #     prompts = [
    #         "I have a very good idea on",
    #         "Today is a sunndy day and",
    #     ]

    #     response = requests.post(
    #         self.base_url + "/generate",
    #         json={
    #             "text": prompts,
    #             "sampling_params": {
    #                 "temperature": 0,
    #                 "max_new_tokens": new_tokens,
    #             },
    #             "return_logprob": True,
    #             "top_logprobs_num": 5,
    #             "logprob_start_len": logprob_start_len,
    #         },
    #     )
    #     response_json = response.json()
    #     print(json.dumps(response_json, indent=2))

    #     for res in response_json:
    #         self.assertEqual(
    #             res["meta_info"]["prompt_tokens"],
    #             logprob_start_len + len(res["meta_info"]["input_token_logprobs"]),
    #         )

    #         self.assertEqual(res["meta_info"]["completion_tokens"], new_tokens)
    #         self.assertEqual(len(res["meta_info"]["output_token_logprobs"]), new_tokens)

    # def test_logprob_match(self):
    #     """Test the output logprobs are close to the input logprobs if we run a prefill again."""

    #     def run_generate(
    #         prompt, return_logprob=False, max_new_tokens=512, logprob_start_len=-1
    #     ):

    #         if isinstance(prompt, str):
    #             prompt_kwargs = {"text": prompt}
    #         else:
    #             prompt_kwargs = {"input_ids": prompt}

    #         response = requests.post(
    #             self.base_url + "/generate",
    #             json={
    #                 **prompt_kwargs,
    #                 "sampling_params": {
    #                     "temperature": 1.0,
    #                     "max_new_tokens": max_new_tokens,
    #                     "ignore_eos": True,
    #                 },
    #                 "return_logprob": return_logprob,
    #                 "return_text_in_logprobs": True,
    #                 "logprob_start_len": logprob_start_len,
    #             },
    #         )
    #         return response.json()

    #     prompt = "I have a very good idea on how to"

    #     gen = run_generate(prompt, return_logprob=True, logprob_start_len=0)
    #     output_logprobs = np.array(
    #         [x[0] for x in gen["meta_info"]["output_token_logprobs"]]
    #     )
    #     num_prompts_tokens = gen["meta_info"]["prompt_tokens"]

    #     input_tokens = [x[1] for x in gen["meta_info"]["input_token_logprobs"]]
    #     output_tokens = [x[1] for x in gen["meta_info"]["output_token_logprobs"]]

    #     new_prompt = input_tokens + output_tokens
    #     score = run_generate(
    #         new_prompt, return_logprob=True, logprob_start_len=0, max_new_tokens=0
    #     )
    #     output_logprobs_score = np.array(
    #         [
    #             x[0]
    #             for x in score["meta_info"]["input_token_logprobs"][num_prompts_tokens:]
    #         ]
    #     )

    #     print(f"{output_logprobs[-10:]=}")
    #     print(f"{output_logprobs_score[-10:]=}")

    #     diff = np.abs(output_logprobs - output_logprobs_score)
    #     max_diff = np.max(diff)
    #     self.assertLess(max_diff, 0.25)

    # def test_logprob_mixed(self):
    #     args = []
    #     temperature = 0
    #     # input_len, output_len, temperature, logprob_start_len, return_logprob, top_logprobs_num
    #     # Llama 2 context length seems to be only 2k, so we can only test small length.
    #     for input_len in [200, 500, 1000, 2000]:
    #         for output_len in [4, 8]:
    #             for logprob_start_len in [0, 100, 300, 800, 1998]:
    #                 for return_logprob in [True, False]:
    #                     for top_logprobs_num in [0, 5]:

    #                         if logprob_start_len >= input_len:
    #                             continue

    #                         args.append(
    #                             (
    #                                 input_len,
    #                                 output_len,
    #                                 temperature,
    #                                 logprob_start_len,
    #                                 return_logprob,
    #                                 top_logprobs_num,
    #                             )
    #                         )

    #     random.shuffle(args)

    #     func = partial(run_logprob_check, self)
    #     with ThreadPoolExecutor(8) as executor:
    #         list(executor.map(func, args))

    # def run_decode(self, sampling_params):
    #     return_logprob = True
    #     top_logprobs_num = 5
    #     return_text = True
    #     n = 1

    #     response = requests.post(
    #         self.base_url + "/generate",
    #         json={
    #             "text": "Human: Write a travel blog post to Hawaii.\n\nAssistant:",
    #             "sampling_params": {
    #                 "max_new_tokens": 48,
    #                 "n": n,
    #                 "temperature": 0.7,
    #                 **sampling_params,
    #             },
    #             "return_logprob": return_logprob,
    #             "top_logprobs_num": top_logprobs_num,
    #             "return_text_in_logprobs": return_text,
    #             "logprob_start_len": 0,
    #         },
    #     )
    #     self.assertEqual(response.status_code, 200)
    #     print(json.dumps(response.json()))
    #     print("=" * 100)

    # def test_penalty_mixed(self):
    #     args = [
    #         {},
    #         {},
    #         {},
    #         {"frequency_penalty": 2},
    #         {"presence_penalty": 1},
    #         {"min_new_tokens": 16},
    #         {"frequency_penalty": 0.2},
    #         {"presence_penalty": 0.4},
    #         {"min_new_tokens": 8},
    #         {"frequency_penalty": 0.4, "presence_penalty": 0.8},
    #         {"frequency_penalty": 0.4, "min_new_tokens": 12},
    #         {"presence_penalty": 0.8, "min_new_tokens": 12},
    #         {"presence_penalty": -0.3, "frequency_penalty": 1.3, "min_new_tokens": 32},
    #         {"presence_penalty": 0.3, "frequency_penalty": -1.3, "min_new_tokens": 32},
    #     ]
    #     random.shuffle(args * 5)
    #     with ThreadPoolExecutor(8) as executor:
    #         list(executor.map(self.run_decode, args))


if __name__ == "__main__":
    unittest.main()
