
import time
import unittest
import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


LLAMA_LONG_MODEL_PATH = "meta-llama/Meta-LLama-3-8B-Instruct"

class TestLlamaDuoAttentionLong(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_LONG_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--tp", "1",
            "--trust-remote-code",
            "--context-length", "32000",

            # DuoAttention Args
            "--enable-duo-attention",
            "--duo-attn-sink-size", "64",
            "--duo-attn-streaming-window", "128",
            "--duo-attn-retrieval-idx", "0",
            "--duo-attn-streaming-idx", "1",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod 
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_needle_in_haystack(self):

        target_len = 8000

        haystack_text = "The quick brown fox jumps over the lazy dog. "
        needle = "The secret password for the SGLang test is: SUPER_DUO_ATTENTION_ROCKS. "

        approx_tokens_per_haystack = len(haystack_text) / 4
        repeats = int(target_len / approx_tokens_per_haystack)

        insertion_point = int(repeats * 0.5)

        prompt = (
            "Below is a ver long text containing a secret password.\n"
            "<text>\n"
            + (haystack_text * insertion_point)
            + needle
            + (haystack_text * (repeats - insertion_point))
            + "\n</text>\n"
            "What is the secret password mentioned in the text above? Answer directly."
        )

        print(f"Constructed prompt with approx length: {len(prompt)} chars.")

        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 64,
                "stop": ["\n", "."]
            }
        }

        start_time = time.time()
        response = requests.post(self.base_url + "generate", json=payload).json()
        end_time = time.time()

        output_text = response["text"]
        print(f"Model Output: {output_text}")
        print(f"Latency: {end_time - start_time:.2f} seconds")

        self.assertIn("SUPER_DUO_ATTENTION_ROCKS", output_text)

if __name__ == "__main__":
    unittes