import os
import time
import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.attention.duoattention.utils import load_attn_pattern, sparsify_attention_heads


ROOT_PATH = "/mnt/cephfs/chengqi/"
LLAMA_LONG_MODEL_PATH = os.path.join(ROOT_PATH, "models/Llama-3-8B-Instruct-Gradient-1048k")
ATTN_PATTERN_PATH = os.path.join(ROOT_PATH, "attn_patterns/Llama-3-8B-Instruct-Gradient-4194k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10")
BASE_URL = DEFAULT_URL_FOR_TEST

def main():
    server_process = None

    try:
        print(f"Loading attention pattern from {ATTN_PATTERN_PATH}...")
        full_attention_heads, sink_size, recent_size = load_attn_pattern(ATTN_PATTERN_PATH)

        print(f"Sink size: {sink_size}")
        print(f"Recent size: {recent_size}")

        sparsity = 0.5 
        full_attention_heads, final_sparsity = sparsify_attention_heads(
            full_attention_heads, None, sparsity
        )


        print(f"Computed DuoAttention sparsity: {final_sparsity}")
        print(f"Sparsified attention heads: {full_attention_heads}")

        # 将 list 转为字符串
        # retrieval_heads_str = ",".join(map(str, full_attention_heads))
        retrieval_heads_str = ";".join(
            ",".join(str(int(val)) for val in layer)
            for layer in full_attention_heads
        )
        print(f"Retrieval heads: {retrieval_heads_str}")

        print(f"BASE_URL: {BASE_URL}")

        # 准备启动参数
        other_args = [
            "--tp", "1",
            "--trust-remote-code",
            "--context-length", "32000",

            # DuoAttention Args
            # "--enable-duo-attention",
            # "--duo-attn-sink-size", str(sink_size),
            # "--duo-attn-streaming-window", "128", # 注意：这里你保留了硬编码 128
            # "--duo-attn-retrieval-idx", retrieval_heads_str,
            # "--duo-attn-streaming-idx", str(recent_size),
        ]

        # ==========================================
        # 2. 启动 Server
        # ==========================================
        print("Launching SGLang server...")
        server_process = popen_launch_server(
            LLAMA_LONG_MODEL_PATH,
            BASE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        print(f"Server launched successfully at {BASE_URL}")


        needle_content = (
            "DuoAttention is an efficient framework for long-context LLM inference. "
            "It observes that not all attention heads need full history. "
            "It splits heads into Retrieval Heads (keeping full KV cache) and "
            "Streaming Heads (keeping only recent tokens), significantly reducing memory usage."
        )

        context = "A quick brown fox jumps over the lazy dog. \n"
        
        target_total_len = 10000
        insertion_point = 0.5
        
        len_context_tokens = len(context) / 4 
        num_repetitions = int(target_total_len / len_context_tokens)
        
        prefix = "This is a very long story book: <book> "
        
        pre_context = context * int(num_repetitions * insertion_point)
        post_context = context * int(num_repetitions * (1 - insertion_point))
        
        suffix = (
            "</book>\n Based on the content of the book, please briefly tell me about DuoAttention.\nAnswer:"
        )

        prompt = prefix + pre_context + needle_content + post_context + suffix

        print(f"Constructed prompt length: {len(prompt)} chars (approx {len(prompt)/4:.0f} tokens).")

        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 128,
                "stop": ["\n", "</s>"]
            }
        }

        print("Sending request to model...")
        start_time = time.time()
        response = requests.post(BASE_URL + "/generate", json=payload).json()
        end_time = time.time()


        output_text = response["text"]
        
        print("\n" + "="*40)
        print(f"Model Output:\n{output_text}")
        print("="*40)
        print(f"Latency: {end_time - start_time:.2f} seconds")

        if "Retrieval Heads" in output_text and "Streaming Heads" in output_text:
            print("\nTest Passed: Keywords found in output.")
        else:
            print("\nTest Failed: Keywords NOT found in output.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    
    finally:
        if server_process:
            print("\nShutting down server...")
            kill_process_tree(server_process.pid)
            print("Server stopped.")

if __name__ == "__main__":
    main()