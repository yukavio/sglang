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
from sglang.test.attention.duoattention.utils import (
    load_attn_pattern, 
    sparsify_attention_heads,
    count_attention_head_types
)


ROOT_PATH = "/mnt/cephfs/chengqi/"
LLAMA_LONG_MODEL_PATH = os.path.join(ROOT_PATH, "models/Llama-3-8B-Instruct-Gradient-1048k")
ATTN_PATTERN_PATH = os.path.join(ROOT_PATH, "attn_patterns/Llama-3-8B-Instruct-Gradient-4194k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10")
BASE_URL = DEFAULT_URL_FOR_TEST

def test_simple_generation(base_url):
    """
    简单的模型生成能力测试，包含自动验证逻辑
    """
    print("\n" + "="*60)
    print("Running Simple Generation Test...")
    print("="*60)
    
    # 测试样例及其验证标准
    test_prompts = [
        {
            "text": "Please write a short poem about artificial intelligence:",
            "description": "创意写作测试",
            "validation": {
                "min_length": 20,  # 最少字符数
                "keywords": ["AI", "artificial", "intelligence", "computer", "machine", "future", "technology"],  # 期望关键词（至少包含一个）
                "check_type": "creative"
            }
        },
        {
            "text": "Q: What is the capital of France?\nA:",
            "description": "知识问答测试",
            "validation": {
                "min_length": 3,
                "keywords": ["Paris", "paris"],  # 期望答案
                "check_type": "factual"
            }
        },
        {
            "text": "Translate the following English to Chinese: 'Hello, how are you?'\nTranslation:",
            "description": "翻译能力测试",
            "validation": {
                "min_length": 3,
                "keywords": ["你好", "您好", "好吗", "怎么样"],  # 期望中文翻译
                "check_type": "translation"
            }
        }
    ]
    
    test_results = []
    
    for i, test_case in enumerate(test_prompts, 1):
        print(f"\n--- Test Case {i}: {test_case['description']} ---")
        print(f"Prompt: {test_case['text']}")
        
        payload = {
            "text": test_case["text"],
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 100,
            }
        }
        
        try:
            start_time = time.time()
            response = requests.post(base_url + "/generate", json=payload, timeout=30).json()
            end_time = time.time()
            
            output_text = response["text"]
            latency = end_time - start_time
            
            print(f"Output: {output_text}")
            print(f"Latency: {latency:.2f} seconds")
            
            # 自动验证
            validation = test_case["validation"]
            passed = True
            validation_msgs = []
            
            # 1. 检查输出长度
            if len(output_text) < validation["min_length"]:
                passed = False
                validation_msgs.append(f"❌ 输出太短 (长度: {len(output_text)}, 期望: >={validation['min_length']})")
            else:
                validation_msgs.append(f"✓ 输出长度合格 ({len(output_text)} 字符)")
            
            # 2. 检查关键词
            keyword_found = any(keyword.lower() in output_text.lower() for keyword in validation["keywords"])
            if keyword_found:
                matched_keywords = [kw for kw in validation["keywords"] if kw.lower() in output_text.lower()]
                validation_msgs.append(f"✓ 包含期望关键词: {matched_keywords}")
            else:
                passed = False
                validation_msgs.append(f"❌ 未包含期望关键词 (期望: {validation['keywords']})")
            
            # 3. 检查是否为空或只有空格
            if not output_text.strip():
                passed = False
                validation_msgs.append("❌ 输出为空")
            else:
                validation_msgs.append("✓ 输出非空")
            
            # 4. 检查延迟
            if latency > 10:
                validation_msgs.append(f"⚠️  延迟较高: {latency:.2f}秒")
            else:
                validation_msgs.append(f"✓ 延迟正常: {latency:.2f}秒")
            
            # 输出验证结果
            print("\n验证结果:")
            for msg in validation_msgs:
                print(f"  {msg}")
            
            if passed:
                print(f"✅ Test Case {i} PASSED")
            else:
                print(f"❌ Test Case {i} FAILED")
            
            test_results.append({
                "case": i,
                "description": test_case["description"],
                "passed": passed,
                "latency": latency
            })
            
            print("-" * 50)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            print("-" * 50)
            test_results.append({
                "case": i,
                "description": test_case["description"],
                "passed": False,
                "error": str(e)
            })
    
    # 输出总结
    print("\n" + "="*60)
    print("测试总结:")
    print("="*60)
    passed_count = sum(1 for r in test_results if r["passed"])
    total_count = len(test_results)
    print(f"通过: {passed_count}/{total_count}")
    
    for result in test_results:
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        print(f"  Test {result['case']} ({result['description']}): {status}")
    
    if passed_count == total_count:
        print("\n🎉 所有测试通过！模型运行正常。")
    else:
        print(f"\n⚠️  有 {total_count - passed_count} 个测试失败，请检查模型输出。")
    
    print("\nSimple Generation Test Completed!\n")
    
    return test_results

def test_needle_in_haystack(base_url):
    """
    Needle-in-Haystack 测试
    """
    print("\n" + "="*60)
    print("Running Needle-in-Haystack Test...")
    print("="*60)
    

    needle_content = (
        "DuoAttention is an efficient framework for long-context LLM inference. "
        "It observes that not all attention heads need full history. "
        "It splits heads into Retrieval Heads (keeping full KV cache) and "
        "Streaming Heads (keeping only recent tokens), significantly reducing memory usage."
    )

    context = "A quick brown fox jumps over the lazy dog. \n"
    
    target_total_len = 1000
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
    response = requests.post(base_url + "/generate", json=payload).json()
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

        full_heads_count, streaming_heads_count = count_attention_head_types(
            full_attention_heads
        )

        print(f"Computed DuoAttention sparsity: {final_sparsity}")
        print(f"Sparsified attention heads: {full_attention_heads}")
        print(f"Full heads count: {full_heads_count}")
        print(f"Streaming heads count: {streaming_heads_count}")

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
            "--enable-duo-attention",
            "--duo-attn-sink-size", str(sink_size),
            "--duo-attn-streaming-window", str(recent_size),
            "--duo-attn-retrieval-idx", retrieval_heads_str,
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

        # ==========================================
        # 3. 运行简单生成测试（验证模型基本能力）
        # ==========================================
        test_simple_generation(BASE_URL)

        # ==========================================
        # 4. 运行 Needle-in-Haystack 测试
        # ==========================================
        test_needle_in_haystack(BASE_URL)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    
    finally:
        if server_process:
            print("\nShutting down server...")
            kill_process_tree(server_process.pid)
            print("Server stopped.")

if __name__ == "__main__":
    main()