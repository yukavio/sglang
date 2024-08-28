from llama_cpp import Llama
llm = Llama.from_pretrained(
    repo_id="yuhuili/EAGLE-Qwen2-7B-Instruct",
    filename="*.bin",
    verbose=False
)