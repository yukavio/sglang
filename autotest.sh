#!/bin/bash

# 日志文件
LOG_FILE="service_benchmark.log"

# 在日志文件中写入分隔符和当前时间
echo "====================== $(date) ======================" >> "$LOG_FILE"

# 启动服务并将其放到后台，重定向输出到日志文件
/workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B \
    --host 0.0.0.0 --port 8080 --mem-fraction-static 0.8 \
    --dp-size 8 --load-balance-method resources_aware \
    --chunked-prefill-size 2048 --disable-radix-cache >> "$LOG_FILE" 2>&1 &
# 获取服务的进程ID
SERVICE_PID=$!

# 等待一段时间以确保服务启动
sleep 300

# 启动 benchmark 脚本，重定向输出到日志文件

/workspace/bin/micromamba run -n sglang python3 -m sglang.bench_serving --backend sglang \
        --host 127.0.0.1 --port 8080 --dataset-name random \
        --tokenizer Qwen/Qwen2-7B --model Qwen/Qwen2-7B \
        --random-output-len 1024 --random-input-len 4096 \
        --random-range-ratio 0.5 --seed 1234 \
        --num-prompts 20000 --request-rate 16.0 >> "$LOG_FILE" 2>&1

mkdir -p resources_aware_0823
mv token_usage* ./resources_aware_0823

kill -9 $SERVICE_PID
sleep 300

/workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B \
    --host 0.0.0.0 --port 8080 --mem-fraction-static 0.8 \
    --dp-size 8 --load-balance-method power_of_2_choice \
    --chunked-prefill-size 2048 --disable-radix-cache >> "$LOG_FILE" 2>&1 &
# 获取服务的进程ID
SERVICE_PID=$!

# 等待一段时间以确保服务启动
sleep 300

/workspace/bin/micromamba run -n sglang python3 -m sglang.bench_serving --backend sglang \
        --host 127.0.0.1 --port 8080 --dataset-name random \
        --tokenizer Qwen/Qwen2-7B --model Qwen/Qwen2-7B \
        --random-output-len 1024 --random-input-len 4096 \
        --random-range-ratio 0.5 --seed 1234 \
        --num-prompts 20000 --request-rate 16.0 >> "$LOG_FILE" 2>&1

mkdir -p power_of_2_choice_0823
mv token_usage* ./power_of_2_choice_0823