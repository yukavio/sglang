#!/bin/bash

# 日志文件
LOG_FILE="service_benchmark.log"

# 在日志文件中写入分隔符和当前时间
echo "====================== $(date) ======================" >> "$LOG_FILE"

# 启动服务并将其放到后台，重定向输出到日志文件
echo "Running with setting: dp=8 tp=1 ======================================================"
/workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B \
    --host 0.0.0.0 --port 8080 --mem-fraction-static 0.8 \
    --dp-size 8 --load-balance-method resources_aware \
    --chunked-prefill-size 2048 --disable-radix-cache >> "$LOG_FILE" 2>&1 &


# 等待一段时间以确保服务启动
sleep 300

# 启动 benchmark 脚本，重定向输出到日志文件
# 循环从 7 到 12，步进为 0.5
for rate in $(seq 7 0.5 12); do
    echo "Running with request-rate: $rate" | tee -a "$LOG_FILE"  # 输出当前的 request-rate 值并追加到日志文件
    /workspace/bin/micromamba run -n sglang python3 -m sglang.bench_serving --backend sglang \
            --host 127.0.0.1 --port 8080 --dataset-name random \
            --tokenizer Qwen/Qwen2-7B --model Qwen/Qwen2-7B \
            --random-output-len 1024 --random-input-len 4096 \
            --random-range-ratio 0.5 --seed 1234 \
            --num-prompts 10000 --request-rate $rate >> "$LOG_FILE" 2>&1
    sleep 100
ps -elf | grep python  | awk '{print $4}' | xargs  kill -s 9
sleep 100

#=============================================================================================================================================================


# 启动服务并将其放到后台，重定向输出到日志文件
echo "Running with setting: dp=4 tp=2 ======================================================"
/workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B \
    --host 0.0.0.0 --port 8080 --mem-fraction-static 0.8 \
    --dp-size 4 --tp-size 2 --load-balance-method resources_aware \
    --chunked-prefill-size 2048 --disable-radix-cache >> "$LOG_FILE" 2>&1 &

sleep 300
for rate in $(seq 7 0.5 12); do
    echo "Running with request-rate: $rate" | tee -a "$LOG_FILE"  # 输出当前的 request-rate 值并追加到日志文件
    /workspace/bin/micromamba run -n sglang python3 -m sglang.bench_serving --backend sglang \
            --host 127.0.0.1 --port 8080 --dataset-name random \
            --tokenizer Qwen/Qwen2-7B --model Qwen/Qwen2-7B \
            --random-output-len 1024 --random-input-len 4096 \
            --random-range-ratio 0.5 --seed 1234 \
            --num-prompts 10000 --request-rate $rate >> "$LOG_FILE" 2>&1
    sleep 100
ps -elf | grep python  | awk '{print $4}' | xargs  kill -s 9
sleep 100

#=============================================================================================================================================================

# 启动服务并将其放到后台，重定向输出到日志文件
echo "Running with setting: dp=2 tp=4 ======================================================"

/workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B \
    --host 0.0.0.0 --port 8080 --mem-fraction-static 0.8 \
    --dp-size 2 --tp-size 4 --load-balance-method resources_aware \
    --chunked-prefill-size 2048 --disable-radix-cache >> "$LOG_FILE" 2>&1 &
sleep 300
for rate in $(seq 7 0.5 12); do
    echo "Running with request-rate: $rate" | tee -a "$LOG_FILE"  # 输出当前的 request-rate 值并追加到日志文件
    /workspace/bin/micromamba run -n sglang python3 -m sglang.bench_serving --backend sglang \
                    --host 127.0.0.1 --port 8080 --dataset-name random \
                    --tokenizer Qwen/Qwen2-7B --model Qwen/Qwen2-7B \
                    --random-output-len 1024 --random-input-len 4096 \
                    --random-range-ratio 0.5 --seed 1234 \
                    --num-prompts 10000 --request-rate $rate >> "$LOG_FILE" 2>&1
    sleep 100
ps -elf | grep python  | awk '{print $4}' | xargs  kill -s 9
sleep 100


#=============================================================================================================================================================


echo "Running with setting: dp=1 tp=8 ======================================================"

/workspace/bin/micromamba run -n sglang python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B \
    --host 0.0.0.0 --port 8080 --mem-fraction-static 0.8 \
    --dp-size 1 --tp-size 8 --load-balance-method resources_aware \
    --chunked-prefill-size 2048 --disable-radix-cache >> "$LOG_FILE" 2>&1 &
sleep 300
for rate in $(seq 7 0.5 12); do
    echo "Running with request-rate: $rate" | tee -a "$LOG_FILE"  # 输出当前的 request-rate 值并追加到日志文件
    /workspace/bin/micromamba run -n sglang python3 -m sglang.bench_serving --backend sglang \
                    --host 127.0.0.1 --port 8080 --dataset-name random \
                    --tokenizer Qwen/Qwen2-7B --model Qwen/Qwen2-7B \
                    --random-output-len 1024 --random-input-len 4096 \
                    --random-range-ratio 0.5 --seed 1234 \
                    --num-prompts 10000 --request-rate $rate >> "$LOG_FILE" 2>&1
    sleep 100
ps -elf | grep python  | awk '{print $4}' | xargs  kill -s 9
sleep 100
#=============================================================================================================================================================