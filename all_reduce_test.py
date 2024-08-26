#!/usr/bin/env python

import os
import sys
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import time

# Values are 4 bytes each, so 2 * 1000 * 1000 * 32 * 4 = 256 MB = 2048 Mbit
MODEL_SIZE_VALUES = 2 * 1000 * 1000 * 32
BIT_PER_VALUE = 4 * 8
BITS_PER_MBIT = 1000 * 1000


def current_time_in_ms():
    return int(round(time.time() * 1000))


def run(rank, size):
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(MODEL_SIZE_VALUES, dtype=torch.float32)
    print("Performing allreduce...")
    print("   > Data to send: %d Mbit" % ((MODEL_SIZE_VALUES * BIT_PER_VALUE) / float(BITS_PER_MBIT)))
    start = current_time_in_ms()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    elapsed_ms = current_time_in_ms() - start
    print("   > Finished.")
    print("   > Time: %.2f s" % (elapsed_ms / 1000.0))
    print("   > Speed: %.2f Mbit/s" % ((MODEL_SIZE_VALUES * BIT_PER_VALUE / BITS_PER_MBIT) / float(elapsed_ms / 1000.0)))
    print('   > Result: Rank ', rank, ' has data ', str(tensor), '.\n')


def init_process(my_rank, size, master_address, master_port, fn, backend='nccl'):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = master_address
    os.environ['MASTER_PORT'] = master_port

    # Initialize process group
    print("Initializing process group...")
    dist.init_process_group(backend, rank=my_rank, world_size=size)
    print("   > Initialized.")
    print("")
    fn(my_rank, size)


def main(size, master_address, master_port):
    process = []
    for i in range(size):
        p = Process(target=init_process, args=(i, size, master_address, master_port, run))
        process.append(p)
        p.start()
    for i in range(size):
        p.join()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 3:
        print("Usage: python allreduce.py <my rank> <size> <master address> <master port>")
        exit(1)
    else:
        main(int(args[0]), int(args[1]), str(args[2]))