import torch
import torch.distributed as dist
import os
import subprocess
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def pairwise_exchange(rank, world_size, tensor_size, src, dst):
    tensor = torch.rand(tensor_size).cuda(rank)
    # Warm-up
    if rank == src or rank == dst:
        for _ in range(10):
            if rank == src:
                dist.send(tensor=tensor, dst=dst)
            elif rank == dst:
                dist.recv(tensor=tensor, src=src)
        torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    if rank == src or rank == dst:
        for _ in range(100):
            if rank == src:
                dist.send(tensor=tensor, dst=dst)
            elif rank == dst:
                dist.recv(tensor=tensor, src=src)
    end_event.record()
    torch.cuda.synchronize()
    
    if rank == src:
        elapsed_time_ms = start_event.elapsed_time(end_event)
        data_transferred = tensor_size * 100 * 4  # size * iterations * bytes/element
        bandwidth = data_transferred / elapsed_time_ms / 1e6  # GB/s
        print(f"Bandwidth from GPU {src} to GPU {dst}: {bandwidth} GB/s")

def test_pairwise_bandwidth(rank, world_size, tensor_size):
    setup(rank, world_size)
    for i in range(world_size):
        for j in range(i + 1, world_size):
            pairwise_exchange(rank, world_size, tensor_size, i, j)
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    tensor_size = 1024 * 1024 * 256  # 256 MB
    mp.spawn(test_pairwise_bandwidth, args=(world_size, tensor_size), nprocs=world_size)