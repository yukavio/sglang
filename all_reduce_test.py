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

def test_nccl_bandwidth(rank, world_size, tensor_size):
    setup(rank, world_size)
    tensor = torch.rand(tensor_size).cuda(rank)
    # Warm-up
    for _ in range(10):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        dist.all_reduce(tensor)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    data_transferred = tensor_size * 100 * 4 * 2  # size * iterations * bytes/element * all_reduce factor
    bandwidth = data_transferred / elapsed_time_ms / 1e6  # GB/s
    
    if rank == 0:
        print(f"Bandwidth: {bandwidth} GB/s")
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    tensor_size = 1024 * 1024 * 256  # 256 MB
    mp.spawn(test_nccl_bandwidth, args=(world_size, tensor_size), nprocs=world_size)