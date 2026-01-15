from .gpu_extension import get_num_gpus, \
    get_gpu_compute_capability, select_gpu_device, \
    get_gpu_memory_mb, get_gpu_name, reset_function_cache

def init_gpus(device = 0):
    n = get_num_gpus()
    if n > 0:
        print("Detected GPUs:")
        for i in range(n):
            comp = get_gpu_compute_capability(i)
            mem = get_gpu_memory_mb(i)
            name = get_gpu_name(i)
            print("{}: {} (Compute Capability: {}, Memory: {:.3f}GB)".format(
                i, name, comp / 10.0, mem /1024.0
            ))
        print("Initialising GPU {}...".format(device))
        select_gpu_device(device)
