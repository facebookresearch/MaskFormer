import os
import subprocess
import numpy as np

def assign_free_gpus(max_gpus=1):
    # Get info about each of the available GPUs
    smi_query_result = subprocess.check_output('nvidia-smi -q -d Memory | grep -A4 GPU', shell=True)
    # Extract the usage information
    gpu_info = smi_query_result.decode('utf-8').split('\n')
    total_mem_info = list(filter(lambda info: 'Total' in info, gpu_info))
    total_mem_info = [int(x.split(':')[1].replace('MiB', '').strip()) for x in total_mem_info]
    used_mem_info = list(filter(lambda info: 'Used' in info, gpu_info))
    used_mem_info = [int(x.split(':')[1].replace('MiB', '').strip()) for x in used_mem_info]
    available_mem_info = np.subtract(total_mem_info, used_mem_info)
    if max_gpus > len(available_mem_info):
        print("Only {:.0f} GPU's available. Lowering max_gpus from {:.0f} to {:.0f}".
            format(len(available_mem_info), max_gpus, len(available_mem_info)))
        max_gpus = len(available_mem_info)
    gpus_to_use = np.sort(np.flip(np.argsort(available_mem_info))[:max_gpus])
    gpus_to_use = ",".join([str(x) for x in gpus_to_use])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
    print("Using GPU(s): {}".format(gpus_to_use if gpus_to_use else "No available GPU found"))
