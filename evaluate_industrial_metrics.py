import time
import torch
import numpy as np
import psutil
import os
# from torchinfo import summary
import pynvml
import os


# def compute_epe(pred, gt, mask=None):
#     # diff = (pred - gt).flatten()
#     # epe = torch.sqrt(torch.sum(diff ** 2))
#     # if mask is not None:
#     #     epe = epe[mask]
#     # return epe.mean().item()
#     return 0

# def compute_d1(pred, gt, mask=None, threshold=3.0):
#     # diff = (pred - gt).flatten()
#     # epe = torch.sqrt(torch.sum(diff** 2, dim=0))
#     # if mask is not None:
#     #     epe = epe[mask]
#     # err_rate = (epe > threshold).float().mean().item()
#     # return err_rate
#     return 0

def res(input_shape, output_shape):
    input_res = f"{input_shape[-2]}x{input_shape[-1]}"
    output_res = f"{output_shape[-2]}x{output_shape[-1]}" if len(output_shape) >= 2 else str(output_shape)
    return (input_res, output_res)
def compute_avgerr(pred, gt, mask=None):
    """
    计算平均绝对误差（MAE）
    """
    diff = (pred - gt)
    abs_err = torch.abs(diff)
    if mask is not None:
        abs_err = abs_err[mask]
    return abs_err.mean().item()

def compute_rmse(pred, gt, mask=None):
    diff = (pred - gt)
    if mask is not None:
        mse = torch.mean((diff[mask]) ** 2)
    else:
        mse = torch.mean(diff ** 2)
    return torch.sqrt(mse).item()

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return round(size_all_mb, 2)

def get_gpu_memory_global(device_id: int = 0) -> float:
    """
    返回指定 GPU 上所有进程占用的总显存（MB）。
    如果只想看当前 Python 进程，可以配合 os.getpid() 过滤。
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    # 方法 B：只算本进程
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    my_pid = os.getpid()
    for p in procs:
        if p.pid == my_pid:
            used_mb = p.usedGpuMemory / 1024 ** 2
            break
    else:
        used_mb = 0.0   # 本进程没在用显存

    pynvml.nvmlShutdown()
    return round(used_mb, 2)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 ** 2
    return round(mem, 2)

def get_power_usage():
    try:
        output = os.popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits").read()
        power = float(output.strip().split('\n')[0])
        return power
    except Exception:
        return -1
    
def compute_bad(epe_flattened, val, threshold):
    """
    epe_flattened: 每个像素的误差 (1D tensor)
    val: 有效mask (bool tensor)
    threshold: 阈值，比如1.0, 2.0, 3.0
    返回：超过阈值的有效像素比例
    """
    out = (epe_flattened > threshold)
    bad = out[val].float().mean().item()
    return bad
    
def format_flops(flops, unit='GFLOPs'):
    units = {
        'FLOPs': 1,
        'KFLOPs': 1e3,
        'MFLOPs': 1e6,
        'GFLOPs': 1e9,
        'TFLOPs': 1e12,
        'PFLOPs': 1e15,
    }
    if unit not in units:
        unit = 'FLOPs'
    return flops / units[unit], unit

def get_flops(model, tup, device):
    # FLOPs
    # try:
    #     sample = next(iter(dataloader))
    #     dummy1 = sample['image1'].to(device)
    #     dummy2 = sample['image2'].to(device)
    #     macs, params = summary(model, input_data=(dummy1, dummy2), verbose=0)
    #     flops = macs
    # except Exception:
    #     flops = -1
    
    dummy1 = tup[0].to(device)
    dummy2 = tup[1].to(device)
    # stats = summary(model, input_data=(dummy1, dummy2), verbose=0)
    # flops = stats.total_mult_adds

    # flops, macs, params = calculate_flops(
    #     model=model,
    #     args=[dummy1, dummy2],
    #     print_results=False,
    #     output_precision=4,
    # )

    return 0

def evaluate(model, dataloader, device='cuda'):
    model.eval()
    epe_list, d1_list, rmse_list, latency_list = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            image1, image2, gt = batch['image1'].to(device), batch['image2'].to(device), batch['gt'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)
            start = time.time()
            pred = model(image1, image2)
            torch.cuda.synchronize()
            end = time.time()
            latency = end - start
            epe = compute_epe(pred, gt, mask)
            d1 = compute_d1(pred, gt, mask)
            rmse = compute_rmse(pred, gt, mask)
            epe_list.append(epe)
            d1_list.append(d1)
            rmse_list.append(rmse)
            latency_list.append(latency)
    avg_epe = np.mean(epe_list)
    avg_d1 = np.mean(d1_list)
    avg_rmse = np.mean(rmse_list)
    avg_latency = np.mean(latency_list)
    throughput = 1.0 / avg_latency if avg_latency > 0 else 0

    flops = get_flops(model, dataloader, device)

    model_size = get_model_size(model)
    memory_usage = get_memory_usage()
    power_usage = get_power_usage()

    metrics = {
        'EPE': avg_epe,
        'D1': avg_d1,
        'RMSE': avg_rmse,
        'Latency(s)': avg_latency,
        'Throughput(FPS)': throughput,
        'FLOPs': flops,
        'ModelSize(MB)': model_size,
        'Memory(MB)': memory_usage,
        'Power(W)': power_usage
    }
    return metrics

# 用法示例
if __name__ == '__main__':
    # 这里需要你根据实际情况定义 model 和 dataloader
    # model = ...
    # dataloader = ...
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # metrics = evaluate(model, dataloader, device=device)
    # print(metrics)