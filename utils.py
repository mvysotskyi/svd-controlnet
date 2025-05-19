import math
import torch


def get_stratified_uniform_samples(shape, dtype=None, device=None):
    offsets = torch.arange(0, shape[-1], dtype=dtype, device=device)
    uniform_samples = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + uniform_samples) / shape[-1]


def logsnr_cosine_schedule(t, logsnr_min, logsnr_max):
        t_min_val = math.atan(math.exp(-0.5 * logsnr_max))
        t_max_val = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min_val + t * (t_max_val - t_min_val)))

def logsnr_cosine_schedule_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
    shift = 2 * math.log(noise_d / image_d)
    return logsnr_cosine_schedule(t, logsnr_min - shift, logsnr_max - shift) + shift

def logsnr_cosine_schedule_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
    logsnr_low = logsnr_cosine_schedule_shifted(
        t, image_d, noise_d_low, logsnr_min, logsnr_max)
    logsnr_high = logsnr_cosine_schedule_shifted(
        t, image_d, noise_d_high, logsnr_min, logsnr_max)
    return torch.lerp(logsnr_low, logsnr_high, t)


def get_rand_cosine_interpolated_samples(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_val=1e-3, max_val=1e3, device='cpu', dtype=torch.float32):
    logsnr_min_val = -2 * math.log(min_val / sigma_data)
    logsnr_max_val = -2 * math.log(max_val / sigma_data)
    u_samples = get_stratified_uniform_samples(
        shape, group_idx=0, total_groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_cosine_schedule_interpolated(
        u_samples, image_d, noise_d_low, noise_d_high, logsnr_min_val, logsnr_max_val)
    return torch.exp(-logsnr / 2) * sigma_data