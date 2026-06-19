
import torch
import torch.nn.functional as F

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)

def ssim(pred, target, window_size=11):
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    mu1 = F.avg_pool2d(pred, window_size, 1, window_size//2)
    mu2 = F.avg_pool2d(target, window_size, 1, window_size//2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.avg_pool2d(pred * pred, window_size, 1, window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, 1, window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, 1, window_size//2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
