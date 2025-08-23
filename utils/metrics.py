import torch
import numpy as np
import math
import torch.nn.functional as F
from losses import compute_local_sums

def dice_ACDC(img_gt, img_pred, voxel_size=None):
    if img_gt.ndim != img_pred.ndim:
        raise ValueError
    res = []
    for c in [3, 2, 4, 1, 5]:
        gt_c_i = np.copy(img_gt)
        if c == 4:
            gt_c_i[gt_c_i == 2] = c
            gt_c_i[gt_c_i == 3] = c
        elif c == 5:
            gt_c_i[gt_c_i > 0] = c
        gt_c_i[gt_c_i != c] = 0
        pred_c_i = np.copy(img_pred)
        if c == 4:
            pred_c_i[pred_c_i == 2] = c
            pred_c_i[pred_c_i == 3] = c
        elif c == 5:
            pred_c_i[pred_c_i > 0] = c
        pred_c_i[pred_c_i != c] = 0
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)
        top = 2 * np.sum(np.logical_and(pred_c_i, gt_c_i))
        bottom = np.sum(pred_c_i) + np.sum(gt_c_i)
        bottom = max(bottom, np.finfo(float).eps)
        dice = top / bottom
        if voxel_size is not None:
            volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
            volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.
        else:
            volpred, volgt = 0, 0
        res += [dice, volpred, volgt, volpred - volgt]
    return res

def ensure_proper_dims(img, desired_dim):
    if img.dim() == desired_dim:
        return img
    elif img.dim() == desired_dim - 2:
        return img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == desired_dim - 1:
        return img.unsqueeze(0)
    else:
        raise ValueError

def prepare_images(img1, img2):
    desired_dim = max(img1.dim(), img2.dim())
    if desired_dim < 4:
        desired_dim = 4
    if desired_dim > 5:
        raise ValueError
    return ensure_proper_dims(img1, desired_dim), ensure_proper_dims(img2, desired_dim), desired_dim

def ssim_metric(img1, img2, window_size=11, data_range=1.0, size_average=True):
    img1, img2, desired_dim = prepare_images(img1, img2)
    if desired_dim == 4:
        conv, dims = F.conv2d, 2
    elif desired_dim == 5:
        conv, dims = F.conv3d, 3
    else:
        raise ValueError
    padding = window_size // 2
    channel = img1.size(1)
    window = torch.ones((channel, 1, *([window_size] * dims)), dtype=img1.dtype, device=img1.device) / (window_size ** dims)
    mu1 = conv(img1, window, padding=padding, groups=channel)
    mu2 = conv(img2, window, padding=padding, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = conv(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = conv(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = conv(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map

def psnr_metric(img1, img2, data_range=1.0):
    img1, img2, _ = prepare_images(img1, img2)
    mse = torch.mean((img1 - img2) ** 2)
    return torch.tensor(float('inf'), device=img1.device) if mse == 0 else 10 * torch.log10((data_range ** 2) / mse)

def ncc_metric(I, J, win=None):
    I, J, desired_dim = prepare_images(I, J)
    ndims = desired_dim - 2
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win], device=I.device)
    pad_no = math.floor(win[0] / 2)
    if ndims == 1:
        stride, padding = 1, pad_no
    elif ndims == 2:
        stride, padding = (1, 1), (pad_no, pad_no)
    else:
        stride, padding = (1, 1, 1), (pad_no, pad_no, pad_no)
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    return torch.mean(cross * cross / (I_var * J_var + 1e-5))

def ensure_flow_dims(flow):
    if flow.dim() == 5:
        return flow
    elif flow.dim() == 4:
        return flow if flow.size(1) in (2, 3) else flow.unsqueeze(0)
    elif flow.dim() in (2, 3):
        return flow.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError

def jacobian_determinant(flow):
    flow = ensure_flow_dims(flow)
    if flow.size(2) == 1:
        flow = flow.squeeze(2)
    if flow.dim() == 4:
        dx = F.pad(flow[:, :, :, 1:] - flow[:, :, :, :-1], (0, 1, 0, 0), mode='replicate')
        dy = F.pad(flow[:, :, 1:, :] - flow[:, :, :-1, :], (0, 0, 0, 1), mode='replicate')
        J00 = 1 + dx[:, 0]
        J01 = dy[:, 0]
        J10 = dx[:, 1]
        J11 = 1 + dy[:, 1]
        return J00 * J11 - J01 * J10
    elif flow.dim() == 5:
        dx = F.pad(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0), mode='replicate')
        dy = F.pad(flow[:, :, :, 1:, :] - flow[:, :, :-1, :, :], (0, 0, 0, 1, 0, 0), mode='replicate')
        dz = F.pad(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1), mode='replicate')
        dux_dx, dux_dy, dux_dz = dx[:, 0], dy[:, 0], dz[:, 0]
        duy_dx, duy_dy, duy_dz = dx[:, 1], dy[:, 1], dz[:, 1]
        duz_dx, duz_dy, duz_dz = dx[:, 2], dy[:, 2], dz[:, 2]
        J00 = 1 + dux_dx; J01 = dux_dy; J02 = dux_dz
        J10 = duy_dx;   J11 = 1 + duy_dy; J12 = duy_dz
        J20 = duz_dx;   J21 = duz_dy;   J22 = 1 + duz_dz
        return J00 * (J11 * J22 - J12 * J21) - J01 * (J10 * J22 - J12 * J20) + J02 * (J10 * J21 - J11 * J20)
    else:
        raise ValueError

def non_positive_jacobian_percentage(flow):
    detJ = jacobian_determinant(flow)
    return 100.0 * (detJ <= 0).sum().item() / detJ.numel()

def std_jacobian(flow):
    return torch.std(torch.abs(jacobian_determinant(flow))).item()
