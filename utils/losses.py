import torch
from torch import nn
from torch.nn.functional import max_pool3d
import torch.nn.functional as F
import numpy as np
import math

def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])
    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
    return (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0

def app_gradient_loss(mask, s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])
    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
    return torch.mean(mask * (dx + dy + dz)) / 3.0

def ncc_loss(I, J, win=None):
    ndims = len(I.size()) - 2
    assert ndims in (1, 2, 3)
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
    cc = cross * cross / (I_var * J_var + 1e-5)
    return 1 - torch.mean(cc)

def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J
    if I.dim() == 4:
        conv = F.conv2d
    elif I.dim() == 5:
        conv = F.conv3d
    else:
        raise ValueError
    I_sum = conv(I, filt, stride=stride, padding=padding)
    J_sum = conv(J, filt, stride=stride, padding=padding)
    I2_sum = conv(I2, filt, stride=stride, padding=padding)
    J2_sum = conv(J2, filt, stride=stride, padding=padding)
    IJ_sum = conv(IJ, filt, stride=stride, padding=padding)
    win_size = int(np.prod(win))
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross

def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    if y_true.ndim == 2 and y_pred.ndim == 2:
        y_true = y_true.unsqueeze(0).unsqueeze(0)
        y_pred = y_pred.unsqueeze(0).unsqueeze(0)
    elif y_true.ndim == 3 and y_pred.ndim == 3:
        y_true = y_true.unsqueeze(0)
        y_pred = y_pred.unsqueeze(0)

    dims = tuple(range(2, y_true.ndim))
    intersection = torch.sum(y_true * y_pred, dim=dims)
    denom = torch.sum(y_true**2 + y_pred**2, dim=dims)
    return (2 * intersection + smooth) / (denom + smooth)

def dice_loss(y_true, y_pred, binary=False, background_weight=0.0, smooth_nr=1e-6, smooth_dr=1e-6):
    return 1 - dice_coef(y_true, y_pred, binary=binary, background_weight=background_weight, smooth_nr=smooth_nr, smooth_dr=smooth_dr)

def att_dice(y_true, y_pred):
    dice = dice_coef(y_true, y_pred).detach()
    return (1 - dice) ** 2 * (1 - dice)

def masked_dice_loss(y_true, y_pred, mask):
    smooth = 1.
    a = torch.sum(y_true * y_pred * mask, (2, 3, 4))
    b = torch.sum((y_true + y_pred) * mask, (2, 3, 4))
    dice = (2 * a) / (b + smooth)
    return 1 - torch.mean(dice)

def MSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def partical_MAE(y_true, y_pred, mask, Lambda=0.5):
    return torch.mean(torch.abs(y_true - y_pred) * mask) * Lambda

def mix_ce_dice(y_true, y_pred):
    return crossentropy(y_true, y_pred) + 1 - dice_coef(y_true, y_pred)

def crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred + smooth))

def mask_crossentropy(y_pred, y_true, mask):
    smooth = 1e-6
    return -torch.mean(mask * y_true * torch.log(y_pred + smooth))

def B_crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred + smooth) + (1 - y_true) * torch.log(1 - y_pred + smooth))