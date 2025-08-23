import torch
import torch.nn as nn
import torch.nn.functional as nnf

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, src, flow, mode='bilinear', padding_mode='zeros'):
        if src.dim() == 3:
            src = src.unsqueeze(0)
        if src.dim() == 4:
            src = src.unsqueeze(0)

        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)

        if torch.cuda.is_available():
            grid = grid.cuda()

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]
            
        # print(f"SpatialTransformer - src: {src.shape}, flow: {flow.shape}, new_locs: {new_locs.shape}")
        src = src.cuda()
        new_locs = new_locs.cuda()
        return nnf.grid_sample(src, new_locs, mode=mode, padding_mode=padding_mode)


class Re_SpatialTransformer(nn.Module):
    def __init__(self):
        super(Re_SpatialTransformer, self).__init__()
        self.stn = SpatialTransformer()

    def forward(self, src, flow, mode='bilinear'):
        flow = -1 * self.stn(flow, flow, mode='bilinear')

        return self.stn(src, flow, mode)