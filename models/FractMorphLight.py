import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from functools import reduce
from operator import mul
from einops import rearrange
from modules.frft3d import FrFT3DModule
from utils.STN import SpatialTransformer, Re_SpatialTransformer


frft_module_45 = FrFT3DModule(order=0.5, log_output=False)
frft_module_90 = FrFT3DModule(order=1.0, log_output=False)


class ConvLayers(nn.Module):
    """3D convolution followed by ReLU (for FrFT0 branches)"""
    def __init__(self, embed_dim):
        super().__init__()
        self.conv = nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class ConvComplex1x1Layers(nn.Module):
    """1×1 complex-valued convolution (for FrFT45 and FrFT90 branches)."""
    def __init__(self, embed_dim):
        super().__init__()
        self.conv = nn.Conv3d(2 * embed_dim, 2 * embed_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class ConvLog1x1(nn.Module):
    """1×1 convolution (for log-magnitude branches)."""
    def __init__(self, embed_dim):
        super().__init__()
        self.conv = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class Mlp(nn.Module):
    """Feed-forward MLP: Linear → Activation → Dropout → Linear → Dropout."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class PatchMerging(nn.Module):
    """Downsample by 2× via strided 3D convolution."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.down = nn.Conv3d(dim, 2 * dim, kernel_size=2, stride=2)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        B, D, H, W, C = x.shape
        pad = [(D % 2), (H % 2), (W % 2)]
        if any(pad):
            x = rearrange(x, 'b d h w c -> b c d h w')
            x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
            x = rearrange(x, 'b c d h w -> b d h w c')
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.down(x)
        x = rearrange(x, 'b c d h w -> b d h w c')
        return self.norm(x)


class PatchExpand(nn.Module):
    """Upsample by 2× via transposed 3D convolution."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.up = nn.ConvTranspose3d(dim, dim // 2, kernel_size=2, stride=2)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.up(x)
        x = rearrange(x, 'b c d h w -> b d h w c')
        return self.norm(x)


class PatchEmbed3D(nn.Module):
    """3D patch embedding via non-overlapping conv."""
    def __init__(self, patch_size=(4, 4, 4), in_chans=1, embed_dim=48, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, C, D, H, W = x.shape
        pd = (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0]
        ph = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
        pw = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd))
        x = self.proj(x)
        if self.norm:
            Dp, Hp, Wp = x.shape[2:]
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(B, -1, Dp, Hp, Wp)
        return x


class CrossAttention3D(nn.Module):
    """Multi-head attention for 3D patches."""
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y=None):
        if y is None:
            y = x
        B, N, C = x.shape
        _, M, _ = y.shape

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(B, N,   self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = k.view(B, M,   self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = v.view(B, M,   self.num_heads, C // self.num_heads).permute(0,2,1,3)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )

        out = out.permute(0,2,1,3).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FractionalCrossAttentionModule(nn.Module):
    """3D transformer FCA module with FrFT-based branches and cross-attention."""
    def __init__(self, dim, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_checkpoint=False, num_heads_attn=4):
        super().__init__()
        p = dim // 3
        r = dim - 2*p
        self.split_sizes = (p, p, r)
        self.use_checkpoint = use_checkpoint

        self.norm1_x  = norm_layer(dim)
        self.norm1_xa = norm_layer(dim)
        self.conv0    = ConvLayers(p)
        self.frft45   = frft_module_45
        self.conv45   = ConvComplex1x1Layers(p)
        self.frft90   = frft_module_90
        self.conv90   = ConvComplex1x1Layers(r)
        self.conv_log = ConvLog1x1(r)
        self.conv_fuse= nn.Sequential(nn.Conv3d(2*r, r, 1), nn.ReLU(inplace=True))

        self.norm_pre_attn = norm_layer(dim)
        self.attn_x  = CrossAttention3D(dim, num_heads_attn, qkv_bias, attn_drop, drop)
        self.attn_xa = CrossAttention3D(dim, num_heads_attn, qkv_bias, attn_drop, drop)

        mlp_hidden    = int(dim * mlp_ratio)
        self.norm2_x  = norm_layer(dim)
        self.norm2_xa = norm_layer(dim)
        self.mlp_x    = Mlp(dim, mlp_hidden, dim, act_layer, drop)
        self.mlp_xa   = Mlp(dim, mlp_hidden, dim, act_layer, drop)

        self.drop_path = nn.Identity()

    @staticmethod
    def _stack_real_imag(z):
        real_imag = torch.view_as_real(z).permute(0,1,5,2,3,4)
        return real_imag.reshape(z.size(0), 2*z.size(1), *z.shape[2:])

    @staticmethod
    def _unstk_to_complex(pair):
        real, imag = pair.chunk(2, dim=1)
        return torch.complex(real, imag)

    def _to_channels_first(self, x):
        if x.dim()!=5:
            raise ValueError(f"Expected 5D, got {x.shape}")
        if x.shape[1]==self.norm1_x.normalized_shape[0]:
            return x
        if x.shape[-1]==self.norm1_x.normalized_shape[0]:
            return x.permute(0,4,1,2,3).contiguous()
        raise ValueError(f"Cannot find channel axis in {x.shape}")

    def _compute_reduced_features(self, x):
        x5 = self._to_channels_first(x)
        p,_,r = self.split_sizes
        x0     = x5[:,:p]
        x45    = x5[:,p:2*p]
        x90_in = x5[:,2*p:]

        out0   = self.conv0(x0)
        z45    = self.frft45.FrFT3D(x45)
        c45    = self.conv45(self._stack_real_imag(z45))
        out45  = self.frft45.IFrFT3D(self._unstk_to_complex(c45)).real

        z90    = self.frft90.FrFT3D(x90_in)
        c90    = self.conv90(self._stack_real_imag(z90))
        out90  = self.frft90.IFrFT3D(self._unstk_to_complex(c90)).real

        logm   = self.conv_log(torch.log1p(torch.abs(z90)))
        mag    = torch.expm1(logm)
        z_log  = torch.polar(mag, torch.angle(z90))
        outlog = self.frft90.IFrFT3D(z_log).real

        fused  = self.conv_fuse(torch.cat([out90, outlog], dim=1))
        out    = torch.cat([out0, out45, fused], dim=1)
        out.add_(x5)
        return out

    def forward_part1(self, x, xa):
        x_n  = self.norm1_x(x);  xa_n  = self.norm1_xa(xa)
        rx   = self._compute_reduced_features(x_n)
        rxa  = self._compute_reduced_features(xa_n)
        tx   = rearrange(rx,  'b c d h w -> b (d h w) c')
        txa  = rearrange(rxa, 'b c d h w -> b (d h w) c')
        tx   = self.norm_pre_attn(tx);  txa = self.norm_pre_attn(txa)
        ax   = self.attn_x(tx, txa)
        axa  = self.attn_xa(txa, tx)
        B,_,C = ax.shape;  D,H,W = rx.shape[2:]
        ox   = ax.view(B,D,H,W,C);  oxa = axa.view(B,D,H,W,C)
        return ox, oxa

    def forward_part2(self, x, xa):
        mx  = self.mlp_x(self.norm2_x(x))
        mxa = self.mlp_xa(self.norm2_xa(xa))
        return mx, mxa

    def forward(self, x, xa):
        sc_x, sc_xa = x, xa
        if self.use_checkpoint:
            ax, axa = checkpoint.checkpoint(self.forward_part1, x, xa)
        else:
            ax, axa = self.forward_part1(x, xa)
        x  = sc_x  + self.drop_path(ax)
        xa = sc_xa + self.drop_path(axa)

        sc2_x, sc2_xa = x, xa
        if self.use_checkpoint:
            mx, mxa = checkpoint.checkpoint(self.forward_part2, x, xa)
        else:
            mx, mxa = self.forward_part2(x, xa)
        x  = sc2_x  + self.drop_path(mx)
        xa = sc2_xa + self.drop_path(mxa)
        return x, xa


class FractBasicLayer(nn.Module):
    """Stack of FractionalCrossAttentionModule at one level."""
    def __init__(self, dim, depth, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, num_heads_attn=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            FractionalCrossAttentionModule(
                dim=dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                num_heads_attn=num_heads_attn
            )
            for i in range(depth)
        ])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, xa):
        for blk in self.blocks:
            x, xa = blk(x, xa)
        if self.downsample:
            return x, xa, self.downsample(x), self.downsample(xa)
        return x, xa, x, xa


class FractMorph(nn.Module):
    """Encoder–decoder transformer consisting of FCA blocks."""
    def __init__(self, patch_size=(4,4,4), in_chans=1, embed_dim=48,
                 depths=[2,2,4,2], num_heads=[3,6,12,24], mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                 patch_norm=False, use_checkpoint=False):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        self.pos_drop = nn.Dropout(drop_rate)

        total = sum(depths)
        dpr   = list(torch.linspace(0, drop_path_rate, total).numpy())
        self.layers = nn.ModuleList()
        cur = 0
        for i, d in enumerate(depths):
            self.layers.append(
                FractBasicLayer(
                    dim=embed_dim * 2**i, depth=d,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[cur:cur+d],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if i < len(depths)-1 else None,
                    use_checkpoint=use_checkpoint,
                    num_heads_attn=num_heads[i]
                )
            )
            cur += d

        self.up_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i in reversed(range(len(depths))):
            cat_dim = 2 * embed_dim * 2**i
            self.concat_back_dim.append(nn.Linear(cat_dim, embed_dim * 2**i))
            self.up_layers.append(
                FractBasicLayer(
                    dim=embed_dim * 2**i, depth=depths[i],
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[cur-depths[i]:cur],
                    norm_layer=norm_layer,
                    downsample=PatchExpand if i>0 else None,
                    use_checkpoint=use_checkpoint,
                    num_heads_attn=num_heads[i]
                )
            )
            cur -= depths[i]

        self.norm  = norm_layer(embed_dim * 2**(len(depths)-1))
        self.norm2 = norm_layer(embed_dim * 2)
        self.rev_pe = nn.ConvTranspose3d(embed_dim * 2, embed_dim//2,
                                         kernel_size=patch_size,
                                         stride=patch_size)

    def forward(self, moving, fixed):
        mv = self.patch_embed(moving); fx = self.patch_embed(fixed)
        mv = self.pos_drop(mv); fx = self.pos_drop(fx)

        if mv.dim()==5:
            mv = rearrange(mv,'b c d h w->b d h w c')
            fx = rearrange(fx,'b c d h w->b d h w c')
        else:
            mv = mv.unsqueeze(1).permute(0,2,3,4,1)
            fx = fx.unsqueeze(1).permute(0,2,3,4,1)

        feats_m, feats_f = [], []
        for layer in self.layers:
            o_m, o_f, mv, fx = layer(mv, fx)
            feats_m.append(o_m); feats_f.append(o_f)
        mv = self.norm(mv); fx = self.norm(fx)

        for idx, up in enumerate(self.up_layers):
            rev = len(self.up_layers)-1-idx
            if idx==0:
                _, _, mv, fx = up(mv, fx)
            else:
                sm, sf = feats_m[rev], feats_f[rev]
                if mv.shape[:4]!=sm.shape[:4]:
                    mv = rearrange(F.interpolate(
                        rearrange(mv,'b d h w c->b c d h w'),
                        size=sm.shape[1:4], mode='trilinear', align_corners=True
                    ), 'b c d h w->b d h w c')
                    fx = rearrange(F.interpolate(
                        rearrange(fx,'b d h w c->b c d h w'),
                        size=sf.shape[1:4], mode='trilinear', align_corners=True
                    ), 'b c d h w->b d h w c')
                mv = self.concat_back_dim[idx](torch.cat([mv, sm], dim=-1))
                fx = self.concat_back_dim[idx](torch.cat([fx, sf], dim=-1))
                _, _, mv, fx = up(mv, fx)

        x = torch.cat([mv, fx], dim=-1)
        x = self.norm2(x)
        x = rearrange(x,'b d h w c->b c d h w')
        return self.rev_pe(x)


class UNet(nn.Module):
    """3D U-Net style network for predicting flow field from feature volume."""
    def __init__(self, in_channels, out_channels, base_channels=32):
        super().__init__()
        # Encoder
        self.down1 = nn.Conv3d(in_channels, base_channels, 3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.down2 = nn.Conv3d(base_channels, base_channels*2, 3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.down3 = nn.Conv3d(base_channels*2, base_channels*4, 3, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        # Bottleneck
        self.bottleneck = nn.Conv3d(base_channels*4, base_channels*4, 3, padding=1)
        self.relu_bn = nn.ReLU(inplace=True)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_channels*4, base_channels*2, 2, stride=2)
        self.conv_up3 = nn.Conv3d(base_channels*4, base_channels*2, 3, padding=1)
        self.relu_up3 = nn.ReLU(inplace=True)

        self.up2 = nn.ConvTranspose3d(base_channels*2, base_channels, 2, stride=2)
        self.conv_up2 = nn.Conv3d(base_channels*2, base_channels, 3, padding=1)
        self.relu_up2 = nn.ReLU(inplace=True)

        self.up1 = nn.ConvTranspose3d(base_channels, base_channels, 2, stride=2)
        self.conv_up1 = nn.Conv3d(base_channels + in_channels, base_channels, 3, padding=1)
        self.relu_up1 = nn.ReLU(inplace=True)

        self.out_conv = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, x):
        x1 = x
        x2 = self.relu1(self.down1(x1))
        x3 = self.relu2(self.down2(x2))
        x4 = self.relu3(self.down3(x3))

        x5 = self.relu_bn(self.bottleneck(x4))

        x6 = self.up3(x5)
        x6 = self.relu_up3(self.conv_up3(torch.cat([x6, x3], dim=1)))

        x7 = self.up2(x6)
        x7 = self.relu_up2(self.conv_up2(torch.cat([x7, x2], dim=1)))

        x8 = self.up1(x7)
        x8 = self.relu_up1(self.conv_up1(torch.cat([x8, x1], dim=1)))

        return self.out_conv(x8)


class Head(nn.Module):
    """Full registration head for FractMorph-Light: FractMorph Transformer → UNet → Spatial Transformers."""
    def __init__(self, n_channels=1):
        super().__init__()
        self.fract = FractMorph(in_chans=n_channels, embed_dim=48, depths=[2,2,4,2])
        self.unet  = UNet(in_channels=48//2, out_channels=3, base_channels=32)
        self.stn   = SpatialTransformer()
        self.rstn  = Re_SpatialTransformer()

    def forward(self, moving, fixed, mov_label=None, fix_label=None):
        feats = self.fract(fixed, moving)
        flow  = self.unet(feats)
        w_m2f = self.stn(moving, flow)
        w_f2m = self.rstn(fixed, flow)
        w_lbl_m2f = self.stn(mov_label, flow, mode='nearest') if mov_label is not None else None
        w_lbl_f2m = self.rstn(fix_label, flow, mode='nearest') if fix_label is not None else None
        return w_m2f, w_f2m, w_lbl_m2f, w_lbl_f2m, flow
