import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from einops import rearrange
import os
import numpy as np
import pdb

SPECIALIZED = int(os.environ["SPECIALIZED"]) if "SPECIALIZED" in os.environ else 0
from spatial_correlation_sampler import SpatialCorrelationSampler


if not SPECIALIZED:
    print("Using specialized kernels: NO")
    from dwoconv1d import DepthwiseOrientedConv1d
else:
    print("Using specialized kernels: YES")
    from dwoconv1d_specialized import DepthwiseOrientedConv1d

# Layer-wise angle utilities
def channel_cycle_offset(c, C, N):
    r"""Given N directions and C channels, output the angle for the current channel c.
    We split the angles to form N groups of ~C/N channels. The channels of group n share the same angle theta = n / N * 180 degrees
    """
    n = (N * (c + 1) - 1) // C
    return n / N


def layer_wise_rotation_offset(k, K, enable_layer_cycle=0):
    r"""Layer-wise rotation. Add a 45 degree offset for every layer at odd depth. Enabled / disabled by enable_layer_cycle"""
    if enable_layer_cycle:
        return (k % 2) / 2
    return 0


def theta_offset(x):
    r"""Convert the offset to an angle theta. Kernels shifted by 180 degrees have same angles so only keep angles [0, 180[."""
    return np.pi * (x % 1)


class Wrapper(nn.Module):
    r"""Wrapper"""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        return self.conv(x)


def get_convolution_1d_arbitrary_size(
    in_channels,
    out_channels,
    kernel_size,
    padding,
    groups,
    stride=1,
    N=8,
    layer_offset=0,
    nhwc=False,
    resid=False,
    bias=True,
):
    r"""1D oriented convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for the depthwise convolution.
        padding (int): Padding for the depthwise convolution.
        groups (int): Number of groups for the depthwise convolution.
        stride (int): Stride for the depthwise convolution. Default: 1.
        N (int): Number of directions for oriented kernel. Default: 8.
        layer_offset (float): layer-wise rotation offset. Default: 0.
        nhwc (bool): Use fused implementation for NHWC input. Default: False.
        resid (bool): Use fused implementation for residual connection. Default: False.
        bias (bool): Use bias. Default: True.
    """
    assert in_channels == out_channels == groups
    theta = [
        theta_offset(channel_cycle_offset(k, in_channels, N) + layer_offset)
        for k in range(in_channels)
    ]
    conv = DepthwiseOrientedConv1d(
        in_channels=in_channels,
        out_channels=in_channels,
        groups=in_channels,
        stride=stride,
        nhwc=nhwc,
        resid=resid,
        kernel_size=kernel_size,
        padding=padding,
        bias=bias,
        angle=theta,
        arbitrary_size=True,
    )
    return Wrapper(conv)

class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel=3,
                 mid_channels=None,
                 is_pooling=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.is_pooling = is_pooling

        if in_channels != 1: 
            self.double_conv = nn.Sequential(
                nn.GroupNorm(2, in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels,
                        mid_channels,
                        kernel_size=kernel,
                        padding=kernel // 2),
                nn.GroupNorm(2, mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels,
                        out_channels,
                        kernel_size=kernel,
                        padding=kernel // 2),
            )
            self.single_conv = nn.Sequential(
                nn.GroupNorm(2, in_channels),
                nn.Conv2d(in_channels,
                        out_channels,
                        kernel_size=kernel,
                        padding=kernel // 2)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels,
                        mid_channels,
                        kernel_size=kernel,
                        padding=kernel // 2),
                nn.GroupNorm(2, mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels,
                        out_channels,
                        kernel_size=kernel,
                        padding=kernel // 2),
            )
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_channels,
                        out_channels,
                        kernel_size=kernel,
                        padding=kernel // 2)
            )





    def forward(self, x):
        shortcut = self.single_conv(x)
        x = self.double_conv(x)
        x = x + shortcut

        if self.is_pooling:
            x = F.max_pool2d(x, 2)
        return x


class Down(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel=3):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,
                    out_channels,
                    kernel),)
  


    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bilinear=True,
                 kernel=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv(in_channels,
                                   out_channels,
                                   kernel=kernel,
                                   mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(in_channels,
                                   out_channels,
                                   kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # x = torch.cat([x2, x1], dim=1)
        x = x1 + x2
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OrientedConv(nn.Module):
    def __init__(self, 
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                layer_num=0,
                n_segments=0,
                is_corr=False):
        super().__init__()
        self.n_segments = n_segments
        self.layer_num = layer_num
        self.is_corr = is_corr


        padding = (kernel_size - stride) // 2
        self.dwconv1 = get_convolution_1d_arbitrary_size(
            in_channels, 
            in_channels,
            kernel_size=5,
            stride=stride,
            padding=padding,
            groups=in_channels,
            nhwc=False,
            resid=False,
            layer_offset=0
        )



        self.dwconv2 = get_convolution_1d_arbitrary_size(
            in_channels, 
            in_channels,
            kernel_size=5,
            stride=stride,
            padding=padding,
            groups=in_channels,
            nhwc=False,
            resid=False,
            layer_offset=0
        )

        self.layer_norm = nn.GroupNorm(2, in_channels * 8)
        self.layer_pwconv = nn.Linear(in_channels * 8 , out_channels)

        self.dwconv3 = get_convolution_1d_arbitrary_size(
            in_channels, 
            in_channels,
            kernel_size=5,
            stride=stride,
            padding=padding,
            groups=in_channels,
            nhwc=False,
            resid=False,
            layer_offset=0
        )


        self.dwconv4 = get_convolution_1d_arbitrary_size(
            in_channels, 
            in_channels,
            kernel_size=5,
            stride=stride,
            padding=padding,
            groups=in_channels,
            nhwc=False,
            resid=False,
            layer_offset=0
        )

        self.temporal_norm = nn.GroupNorm(2, in_channels * 8)
        self.temporal_pwconv = nn.Linear(in_channels * 8 , out_channels)

        if self.is_corr:
            self.max_size = 7
            self.corr_module = SpatialCorrelationSampler(kernel_size=1, patch_size=self.max_size, stride=1, padding=0, dilation=1, dilation_patch=1)


            self.vertical = nn.Sequential(
                nn.Conv3d(3, out_channels // 4, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
                nn.GroupNorm(2, in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels // 4, 64, kernel_size=(5, 1, 1), padding=(2, 0, 0))
            )

            self.horizontal = nn.Sequential(
                nn.Conv3d(2, out_channels // 4, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
                nn.GroupNorm(2, in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels // 4, 64, kernel_size=(5, 1, 1), padding=(2, 0, 0))
            )
 
    

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        residual_x = x.clone()


        if self.is_corr:
            corr_x = rearrange(x, '(b l t) c h w -> (b l) t c h w', l=self.layer_num, t=self.n_segments)

            prefix_x = corr_x.clone()
            prefix_x = rearrange(prefix_x, '(b l) t c h w -> (b t) l c h w', l=self.layer_num)
            prefix_x = torch.repeat_interleave(prefix_x, self.layer_num, dim=1)
            prefix_x = rearrange(prefix_x, 'b l c h w -> (b l) c h w')

            postfix_x = corr_x[:, 1:, ...]
            postfix_x = torch.cat([postfix_x, postfix_x[:, -1:, ...]], dim=1)
            postfix_x = rearrange(postfix_x, '(b l) t c h w -> (b t) l c h w', l=self.layer_num)
            postfix_x = postfix_x.repeat([1, self.layer_num, 1, 1, 1])
            postfix_x = rearrange(postfix_x, 'b l c h w -> (b l) c h w')

            # Perform correlation between an anchor frame and its neighboring frames at various layers
            corr = self.corr_module(prefix_x, postfix_x)
            corr = corr.reshape(-1, self.max_size**2, corr.shape[-2], corr.shape[-1])

            # Normalize the correlation scores into a probability distribution
            corr = F.softmax(corr/torch.sqrt(torch.tensor(corr_x.shape[-3]).float()), dim=1)

            # Compute the displacement vectors(u,v) for each frame
            x_disp = torch.arange(-self.max_size // 2 + 1, self.max_size // 2 + 1).to(corr.device)
            y_disp = torch.arange(-self.max_size // 2 + 1, self.max_size // 2 + 1).to(corr.device)
            x_disp, y_disp = torch.meshgrid(x_disp, y_disp, indexing='xy')
            x_disp, y_disp = x_disp.float(), y_disp.float()
            x_disp = x_disp.reshape(1, self.max_size**2, 1, 1)
            y_disp = y_disp.reshape(1, self.max_size**2, 1, 1)

            # equation A2 
            x_disp = x_disp * corr
            x_disp = x_disp.sum(dim=1)

            y_disp = y_disp * corr
            y_disp = y_disp.sum(dim=1)
        

            # Compute the partial derivatives of the displacement vectors
            partial_x = torch.gradient(x_disp, dim=-1)[0]
            partial_y = torch.gradient(y_disp, dim=-2)[0]

            # equation A5
            partial_xy = partial_x + partial_y
            partial_xy = -1 * partial_xy
            partial_xy = rearrange(partial_xy, '(b t l q) h w -> b t l q h w', l=self.layer_num, q=self.layer_num, t=self.n_segments)

            # Summarize the partial derivatives of the displacement vectors across different layers 
            w = torch.zeros(partial_xy.shape[0], partial_xy.shape[1], partial_xy.shape[2], 1, partial_xy.shape[4], partial_xy.shape[5]).to(partial_xy.device)
            for i in range(self.layer_num):
                w[:, :, i, ...] = partial_xy[:, :, i, i:, ...].sum(dim=2, keepdim=True)
            
            w = rearrange(w, 'b t l c h w -> (b l t) c h w')
            x_disp = rearrange(x_disp, '(b t l q) h w -> b t h w l q', t=self.n_segments, l=self.layer_num, q=self.layer_num)
            x_disp = x_disp.diagonal(dim1=-2, dim2=-1)
            x_disp = rearrange(x_disp, 'b t h w l -> (b l t) h w')

            y_disp = rearrange(y_disp, '(b t l q) h w -> b t h w l q', t=self.n_segments, l=self.layer_num, q=self.layer_num)
            y_disp = y_disp.diagonal(dim1=-2, dim2=-1)
            y_disp = rearrange(y_disp, 'b t h w l -> (b l t) h w')

            # Represent the three-dimensional motion field
            motion = torch.cat([
                x_disp.unsqueeze(dim=1),
                y_disp.unsqueeze(dim=1),
                w
            ], dim=1)

            # Generate the probability vector for motion guidiance
            motion = rearrange(motion, '(b l t) c h w -> (b t) c l h w', t=self.n_segments, l=self.layer_num)
            vertical = self.vertical(motion)
            vertical = rearrange(vertical, '(b t) c l h w -> (b l t) c h w', t=self.n_segments)
            vertical = vertical.unsqueeze(dim=2)
            vertical = vertical / 100.0
            vertical = vertical.softmax(dim=1)


            displacement = torch.cat([
                x_disp.unsqueeze(dim=1),
                y_disp.unsqueeze(dim=1)
            ], dim=1)
            displacement = rearrange(displacement, '(b l t) c h w -> (b l) c t h w', t=self.n_segments, l=self.layer_num)

            horizontal = self.horizontal(displacement)
            horizontal = rearrange(horizontal, '(b l) c t h w -> (b l t) c h w', l=self.layer_num)
            horizontal = horizontal.unsqueeze(dim=2)
            horizontal = horizontal / 100.0
            horizontal = horizontal.softmax(dim=1)

        x1 = rearrange(x, '(b l t) c h w -> (b l h) c t w', l=self.layer_num, t=self.n_segments)
        x1 = self.dwconv1(x1)
        x1 = rearrange(x1, '(b l h) c t w -> (b l t) c h w', l=self.layer_num, h=H)
        x1 = rearrange(x1, 'b (n c) h w -> b n c h w',  n=8)

        x2 = rearrange(x, '(b l t) c h w -> (b l w) c t h', l=self.layer_num, t=self.n_segments)
        x2 = self.dwconv2(x2)
        x2 = rearrange(x2, '(b l w) c t h -> (b l t) c h w', l=self.layer_num, w=W)
        x2 = rearrange(x2, 'b (n c) h w -> b n c h w', n=8)


        layer_x = torch.einsum('b m c h w, b n c h w -> b m n c h w', x1, x2)
        layer_x = rearrange(layer_x, 'b m n c h w -> b (m n) c h w')

        if self.is_corr:
            layer_x = horizontal * layer_x
        layer_x = rearrange(layer_x, 'b n c h w -> b (n c) h w')
        layer_x = self.layer_norm(layer_x)
        layer_x = F.relu(layer_x)
        layer_x = rearrange(layer_x, 'b c h w -> b h w c')
        layer_x = self.layer_pwconv(layer_x)
        layer_x = rearrange(layer_x, 'b h w c -> b c h w')


        # Volumetric explortion
        x3 = rearrange(x, '(b l t) c h w -> (b t h) c l w', l=self.layer_num, t=self.n_segments) #Decompose the volumetric strcuture into 2D planes and perform orientation detection
        x3 = self.dwconv3(x3)
        x3 = rearrange(x3, '(b t h) c l w -> (b l t) c h w', t=self.n_segments, h=H)
        x3 = rearrange(x3, 'b (n c) h w -> b n c h w', n=8)

        x4 = rearrange(x, '(b l t) c h w -> (b t w) c l h', l=self.layer_num, t=self.n_segments) #Decompose the volumetric strcuture into 2D planes and perform orientation detection
        x4 = self.dwconv4(x4)
        x4 = rearrange(x4, '(b t w) c l h -> (b l t) c h w', t=self.n_segments, w=W)
        x4 = rearrange(x4, 'b (n c) h w -> b n c h w', n=8)

        # Orientation combination
        temporal_x = torch.einsum('b m c h w, b n c h w -> b m n c h w', x3, x4)
        temporal_x = rearrange(temporal_x, 'b m n c h w -> b (m n) c h w')

        # Orientation selection
        if self.is_corr:
            temporal_x = vertical * temporal_x

        # Feature transformation
        temporal_x = rearrange(temporal_x, 'b n c h w -> b (n c) h w')
        temporal_x = self.temporal_norm(temporal_x)
        temporal_x = F.relu(temporal_x)
        temporal_x = rearrange(temporal_x, 'b c h w -> b h w c')
        temporal_x = self.temporal_pwconv(temporal_x)
        temporal_x = rearrange(temporal_x, 'b h w c -> b c h w')

        x = residual_x + layer_x + temporal_x

        return x








class YuShi3D(nn.Module):
    def __init__(self,
                 n_channels=1,
                 n_classes=20,
                 base_c=16,
                 bilinear=True,
                 layer_num=7,
                 n_segments=10,
                 use_checkpoint=False):
        super().__init__()
        self.n_channels = n_channels
        self.layer_num = layer_num
        self.n_segments = n_segments
        self.use_checkpoint = use_checkpoint

 
        self.inc = DoubleConv(1, base_c, is_pooling=True)

        self.down1 = Down(base_c * 1, base_c * 2)
        self.oriented_down1 = OrientedConv(base_c * 2, base_c * 2, kernel_size=5, layer_num=self.layer_num, n_segments=self.n_segments, is_corr=False)


        self.down2 = Down(base_c * 2, base_c * 3)
        self.oriented_down2 = OrientedConv(base_c * 3, base_c * 3, kernel_size=5, layer_num=self.layer_num, n_segments=self.n_segments, is_corr=True)


        self.down3 = Down(base_c * 3, base_c * 4)
        self.oriented_down3 = OrientedConv(base_c * 4, base_c * 4, kernel_size=5, layer_num=self.layer_num, n_segments=self.n_segments, is_corr=True)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 4, base_c * 8 // factor)
        self.oriented_down4 = OrientedConv(base_c * 8 // factor, base_c * 8 // factor, kernel_size=5, layer_num=self.layer_num, n_segments=self.n_segments, is_corr=True)
        
        self.up1 = Up(base_c * 8// factor * self.n_segments, base_c * 6 // factor * self.n_segments, bilinear)

        self.up2 = Up(base_c * 6 // factor * self.n_segments, base_c * 4 // factor * self.n_segments, bilinear)

        self.up3 = Up(base_c * 4 // factor * self.n_segments, base_c * 2 // factor * self.n_segments, bilinear)

        self.up4 = Up(base_c * 2 // factor * self.n_segments, base_c * 1, bilinear)

        self.outc = OutConv(base_c, n_classes)


    def forward(self, x):
        B, T, L, H, W = x.shape
        x = rearrange(x, 'b t l h w -> (b l) t h w')

        input_img = x

        x = rearrange(x, 'b t h w -> (b t) 1 h w')

        x1 = self.inc(x)
        
        x2 = self.down1(x1)
        x2 = self.oriented_down1(x2)

        x3 = self.down2(x2)
        x3 = self.oriented_down2(x3)

        x4 = self.down3(x3)
        x4 = self.oriented_down3(x4)

        x5 = self.down4(x4)
        x5 = self.oriented_down4(x5)


        x4 = rearrange(x4, '(b l t) c h w -> (b l) (t c) h w', l=self.layer_num, t=self.n_segments)
        x5 = rearrange(x5, '(b l t) c h w -> (b l) (t c) h w', l=self.layer_num, t=self.n_segments)


        x = self.up1(x5, x4)

        x3 = rearrange(x3, '(b l t) c h w -> (b l) (t c) h w', l=self.layer_num, t=self.n_segments)

        x = self.up2(x, x3)

        x2 = rearrange(x2, '(b l t) c h w -> (b l) (t c) h w', l=self.layer_num, t=self.n_segments)

        x = self.up3(x, x2)

        x1 = rearrange(x1, '(b l t) c h w -> (b l) (t c) h w', l=self.layer_num, t=self.n_segments)

        x = self.up4(x, x1)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)

        diffY = input_img.size()[2] - x.size()[2]
        diffX = input_img.size()[2] - x.size()[2]
        x = F.pad(x, [diffX // 2, diffX - diffY // 2, diffY // 2, diffY - diffY // 2])

        x = self.outc(x)

        x = rearrange(x, '(b l) t h w -> b t l h w', l=self.layer_num)
        
        return x
        
