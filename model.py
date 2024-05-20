from typing import List

import einops
import numpy as np
import torch
from einops.layers.torch import Reduce
from scipy.special import jacobi
from torch import nn
from torch.nn import functional as F


def get_ortho_polynomials(degrees: int, length: int, alpha=0, beta=0):
    """Generate the set of Jacobi orthogonal polynomials with shape (degrees + 1, length)

    Args:
        degrees (int): The maximum polynomial degree. 
            Note that degrees + 1 polynomials will be generated (counting the constant)
        length (int): The length of the discretized temporal kernel, 
            assuming the range [0, 1] for the polynomials.
        alpha (int, optional): The alpha Jacobi parameter. Defaults to 0.
        beta (int, optional): The beta Jacobi parameter. Defaults to 0.

    Returns:
        np.ndarray: shaped (degrees + 1, length)
    """
    coeffs = np.vstack([np.pad(np.flip(jacobi(degree, alpha, beta).coeffs), (0, degrees - degree)) 
                        for degree in range(degrees + 1)]).astype(np.float32)
    steps = np.linspace(0, 1, length + 1)  # the discrete steps
    X = np.stack([steps ** (i + 1) / (i + 1) for i in range(degrees + 1)])
    polynomials_integrated = coeffs @ X
    ortho_polynomials = np.diff(polynomials_integrated, 1, -1) * length
    return ortho_polynomials


class PleiadesLayer(nn.Conv3d):
    def __init__(self, *args, degree=4, alpha=0, beta=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.degree, self.alpha, self.beta = degree, alpha, beta
        self._parameterize_weight(degree, alpha, beta)
    
    def _parameterize_weight(self, degree, alpha, beta):
        transform = get_ortho_polynomials(degree, self.kernel_size[-1], alpha=alpha, beta=beta)
        transform = torch.tensor(transform).float()
        self.scale = (self.weight.shape[1] ** 0.5) * (self.kernel_size[-1] ** 0.5)
        transform = transform / self.scale
        
        self.transform = nn.Parameter(transform, requires_grad=False)  # (coeffs, kernel_size)
        self.weight = nn.Parameter(torch.rand(self.out_channels, self.weight.shape[1], *self.kernel_size[:-1], degree + 1))
    
    def resample(self, length):
        transform = get_ortho_polynomials(self.degree, length, alpha=self.alpha, beta=self.beta)
        transform = torch.tensor(transform).float()
        transform = transform / self.scale
        self.transform = nn.Parameter(transform, requires_grad=False)
    
    def forward(self, input):
        kernel = self.weight @ self.transform
        return F.conv3d(input, kernel, 
                        bias=self.bias, groups=self.groups, 
                        stride=self.stride, padding=self.padding, dilation=self.dilation)
        
        
class CausalGroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, input):
        batch_size = input.shape[0]
        input = einops.rearrange(input, 'b ... t -> (b t) ...')
        output = super().forward(input)
        return einops.rearrange(output, '(b t) ... -> b ... t', b=batch_size)
    

class STBlock(nn.Module):
    """A Conv(1+2)D block consisting of a temporal convolution block followed by a spatial convolution block.
    """
    def __init__(self, 
                 in_channels, 
                 med_channels, 
                 out_channels, 
                 kernel_size=(3, 3, 3), 
                 stride=(1, 1, 1), 
                 padding=(0, 0, 0), 
                 dws=False, 
                 degree=4, 
                 alpha=-0.25, 
                 beta=-0.25, 
                 **kwargs):
        super().__init__()
        self.dws = dws
        self.t_kernel_size = kernel_size[-1]
        
        if dws:
            self.conv_t = nn.Sequential(
                PleiadesLayer(in_channels, in_channels, 
                               (1, 1, kernel_size[-1]), stride=(1, 1, stride[-1]), padding=(0, 0, padding[-1]), 
                               groups=in_channels, bias=False, degree=degree, alpha=alpha, beta=beta), 
                nn.Sequential(CausalGroupNorm(4, in_channels), nn.ReLU()), 
                nn.Conv3d(in_channels, med_channels, 1, bias=False), 
                nn.Sequential(CausalGroupNorm(4, med_channels)), 
            )
            
            self.t_skip = nn.Conv3d(in_channels, med_channels, 1)
            
            self.conv_s = nn.Sequential(
                nn.Conv3d(med_channels, med_channels, 
                          (*kernel_size[:-1], 1), (*stride[:-1], 1), (*padding[:-1], 0), 
                          groups=med_channels, bias=False), 
                nn.Sequential(nn.BatchNorm3d(med_channels), nn.ReLU()), 
                nn.Conv3d(med_channels, out_channels, 1, bias=False), 
                nn.Sequential(nn.BatchNorm3d(out_channels), nn.ReLU()), 
            )
            
        else:
            self.conv_t = nn.Sequential(
                PleiadesLayer(in_channels, med_channels, 
                               (1, 1, kernel_size[-1]), stride=(1, 1, stride[-1]), padding=(0, 0, padding[-1]), 
                               bias=False, degree=degree, alpha=alpha, beta=beta), 
                nn.Sequential(CausalGroupNorm(4, med_channels), nn.ReLU())
            )
            
            self.conv_s = nn.Sequential(
                nn.Conv3d(med_channels, out_channels, 
                          (*kernel_size[:-1], 1), (*stride[:-1], 1), (*padding[:-1], 0), 
                          bias=False), 
                nn.Sequential(nn.BatchNorm3d(out_channels), nn.ReLU()), 
            )
    
    def forward(self, input):
        if self.dws:
            x = F.relu(self.t_skip(input[..., self.t_kernel_size-1:]) + self.conv_t(input))
        else:
            x = self.conv_t(input)
        
        return self.conv_s(x)
    
    
class PleiadesClassifier(nn.Module):
    """A simple classifier consisting of multiple spatiotemporal blocks 
    stacked together, and a 2-layer MLP classifier at the end.
    """
    def __init__(self, 
                 in_channels: int, 
                 num_classes: int, 
                 channels: List[int], 
                 features: int, 
                 depthwises: List[bool], 
                 **kwargs):
        super().__init__()
        channels = [in_channels] + channels
    
        self.conv_blocks = nn.Sequential()
        for block_id in range(len(depthwises)):
            in_channels, med_channels, out_channels = channels[2*block_id:2*block_id+3]
            dws = depthwises[block_id]
            self.conv_blocks.append(
                STBlock(in_channels, med_channels, out_channels, dws=dws, **kwargs), 
            )
        
        self.classifier = nn.Sequential(
            Reduce('b c h w t -> b c t', 'mean'), 
            nn.Sequential(
                nn.Conv1d(channels[-1], features, 1), 
                nn.ReLU(), 
                nn.Conv1d(features, num_classes, 1), 
            )
        )
        
    def forward(self, inputs):
        return self.classifier(self.conv_blocks(inputs))


# model = PleiadesClassifier(
#     in_channels=2, 
#     num_classes=10, 
#     channels=[8, 16, 32, 48, 64, 80, 96, 112, 128, 256], 
#     features=256, 
#     depthwises=[False, False, True, True, True], 
#     kernel_size=[3, 3, 10], 
#     stride=[2, 2, 1], 
#     padding=[1, 1, 0], 
# )


# ckpt = torch.load('dvs128.ckpt', map_location='cpu')
# state_dict = ckpt['state_dict']

# state_dict = {k.replace('model._orig_mod.', ''): v for k, v in state_dict.items()}
# state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

# state_dict = {k.replace('connect_t.norm_drop.0', 'conv_t.3.0'): v for k, v in state_dict.items()}
# state_dict = {k.replace('connect_t.skip_path', 't_skip'): v for k, v in state_dict.items()}

# model.load_state_dict(state_dict)
# torch.save(model.state_dict(), 'dvs128_ckpt.pt')
