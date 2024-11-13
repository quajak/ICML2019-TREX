import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, name, input_dim, output_dim, k_h=4, k_w=4, d_h=2, d_w=2,
                 stddev=0.02, data_format='NCHW', padding='SAME'):
        super(Conv2d, self).__init__()
        # Note: PyTorch always uses NCHW format internally
        self.data_format = data_format
        padding = 'same' if padding == 'SAME' else 'valid'
        
        self.conv = nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=(k_h, k_w),
            stride=(d_h, d_w),
            padding=padding,
            bias=True
        )
        
        # Initialize weights using normal distribution
        nn.init.normal_(self.conv.weight, mean=0.0, std=stddev)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        if self.data_format == 'NHWC':
            # Convert NHWC to NCHW
            x = x.permute(0, 3, 1, 2)
            
        x = self.conv(x)
        
        if self.data_format == 'NHWC':
            # Convert back to NHWC
            x = x.permute(0, 2, 3, 1)
        
        return x

class WeightNormConv2d(nn.Module):
    def __init__(self, name, input_dim, output_dim, k_h=4, k_w=4, d_h=2, d_w=2,
                 stddev=0.02, data_format='NHWC', padding='SAME', epsilon=1e-9):
        super(WeightNormConv2d, self).__init__()
        assert data_format == 'NHWC', 'Only NHWC format supported in this implementation'
        padding = 'same' if padding == 'SAME' else 'valid'
        
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=(k_h, k_w),
                stride=(d_h, d_w),
                padding=padding
            )
        )
        
        # Initialize weights
        nn.init.normal_(self.conv.weight, mean=0.0, std=stddev)
        self.epsilon = epsilon

    def forward(self, x):
        # Convert NHWC to NCHW
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        # Convert back to NHWC
        x = x.permute(0, 2, 3, 1)
        return x

class Linear(nn.Module):
    def __init__(self, name, input_dim, output_dim, stddev=0.02):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.linear.weight, mean=0.0, std=stddev)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        if x.dim() > 2:
            # Flatten all dimensions except batch
            x = x.view(x.size(0), -1)
        return self.linear(x)

class LayerNorm(nn.Module):
    def __init__(self, name, axis, out_dim=None, epsilon=1e-7, data_format='NHWC'):
        super(LayerNorm, self).__init__()
        self.data_format = data_format
        self.axis = axis
        self.epsilon = epsilon
        
        if out_dim is not None:
            self.gamma = nn.Parameter(torch.ones(1, 1, 1, out_dim))
            self.beta = nn.Parameter(torch.zeros(out_dim))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        # Convert axis notation from TF to PyTorch if needed
        if isinstance(self.axis, list):
            dims = [dim if dim >= 0 else x.dim() + dim for dim in self.axis]
        else:
            dims = [self.axis]
            
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.epsilon)
        
        if self.gamma is not None:
            if self.data_format == 'NHWC':
                x = x * self.gamma + self.beta
            else:
                # NCHW format
                x = x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        
        return x

# Note: This implementation assumes MuJoCo-specific modifications may be needed for:
# 1. Input/output tensor shapes matching MuJoCo state/action spaces
# 2. Proper handling of MuJoCo observation normalization
# 3. Additional layers specific to MuJoCo policy networks

class ResidualBlock(nn.Module):
    def __init__(self, name, filters, filter_size=3, non_linearity=nn.LeakyReLU,
                 normal_method=lambda x: nn.InstanceNorm2d(filters)):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d(name+'_1', filters, filters, filter_size, filter_size, 1, 1)
        self.norm = normal_method(name+'_norm')
        self.nl = non_linearity()
        self.conv2 = Conv2d(name+'_2', filters, filters, filter_size, filter_size, 1, 1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.nl(out)
        out = self.conv2(out)
        return out + identity

# Uncertain areas and potential modifications needed for MuJoCo:
# 1. The ideal padding strategy for MuJoCo state processing
# 2. Whether NHWC vs NCHW format matters for MuJoCo state representations
# 3. The best initialization strategies for MuJoCo policy networks
# 4. Whether LayerNorm axis configurations need adjustment for MuJoCo
# 5. If additional layer types are needed for specific MuJoCo architectures
