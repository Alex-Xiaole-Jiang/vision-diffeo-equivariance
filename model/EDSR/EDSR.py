#%%
import torch
import torch.nn as nn
from math import floor
from dataclasses import dataclass, field

@dataclass
class SingleScaleEDSRConfig:
    num_resid_blocks: int
    res_net_in_filters: list[int] 
    res_net_out_filters: list[int]
    res_net_kernel_size: list[int]
    in_channels: int = 1
    out_channels: int = 1
    Res_block_kwargs: dict = field(default_factory = lambda: dict())

class EDSRResBlock(nn.Module):
    def __init__(self, 
                in_channels: list[int],
                out_channels: list[int],
                kernel_size:list[int],
                non_linear_fn = nn.ReLU(),
                skip_connection = True,
                **kwargs):
        super().__init__()
        length_of_block = len(in_channels)
        assert length_of_block == len(out_channels)
        assert length_of_block == len(kernel_size)
        
        padding = [int(floor(kernel/2)) for kernel in kernel_size]

        self.block = nn.ModuleList()
        for i in range(length_of_block):
            self.block.append(nn.Conv2d(
                in_channels = in_channels[i],
                out_channels = out_channels[i],
                kernel_size = kernel_size[i],
                padding = padding[i],
                **kwargs
            ))
            if i != (length_of_block - 1):
                self.block.append(non_linear_fn)
            if i == (length_of_block - 1):
                self.block.append(nn.BatchNorm2d(out_channels[i]))
        
        self.block = nn.Sequential(*self.block)

        self.skip = skip_connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip == False:
            return self.block(x)
        
        return x + self.block(x)


class SingleScaleEDSR(nn.Module):
    def __init__(self, config):
        """
        Initialize EDSR with a configuration dictionary.
        
        Args:
            config (dict): Configuration with the following fields:
                - in_channels (int): Number of input channels
                - out_channels (int): Number of output channels
                - res_net_in_filters (list of int): config for Res block
                - res_net_out_filters (list of int): config for Res block
                - res_net_kernel_size (list of int): config for Res block
                - num_resid_blocks (list of int):
                    Number of residual blocks at each depth
                - Res_block_kwargs (dict): Keywords for Res block
        """
        super().__init__()
        
        in_channels = config.in_channels
        out_channels = config.out_channels
        num_resid_blocks = config.num_resid_blocks
        self.num_resid_blocks = num_resid_blocks

        in_filters_list = config.res_net_in_filters
        out_filters_list = config.res_net_out_filters
        kernel_size_list = config.res_net_kernel_size
        
        self.in_conv  = nn.Conv2d(
            in_channels, 
            in_filters_list[0], 
            kernel_size = 7,
            padding = 3)
        self.out_conv = nn.Conv2d(
            in_filters_list[0], 
            out_channels, 
            kernel_size = 3,
            padding = 1
        )

        self.res_blocks = nn.ModuleList()
        for i in range(self.num_resid_blocks): 
            self.res_blocks.append(
                EDSRResBlock(in_filters_list, 
                             out_filters_list, 
                             kernel_size_list, 
                             **config.Res_block_kwargs)
            )

    def forward(self, x):
        x = self.in_conv(x)
        x = nn.Sequential(*self.res_blocks)(x)
        x = self.out_conv(x)
            
        return x
# %%
