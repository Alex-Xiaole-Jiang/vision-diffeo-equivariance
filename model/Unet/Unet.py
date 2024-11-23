#%%
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        # Set default values for kernel_size and padding if not provided
        conv_kwargs = {
            'kernel_size': 3,
            'padding': 1,
            'bias': False  # We don't need bias since we're using BatchNorm
        }
        # Update with any provided kwargs
        conv_kwargs.update(kwargs)
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            **conv_kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ConvBlockNet(nn.Module):
    """
    Build ConvNet using any number of ConvBlocks.
    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels for first conv and subsequent blocks
        num_blocks (int): Number of convolutional blocks to use
        **block_kwargs: Additional arguments to pass to each ConvBlock
    """
    def __init__(self, input_channels: int, output_channels: int, num_blocks: int, 
                    skip_connection = True, **block_kwargs):
        super().__init__()
        self.in_channel = input_channels
        self.out_channel = output_channels
        self.skip_connection = skip_connection
        if num_blocks < 1:
            raise ValueError("Number of blocks must be at least 1")
            
        # Create a ModuleList to hold our conv blocks
        self.blocks = nn.ModuleList()
        
        # First block has different input channels
        self.blocks.append(ConvBlock(input_channels, output_channels, **block_kwargs))
        
        # Remaining blocks have same in/out channels
        for _ in range(num_blocks - 1):
            self.blocks.append(ConvBlock(output_channels, output_channels, **block_kwargs))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First block output (for residual connection)
        if self.skip_connection:
            if self.in_channel == self.out_channel:
                residual_connection = x
                x = self.blocks[0](x)
                if len(self.blocks) == 1:
                    return x
            else:
                residual_connection = self.blocks[0](x)
                x = residual_connection
            # Pass input through all blocks sequentially
            for block in self.blocks[1:]:
                x = block(x)            
            return residual_connection + x
        else:
            for block in self.blocks:
                x = block(x)            
            return x

class UNet(nn.Module):
    def __init__(self, config):
        """
        Initialize UNet with a configuration dictionary.
        
        Args:
            config (dict): Configuration with the following fields:
                - in_channels (int): Number of input channels
                - out_channels (int): Number of output channels
                - initial_filters (int): Number of filters in first layer
                - depth (int): Number of down/up sampling operations
                - num_conv_in_resid (list of int): 
                    Number of Conv blocks in each residual block at each depth
                - num_resid_blocks (list of int):
                    Number of residual blocks at each depth
                - num_conv_in_skip (list of int):
                    Number of Conv blocks in each skip connection
                - conv_kwargs (dict): Keywords for ConvBlockNet
                - upsample_kwargs (dict): Keywords for upsampling
        """
        super().__init__()
        
        self.depth = config['depth']
        initial_filters = config['initial_filters']
        
        # Create encoders (contracting path)
        self.encoders = nn.ModuleList()
        in_channels = config['in_channels']
        num_conv_in_resid = config['num_conv_in_resid']
        num_resid_blocks = config['num_resid_blocks']
        num_conv_in_skip = config['num_conv_in_skip']
        self.num_resid_blocks = num_resid_blocks

        assert len(num_conv_in_resid) == self.depth + 1, \
            'need to specify how many Conv blocks to use for each residual block at each depth!'
        assert len(num_resid_blocks) == self.depth + 1, \
            'need to specify how many residual block to use at each depth!'
        assert len(num_conv_in_skip) == self.depth, \
            'need to specify how many Conv block to use at each skip connection!'


        for i in range(self.depth + 1):  # +1 for bottom level
            out_channels = initial_filters * (2 ** i)
            self.encoders.append(
                ConvBlockNet(in_channels, out_channels, num_conv_in_resid[i], **config['conv_kwargs'])
            )
            in_channels = out_channels
            for _ in range(num_resid_blocks[i] - 1):
                self.encoders.append(
                    ConvBlockNet(out_channels, out_channels, num_conv_in_resid[i], **config['conv_kwargs'])
                )
            
        # Create skip connection processors
        self.skip_processors = nn.ModuleList()
        for i in range(self.depth):  # Don't need skip for bottom level
            channels = initial_filters * (2 ** i)
            self.skip_processors.append(
                ConvBlockNet(channels, channels, num_conv_in_skip[i], skip_connection=False, **config['conv_kwargs'])
            )
            
        # Create decoders (expanding path)
        self.decoders = nn.ModuleList()
        for i in range(self.depth):
            in_channels = initial_filters * (2 ** (self.depth - i ))
            out_channels = initial_filters * (2 ** (self.depth - i - 1))
            self.decoders.append(
                ConvBlockNet(in_channels, out_channels, num_conv_in_resid[-i-2], **config['conv_kwargs'])
            )
            for _ in range(num_resid_blocks[-i-2] - 1):
                self.decoders.append(
                    ConvBlockNet(out_channels, out_channels, num_conv_in_resid[-i-2], **config['conv_kwargs'])
                )
                
            
        # Pooling and upsampling
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, **config['upsample_kwargs'])
        
        # Final convolution
        self.out_conv = nn.Conv2d(
            initial_filters, 
            config['out_channels'], 
            **config['conv_kwargs']
        )

    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Contracting path
        count = 0
        for i in range(self.depth):
            for j in range(self.num_resid_blocks[i]):
                x = self.encoders[count](x)
                count += 1
            if count != sum(self.num_resid_blocks):
                encoder_outputs.append(x)
            x = self.pool(x)
            
            
        # Bottom level
        for i in range(count, sum(self.num_resid_blocks)):
            x = self.encoders[i](x)

        # Expanding path
        count = 0
        for i in range(self.depth):
            x = self.up(x)
            for j in range(self.num_resid_blocks[-i-2]):
                x = self.decoders[count](x)
                count += 1
                if j == 1:
                    skip = self.skip_processors[-(i+1)](encoder_outputs[-(i+1)])
                    x = x + skip  # Additive skip connection
            #print(skip.shape)
            
        return self.out_conv(x)
# %%
