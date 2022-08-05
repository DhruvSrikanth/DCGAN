import torch
import torch.nn as nn

import typing
import numpy as np

class Generator(nn.Module):
    def __init__(self, z_dim: int, out_shape: tuple, name:str=None) -> None:
        '''
        Initialize the generator.
        Parameters:
            z_dim: The dimension of the latent space.
            out_shape: The shape of the output image.
            name: The name of the generator.
        Returns:
            None
        '''
        super(Generator, self).__init__()
        self.name = "Generator" if name is None else name
        self.z_dim = z_dim
        self.out_shape = out_shape
        self.n_channels = out_shape[0]
        self.n_fmaps = out_shape[1]
        self.n_blocks = np.log2(np.sqrt(self.n_fmaps)).astype(np.int64)

        # Define input block
        out_channels = self.n_fmaps * np.sqrt(self.n_fmaps).astype(np.int64)
        self.in_block = nn.ModuleDict(
            {
                'in_block' : nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.z_dim, out_channels=out_channels, kernel_size=4, stride=1, padding=0, bias=False), 
                    nn.BatchNorm2d(num_features=out_channels), 
                    nn.ReLU(inplace=True)
                )
            }
        )

        def normalized_upsampling_block(in_channels: int, out_channels: int) -> typing.List[nn.Module]:
            '''
            Each block that makes up the generator.
            Parameters: 
                in_channels: The input channels of the block.
                out_channels: The output channels of the block.
            Returns:
                A list of modules that make up the block.
            '''
            layers = []
            layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            return layers

        
        # Define intermediate blocks
        self.inter_blocks = nn.ModuleDict({})
        for block_num in range(self.n_blocks):
            in_channels = out_channels
            out_channels = in_channels // 2
            self.inter_blocks[f'inter_block_{block_num+1}'] = nn.Sequential(*normalized_upsampling_block(in_channels=in_channels, out_channels=out_channels))
        
        # Define output block
        self.out_block = nn.ModuleDict(
            {
                'out_block' : nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.n_fmaps, out_channels=self.n_channels, kernel_size=4, stride=2, padding=1, bias=False), 
                    nn.Tanh()
                )
            }
        )

        # Initialize weights
        self.apply(self._init_weights)
        self.n_blocks = len(self.inter_blocks) + 2

    @torch.no_grad()
    def _init_weights(self, m: nn.Module) -> None:
        '''
        Initialize the weights of the generator.
        Parameters:
            m: The module to initialize.
        Returns:
            None
        '''
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
        
    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Forward pass of the generator.
        Parameters:
            z: The latent space.
        Returns:
            The output sample.
        '''
        # Input block
        x = self.in_block['in_block'](z)
        
        # Intermediate blocks
        for i in range(len(self.inter_blocks)):
            x = self.inter_blocks[f'inter_block_{i+1}'](x)
        
        # Output block
        sample = self.out_block['out_block'](x)

        return sample