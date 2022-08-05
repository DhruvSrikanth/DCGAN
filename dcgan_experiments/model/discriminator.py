import torch
import torch.nn as nn

import typing
import numpy as np

import warnings

class Discriminator(nn.Module):
    def __init__(self, in_shape: tuple, name:str=None) -> None:
        '''
        Initialize the discriminator.
        Parameters:
            in_shape: The shape of the input image.
            name: The name of the discriminator.
        Returns:
            None
        '''
        super(Discriminator, self).__init__()
        self.name = "Discriminator" if name is None else name
        self.in_shape = in_shape
        self.n_channels = in_shape[0]
        self.n_fmaps = in_shape[1]
        self.n_blocks = np.log2(np.sqrt(self.n_fmaps)).astype(np.int64)
        
        # Define input block
        self.in_block = nn.ModuleDict(
            {
                'in_block' : nn.Sequential(
                    nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_fmaps, kernel_size=4, stride=2, padding=1, bias=False), 
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
            }
        )

        def intermediate_block(in_channels: int, out_channels: int) -> typing.List[nn.Module]:
            '''
            Each block that makes up the generator.
            Parameters: 
                in_channels: The input channels of the block.
                out_channels: The output channels of the block.
            Returns:
                A list of modules that make up the block.
            '''
            layers = []
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            return layers

        
        # Define intermediate blocks
        out_channels = self.n_fmaps
        self.inter_blocks = nn.ModuleDict({})
        for block_num in range(self.n_blocks):
            in_channels = out_channels
            out_channels = in_channels * 2
            self.inter_blocks[f'inter_block_{block_num+1}'] = nn.Sequential(*intermediate_block(in_channels=in_channels, out_channels=out_channels))
        
        # Define output block
        out_channels = self.n_fmaps * np.sqrt(self.n_fmaps).astype(np.int64)
        self.out_block = nn.ModuleDict(
            {
                'out_block' : nn.Sequential(
                    nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
                    nn.Sigmoid()
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
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Forward pass of the discriminator.
        Parameters:
            x: The input image.
        Returns:
            The output score.
        '''
        # Input block
        x = self.in_block['in_block'](x)

        # Intermediate blocks
        for i in range(len(self.inter_blocks)):
            x = self.inter_blocks[f'inter_block_{i+1}'](x)
        
        # Output block
        validity = self.out_block['out_block'](x)
        
        return validity
