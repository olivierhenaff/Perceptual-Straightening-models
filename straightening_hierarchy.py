import torch 
import torch.nn as nn
import torch.nn.functional as F 

import pytorch_fft.fft.autograd as fft
import pytorch_fft.fft          as fftf

import sys 
import os

from models.retinalDN.retinalDN import RetinalDN
from models.steerable.steerable import SteerablePyramid

class Straightening_Hierarchy(nn.Module):

    def __init__( self, imgSize, N, K ):
        super(Straightening_Hierarchy, self).__init__()

        self.retinalDN = RetinalDN() 
        self.pyr = SteerablePyramid( imgSize, K=K, N=N, hilb=True )


    def compute_pyr(self, x):

        x = self.pyr( x.unsqueeze(1) )

        x.pop( 0 ) # remove low-pass
        x.pop(   ) # remove high-pass

        for i in range( len(x) ):
            x[i] = ( x[i] ** 2 ).sum(1).select( 1, 0 ) # compute complex magnitude 

        for i in range( len(x) ):
            x[i] = x[i].view( x[i].size(0), -1 ) 
        
        return torch.cat( tuple(x), -1 ) # concatenate all pyramid bands 

    def forward(self, x):

        y = {}

        y['pixel'] = x / 255 

        y['retina'] = self.retinalDN( y['pixel'] )

        y['v1'] = self.compute_pyr( y['retina'] )

        return y
