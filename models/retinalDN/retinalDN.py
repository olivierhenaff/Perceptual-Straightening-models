import torch
import torch.nn as nn
import scipy.io as sio
import torch.nn.functional as F 

class RetinalDN(nn.Module):

    def __init__(self):
        super(RetinalDN, self).__init__()

        loaddir = 'models/retinalDN/'
        filters = sio.loadmat(loaddir + 'filters.mat')
        params  = sio.loadmat(loaddir +  'params.mat')

        filters = torch.Tensor( filters['filters'] ) 
        params  = torch.Tensor(  params['params' ] ).squeeze()

        kerSize = filters.size(-1)

        self.pad = (kerSize-1)//2
        self.pad = ( self.pad, self.pad, self.pad, self.pad )

        self.linConv = nn.Conv2d( 1, 1, kerSize, bias=False )
        self.lumConv = nn.Conv2d( 1, 1, kerSize, bias=False )
        self.conConv = nn.Conv2d( 1, 1, kerSize, bias=False )

        self.linConv.weight.data.copy_( filters[0] ) 
        self.lumConv.weight.data.copy_( filters[1] ) 
        self.conConv.weight.data.copy_( filters[2] ) 

        self.softplus = nn.Softplus() 

        self.params = params


    def forward(self, x):

        x = x.unsqueeze( 1 ) 

        x = F.pad( x, self.pad, mode='replicate' )
        y = self.linConv( x ) 
        l = self.lumConv( x ) 

        y = y / ( 1 + self.params[0]*l ) 

        c = F.pad( y**2, self.pad, mode='replicate' )
        c = ( self.conConv(c) + 1e-6 ).sqrt() 

        y = y / ( 1 + self.params[1]*c ) 

        y = self.softplus( y ) 

        y = y.select( 1, 0 )

        return y