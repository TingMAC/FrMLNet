import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

   
# --- Residual block in GridDehazeNet  --- #    
class Residual_Block(nn.Module): 
    def __init__(self, inc = 64, outc = 64, ks =3, groups=1, dilation=1):
        super(Residual_Block, self).__init__()        
        if inc is not outc:
            self.conv_expand = nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        else:
            self.conv_expand = None 
        if ks == 1:
            self.conv1 = nn.Conv2d(inc, outc, kernel_size=ks, stride=1, padding=0, groups=groups, bias=False)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(outc, outc, kernel_size=ks, stride=1, padding=0, groups=groups, bias=False)
            self.relu2 = nn.ReLU(inplace=True)
        else:            
            self.conv1 = nn.Conv2d(inc, outc, kernel_size=ks, stride=1, padding=dilation, groups=groups, bias=False, dilation=dilation)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(outc, outc, kernel_size=ks, stride=1, padding=dilation, groups=groups, bias=False, dilation=dilation)
            self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x): 
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x
        output = self.relu1(self.conv1(x))
        output = self.conv2(output)
        output = self.relu2(torch.add(output,identity_data))
        return output
    
# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out  


 