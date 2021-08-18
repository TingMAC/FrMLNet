from modelUtil import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

    
class FrMLNet(nn.Module):
    def __init__(self,ms_channels=4, fc = 64, rbc = 2):
        super(FrMLNet,self).__init__()
        self.ms_channels = ms_channels
        g = 9
        temp = fc
        level = 4      
        #----------input conv-------------------
        
        #self.coefficient = nn.Parameter(torch.Tensor(np.ones((g*256,1))), requires_grad=True)
        self.conv_pan0_1 = nn.Conv2d(in_channels=1, out_channels=temp, kernel_size=3, stride=1, padding=1, bias=False)
        self.MSRB_pan1 = Residual_Block(inc=temp, outc=temp, groups=1)
        self.MSRB_pan2 = Residual_Block(inc=temp, outc=temp, groups=1)
        self.MSRB_pan3 = Residual_Block(inc=temp, outc=temp, groups=1)
        self.MSRB_pan4 = Residual_Block(inc=temp, outc=temp, groups=1)
        #self.residual_pan4_1 = Residual_Block(inc=512, outc=1024, groups=1)         
        
        self.conv_ms0_1 = nn.Conv2d(in_channels=self.ms_channels, out_channels=temp, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)                
        self.MSRB_ms1 = Residual_Block(inc=temp, outc=temp, groups=1)   
        self.MSRB_ms2 = Residual_Block(inc=temp, outc=temp, groups=1)    
        self.MSRB_ms3 = Residual_Block(inc=temp, outc=temp, groups=1)
        self.MSRB_ms4 = Residual_Block(inc=temp, outc=temp, groups=1)
        
        self.fusion1 = Residual_Block(inc=2*temp, outc=temp, groups=1)         
        self.fusion2 = Residual_Block(inc=2*temp, outc=temp, groups=1)   
        self.fusion3 = Residual_Block(inc=2*temp, outc=temp, groups=1)
        self.fusion4 = Residual_Block(inc=2*temp, outc=temp, groups=1)
        
        self.conv_agg1 = Residual_Block(inc=level*temp, outc=temp, groups=1) 
        self.fusion = RDB(in_channels = temp, num_dense_layer = 5, growth_rate = temp) 
        self.conv_agg2 = Residual_Block(inc=temp, outc=temp, groups=1)
        self.residual_framelet = nn.Sequential(make_layer(Residual_Block, rbc, inc=temp*g, outc=temp*g,groups = g))
        self.conv_output = nn.Conv2d(in_channels=temp*g, out_channels=self.ms_channels*g, kernel_size=3, stride=1, padding=1, groups=g,bias=False)        
        
        
    def forward(self,pan,ms):  
        pan0 = self.relu(self.conv_pan0_1(pan))  #64       
        pan1= self.MSRB_pan1(pan0)        
        pan2 = self.MSRB_pan2(pan1)
        pan3 = self.MSRB_pan3(pan2)
        pan4 = self.MSRB_pan4(pan3)
                 
        ms0 = self.relu(self.conv_ms0_1(ms)) #64       
        ms1 = self.MSRB_ms1(ms0)        
        ms2 = self.MSRB_ms2(ms1)       
        ms3 = self.MSRB_ms3(ms2)       
        ms4 = self.MSRB_ms4(ms3)            
        
        x1 = self.fusion1(torch.cat((pan1,ms1),1))
        x2 = self.fusion2(torch.cat((pan2,ms2),1))
        x3 = self.fusion3(torch.cat((pan3,ms3),1))
        x4 = self.fusion4(torch.cat((pan4,ms4),1))
        x5 = torch.cat((x1,x2,x3,x4),1)
        x6 = self.fusion((self.conv_agg1(x5)))
        x6 = self.conv_agg2(x6)
        x7 = torch.cat((x6,x6,x6,x6,x6,x6,x6,x6,x6),1)  
        #x8 = self.conv_framelet(x7)
        #x8 = self.coefficient[:2304,0][None, :, None, None] *x7
        x9 = self.residual_framelet(x7)
        out = self.conv_output(x9)
        
        return out