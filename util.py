import numpy as np
from numpy import linalg as LA
import os
import cv2
import torch.nn as nn
import math
import pickle
import torch
from numpy import *


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def SAM(I1,I2):
    p1 = np.sum(I1*I2,0)
    p2 = np.sum(I1*I1,0)
    p3 = np.sum(I2*I2,0)
    p4 = np.sqrt(p2*p3)
    p5 = p4.copy()
    p5[p5==0]=1e-15

    sam = np.arccos(p1/p5)
    p1 = p1.ravel()
    p4 = p4.ravel()
    s1 = p1[p4!=0]
    s2 = p4[p4!=0]
    x = (s1/s2)
    x[x>1] = 1
    angolo = np.mean(np.arccos(x))
    sam = np.real(angolo)*180/np.pi
    
    return sam

def ERGAS(I1,I2,c=4):
    s = 0
    R = I1-I2
    for i in range(c):
        res = R[i]
        s += np.mean(res*res)/(np.mean(I1[i])*np.mean(I1[i]))
    s = s/c
    ergas = (100/4) * np.sqrt(s)
    
    return ergas



def loss_MSE(x, y, size_average=False):
    z = x - y 
    z2 = z * z
    if size_average:
        return z2.mean()
    else:
        return z2.sum().div(x.size(0)*2)
    
class FrameletTransform(nn.Module): 
    def __init__(self, scale=1, channel=4, dec=True, params_path='framelet_weight.pkl', transpose=True):
        super(FrameletTransform, self).__init__()
        self.channel = channel
        self.scale = scale
        self.dec = dec
        self.transpose = transpose        
        ks = 3
        nc = self.channel*9
        
        if dec:
            self.conv = nn.Conv2d(in_channels=self.channel, out_channels=nc, kernel_size=ks, stride=1, padding=1, groups=self.channel, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=self.channel, kernel_size=ks, stride=1, padding=1, groups=self.channel, bias=False)
        #print(self.conv)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = open(params_path,'rb')
                dct = pickle.load(f,encoding='iso-8859-1')
                f.close()                
                m.weight.data = torch.from_numpy(np.array(dct['f%d'%self.channel]))
                m.weight.requires_grad = False  
                           
    def forward(self, x): 
        if self.dec:
            output = self.conv(x)            
            if self.transpose:
                osz = output.size()
            #print(osz)
                output = output.view(osz[0], self.channel, -1, osz[2], osz[3]).transpose(1,2).contiguous().view(osz)            
                #output = torch.reshape(torch.reshape(output, (osz[0], self.channel, -1, osz[2], osz[3])).transpose(1,2), osz)
        else:
            if self.transpose:
                xx = x
                xsz = xx.size()
                xx = xx.view(xsz[0], -1, self.channel, xsz[2], xsz[3]).transpose(1,2).contiguous().view(xsz)             
                #xx = torch.reshape(torch.reshape(xx, (xsz[0], -1, self.channel, xsz[2], xsz[3])).transpose(1,2), xsz)
            output = self.conv(xx) 
        #print(output.shape)
        return output 