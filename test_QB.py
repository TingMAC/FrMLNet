#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
# This is a implementation of validation code of this paper:
# FrMLNet: Framelet-based Multi-level Network for Pansharpening
# author: Tingting Wang
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from skimage.measure import compare_psnr as psnr
import numpy as np
import h5py
from network import *
from util import *
from dataset import *
import math
import os
import pickle
import sys
import importlib
importlib.reload(sys)
import scipy.io


os.environ['CUDA_VISIBLE_DEVICES']='2'
devicesList = [0]
dtype = torch.cuda.FloatTensor
nc = 4
framelet_dec = FrameletTransform(channel=nc, dec=True)
framelet_rec = FrameletTransform(scale=1,channel=nc,dec=False)
framelet_dec = nn.DataParallel(framelet_dec,device_ids = devicesList).type(dtype)
framelet_rec = nn.DataParallel(framelet_rec,device_ids = devicesList).type(dtype)

if __name__ == "__main__":
    ##### read dataset #####
    test_bs = 1    
    tmpPath = "model/FrMLNet_QB.pth"
    SaveDataPath = "your_path.mat"
    test_data_name  = 'your_test_data.mat' 
    test_data = h5py.File(test_data_name,'r')
    test_dataset = my_dataset(test_data)
    del test_data
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, shuffle=True) 

    CNN = FrMLNet(nc,64,2)   
    CNN = nn.DataParallel(CNN,device_ids=devicesList).cuda()
    
    #summary(iBlock,[(1,64,64),(4,16,16)])
     
    CNN.load_state_dict(torch.load(tmpPath))
    CNN.eval()
    count = 0    
    for index, data in enumerate(test_dataloader):  
        pan_expand = torch.cat((data[1],data[1],data[1],data[1]),1)
        gtVar = Variable(data[0]).type(dtype) 
        panVar = Variable(data[1]).type(dtype) 
        framelet_gt = framelet_dec(gtVar)
        lmsVar = Variable(data[2]).type(dtype) 
        msVar = Variable(data[3]).type(dtype)  
        panVar_Exp = Variable(pan_expand).type(dtype)  
        framelet_lr = framelet_dec(lmsVar)             
        framelet_hr = framelet_dec(panVar_Exp)            
        framelet_predict = CNN(panVar, lmsVar)  
        main_framelet = torch.cat((framelet_lr[:,0:nc,:,:],framelet_hr[:,nc:,:,:]),1)
        #main_framelet = framelet_lr     
        output = framelet_rec(framelet_predict+main_framelet) 
        netOutput_np = output.cpu().data.numpy() 
        lms_np = data[2].numpy()
        ms_np = data[3].numpy()
        pan_np = data[1].numpy()
        gt_np = data[0].numpy()
        if count == 0:
            Output_np = netOutput_np
            ms = ms_np
            lms = lms_np
            pan = pan_np
            gt = gt_np
        else:
            Output_np = np.concatenate((netOutput_np, Output_np), axis=0)
            ms = np.concatenate((ms_np, ms), axis=0)
            lms = np.concatenate((lms_np, lms), axis=0)
            pan = np.concatenate((pan_np, pan), axis=0)
            gt = np.concatenate((gt_np, gt), axis=0)
        count = count + 1
    print(Output_np.shape)   
    scipy.io.savemat(SaveDataPath,{'QB256':Output_np, 'PAN256': pan, 'MS64': ms,'MS256':lms})   
    #scipy.io.savemat(SaveDataPath,{'QB256':Output_np, 'GT256': gt})  
            