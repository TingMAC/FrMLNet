#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
# This is a implementation of training code of this paper:
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
import numpy as np
import h5py
from skimage.measure import compare_psnr as PSNR
from network import *
from util import *
from dataset import *
import math
import os
import pickle
import sys
import importlib
importlib.reload(sys)
import scipy.io as sio

validRecord = {"epoch":[],"PSNR":[],"SAM":[],"ERGAS":[]}

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
devicesList = [0,1]
dtype = torch.cuda.FloatTensor
MAEloss = torch.nn.L1Loss().type(dtype)
MSEloss = torch.nn.MSELoss().type(dtype)

framelet_dec = FrameletTransform(scale = 1, channel=4, dec=True)
framelet_rec = FrameletTransform(scale=1,channel=4, dec=False)
framelet_dec = nn.DataParallel(framelet_dec,device_ids = devicesList).type(dtype)
framelet_rec = nn.DataParallel(framelet_rec,device_ids = devicesList).type(dtype)

def validation(dataloader):
    sum_psnr = 0
    sum_sam = 0
    sum_ergas = 0
    count = 0
    CNN.eval()
    for index, data in enumerate(dataloader):
        #count += data[0].shape[0]
        count += 1
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
        netOutput_np = output.cpu().data.numpy()[0]
        gtLabel_np = gtVar.cpu().data.numpy()[0]
        samValue = SAM(gtLabel_np,netOutput_np)
        ergasValue = ERGAS(gtLabel_np,netOutput_np)        
        psnrValue = PSNR(gtLabel_np,netOutput_np)   
        sum_sam += samValue
        sum_psnr += psnrValue
        sum_ergas += ergasValue       
 
    avg_psnr = sum_psnr/count
    avg_sam = sum_sam/count
    avg_ergas = sum_ergas/count   
    
    print('psnr:%.4f sam:%.4f ergas:%.4f'%(avg_psnr,avg_sam,avg_ergas))
    return avg_psnr,avg_sam,avg_ergas

if __name__ == "__main__":
    ## parameters setting and network selection ##
    train_bs = 24
    val_bs = 1
    test_bs = 1
    nc = 4
    epoch = 200
    LR = 0.0001
    clip_boundary = 2
    CNN = FrMLNet(nc,64,3)
    CNN = nn.DataParallel(CNN,device_ids=devicesList).cuda()  
    ## parameters setting ##

    log_file = 'your_logfile_path'
    resume_train = False
    if not os.path.exists(log_file):
        os.makedirs(log_file)
    ### read dataset ###    
    train_data_name = 'your_trainning_data.mat'
    train_data = h5py.File(train_data_name,'r')
    train_dataset = my_dataset(train_data)
    trainsetSize = train_data['gt'].shape[3]
    del train_data
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
    validation_data_name  = 'your_validation_data.mat'
    validation_data = h5py.File(validation_data_name,'r')
    validation_dataset = my_dataset(validation_data)
    del validation_data  
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=val_bs, shuffle=True)
    savemat_data_name = 'your_save_data.mat'
    savenet_data_name = 'your_logfile.pth'

    
    optimizer = torch.optim.Adam(CNN.parameters(), lr=LR,betas=(0.9, 0.999),weight_decay=0.0001)
    
    for i in range(1,epoch+1):          
        count = 0     
        CNN.train()
        for index, data in enumerate(train_dataloader):    
            count += data[0].shape[0]
            optimizer.zero_grad()            
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
            img_predict = framelet_rec(framelet_predict+main_framelet)            
          
            loss_lr = MAEloss(framelet_predict[:,0:nc,clip_boundary:-clip_boundary,clip_boundary:-clip_boundary]+framelet_lr[:,0:nc,clip_boundary:-clip_boundary,clip_boundary:-clip_boundary], framelet_gt[:,0:nc,clip_boundary:-clip_boundary,clip_boundary:-clip_boundary])
            loss_sr = MAEloss(framelet_predict[:,nc:,clip_boundary:-clip_boundary,clip_boundary:-clip_boundary]+framelet_hr[:,nc:,clip_boundary:-clip_boundary,clip_boundary:-clip_boundary], framelet_gt[:,nc:,clip_boundary:-clip_boundary,clip_boundary:-clip_boundary])            
            
            loss_mae = MAEloss(img_predict,gtVar)
            loss_img = loss_mae
            
            loss = loss_sr.mul(10) + loss_lr.mul(0.1)+ loss_img.mul(1) 
            loss.backward()            
            optimizer.step()
            print('epoch:%04d [%05d/%05d] loss %.8f imgloss %.8f'%(i,count,trainsetSize,loss.data,loss_img.data), '\r',end = '\r')
            #torch.save(CNN.state_dict(),'./log_FrResPanNet_QB/FrResPanNet0519_QB16_1_01_01.pth'.format(i))
            
        if (i)%50 == 0:
            LR = LR/2
            optimizer = torch.optim.Adam(CNN.parameters(), lr=LR,betas=(0.9, 0.999),weight_decay=0.0001)
            #torch.save(CNN.state_dict(),'./log_FTMSPNN_QB/FTMSPNN_QB16_1_1_0_epoch{}.pth'.format(i))
         
        if (i)%2 == 0:
            print("")
            #validation(validation_dataloader)
            #psnr,sam,ergas = validation(test_dataloader)
            psnr,sam,ergas = validation(validation_dataloader)            
            validRecord["epoch"].append(i)
            validRecord["PSNR"].append(psnr)
            validRecord["SAM"].append(sam)
            validRecord["ERGAS"].append(ergas)
            #torch.save(CNN.state_dict(),'./log_FrResPanNet_QB/FrResPanNet_QB32_10_01_1.pth')
            sio.savemat(savemat_data_name,validRecord)
    torch.save(CNN.state_dict(),savenet_data_name.format(i))
            
            