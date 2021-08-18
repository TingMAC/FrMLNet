#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import h5py
import torch.utils.data as data
import numpy as np
import torch
import torch.nn as nn

class my_dataset(data.Dataset):
    def __init__(self,mat_data):
        gt_set = mat_data['gt'][...]
        gt_set = np.transpose(gt_set,(3,0,1,2))
        pan_set = mat_data['pan'][...]
        pan_set = np.transpose(pan_set,(2,0,1))
        pan_set = pan_set[:,np.newaxis,:,:]
        ms_set = mat_data['ms'][...]
        ms_set = np.transpose(ms_set,(3,0,1,2))
        lms_set = mat_data['lms'][...]
        lms_set = np.transpose(lms_set,(3,0,1,2))
        self.gt_set = np.array(gt_set,dtype = np.float32) / 1.
        self.pan_set = np.array(pan_set, dtype = np.float32) /1.
        self.ms_set = np.array(ms_set, dtype = np.float32) / 1.
        self.lms_set  = np.array(lms_set, dtype = np.float32) /1.
        
    def __getitem__(self, index):
        gt = self.gt_set[index,:,:,:]
        pan = self.pan_set[index,:,:]
        ms = self.ms_set[index,:,:,:]
        lms  = self.lms_set[index,:,:,:]
        return gt, pan, lms, ms
    
    def __len__(self):
        return self.gt_set.shape[0]

if __name__ == "__main__":
    validation_data_name  = 'data.mat'  #your data path
    validation_data = h5py.File(validation_data_name,'r')
    validation_dataset = my_dataset(validation_data)
    del validation_data
    data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)    
    for index,item in enumerate(data_loader):
        print(index) 
        print(type(item[2]))

