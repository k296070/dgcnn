#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx


#def print(*args):
#    return

def knn(x, k=20):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        #nvtx.range_push("knn")
        idx = knn(x, k=k)   # (batch_size, num_points, k)
        #nvtx.range_pop()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    num_dims = x.size(1)

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    
    feature = feature.max(dim= -2, keepdim=True)[0]
    
    x = x.view(batch_size, num_points, 1, num_dims)
    
    feature = torch.cat((x,feature-x), dim=3).permute(0, 3, 1, 2).contiguous()
    #nvtx.range_pop()
  
    return feature

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        #nvtx.range_push("knn")
        idx = knn(x, k=k)   # (batch_size, num_points, k)
        #nvtx.range_pop()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    num_dims = x.size(1)

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    
    feature = feature.max(dim= -2, keepdim=True)[0]
    
    x = x.view(batch_size, num_points, 1, num_dims)
    
    feature = torch.cat((x,feature-x), dim=3).permute(0, 3, 1, 2).contiguous()
    #nvtx.range_pop()
  
    return feature

class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        #nvtx.range_push("forward_default")
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        #nvtx.range_pop()
        return x

class DGCNN_DA(nn.Module):
    def __init__(self, args = None, output_channels=40):
        super(DGCNN_DA, self).__init__()
        #self.args = args
        #self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(640, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 40)
        
    def forward(self, x):
        #nvtx.range_push("forward_DA")
        batch_size = x.size(0)
        dim = x.size(1)
        num_points = x.size(2)
        
        
        print("--------------------------------------------------------------------\nm1")
        nn_idx = knn(x)
        x = x.view(batch_size, dim, num_points, -1)
        
        print("(get_model) conv2d input:", x.shape)
        x = self.conv1(x)
        print("(get_model) conv2d output:", x.shape)
        x = get_graph_feature(x, idx=nn_idx)

        print("(get_model) neighbor search output:", x.shape)
        x1 = x.max(dim=-1, keepdim= False)[0]
        print("concat:", x1.shape)
        
        print("--------------------------------------------------------------------\nm2")
        nn_idx = knn(x1)
        batch_size = x1.size(0)
        dim = x1.size(1)
        num_points = x1.size(2)        
        x1 = x1.view(batch_size, dim, num_points, -1)

        print("(get_model) conv2d input:", x1.shape)
        x2 = self.conv2(x1)
        print("(get_model) conv2d output:", x2.shape)
        x2 = get_graph_feature(x2, idx=nn_idx)

        print("(get_model) neighbor search output:", x2.shape)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        print("concat:", x2.shape)
        print("--------------------------------------------------------------------\nm3")
        nn_idx = knn(x2)
        batch_size = x2.size(0)
        dim = x2.size(1)
        num_points = x2.size(2)        
        x2 = x2.view(batch_size, dim, num_points, -1)

        print("(get_model) conv2d input:", x2.shape)
        x3 = self.conv3(x2)
        print("(get_model) conv2d output:", x3.shape)
        x3 = get_graph_feature(x3, idx=nn_idx)

        print("(get_model) neighbor search output:", x3.shape)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        print("concat:", x3.shape)
        print("--------------------------------------------------------------------\nm4")
        nn_idx = knn(x3)
        batch_size = x3.size(0)
        dim = x3.size(1)
        num_points = x3.size(2)

        x3 = x3.view(batch_size, dim, num_points, -1)

        print("(get_model) conv2d input:", x3.shape)
        x4 = self.conv4(x3)
        print("(get_model) conv2d output:", x4.shape)
        x4 = get_graph_feature(x4, idx=nn_idx)

        print("(get_model) neighbor search output:", x4.shape)
        x4 = x4.max(dim=-1, keepdim=True)[0]
        print("concat:", x4.shape)
        print("--------------------------------------------------------------------\nm5")
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        print("(get_model) conv2d input:", x.shape)
        x = self.conv5(x)
        print("(get_model) conv2d output:", x.shape)
        x = x.max(dim=1, keepdim=True)[0]
        x = x.view(batch_size,-1)
        print("(get_model) max output:", x.shape)
        
        print("--------------------------------------------------------------------\nm6")
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        #nvtx.range_pop()
        return x
    
if __name__=='__main__':
    dgcnn = DGCNN_DA()
    x=(torch.arange(1,49).view(4,4,3))
    x=x.transpose(1,2)
    knn_1 = knn(x,2)
    print(knn_1.size())
    
    x=(torch.arange(64,0,-1).view(4,4,4))
    x=x.transpose(1,2)
    
    ggf = get_graph_feature(x,2,knn_1)
    ggf_1 = get_graph_feature(x,2)
    
    print(ggf_1.size())
    
    x = ggf_1.permute(0,2,3,1)
    
    #print(knn_1,"\n",ggf.permute(0,2,3,1)==ggf_1.permute(0,2,3,1))
    #print(ggf_1.permute(0,2,3,1))
    #x = dgcnn(x)
    print("--------------------------------------------------------------------\nRESULT")
    print(x, x.shape)