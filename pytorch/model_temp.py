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

from dgcnn_utils import knn, get_graph_feature, get_graph_feature_DA
from pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstraction_DA

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

class PointNet2(nn.Module):
    def __init__(self,num_class=40,normal_channel=False):
        super(PointNet2, self).__init__()
        in_channel = 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        print("l3",l3_points.shape)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x

class PointNet2_DA(nn.Module):
    def __init__(self,num_class=40,normal_channel=False):
        super(PointNet2_DA, self).__init__()
        in_channel = 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction_DA(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction_DA(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction_DA(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        print("!")
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        print("stage 2")
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        print("stage 3")
        print("l3",l3_points.shape)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x

class DGCNN(nn.Module):
    def __init__(self, args=None, output_channels=40):
        super(DGCNN, self).__init__()
        #self.args = args
        #self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(512)

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
        self.conv5 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        #nvtx.range_push("forward_default")
        batch_size = x.size(0)
        x = get_graph_feature(x, k=20)
        print(x.shape)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=20)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=20)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=20)
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
        x = get_graph_feature_DA(x, idx=nn_idx)

        print("(get_model) neighbor search output:", x.shape)
        x1 = x
        print("concat:", x1.shape)
        
        print("--------------------------------------------------------------------\nm2")

        batch_size = x1.size(0)
        dim = x1.size(1)
        num_points = x1.size(2) 
        
        nn_idx = knn(x1.view(batch_size,dim,num_points))       

        print("(get_model) conv2d input:", x1.shape)
        x2 = self.conv2(x1)
        print("(get_model) conv2d output:", x2.shape)
        x2 = get_graph_feature_DA(x2, idx=nn_idx)

        print("(get_model) neighbor search output:", x2.shape)

        print("concat:", x2.shape)
        print("--------------------------------------------------------------------\nm3")
        batch_size = x2.size(0)
        dim = x2.size(1)
        num_points = x2.size(2) 
        
        nn_idx = knn(x2.view(batch_size,dim,num_points))  

        print("(get_model) conv2d input:", x2.shape)
        x3 = self.conv3(x2)
        print("(get_model) conv2d output:", x3.shape)
        x3 = get_graph_feature_DA(x3, idx=nn_idx)

        print("(get_model) neighbor search output:", x3.shape)

        print("concat:", x3.shape)
        print("--------------------------------------------------------------------\nm4")
        batch_size = x3.size(0)
        dim = x3.size(1)
        num_points = x3.size(2) 
        
        nn_idx = knn(x3.view(batch_size,dim,num_points))  

        print("(get_model) conv2d input:", x3.shape)
        x4 = self.conv4(x3)
        print("(get_model) conv2d output:", x4.shape)
        x4 = get_graph_feature_DA(x4, idx=nn_idx)

        print("(get_model) neighbor search output:", x4.shape)

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
        print("(get_model) fc1 output:", x.shape)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        print("(get_model) fc2 output:", x.shape)
        x = self.dp2(x)
        x = self.linear3(x)
        print("(get_model) fc3 output:", x.shape)
        #nvtx.range_pop()
        return x
    
if __name__=='__main__':
    dgcnn = DGCNN_DA()
    x=(torch.rand(40, 3, 1024))
    #x=(torch.arange(1,61).view(5,4,3))
    #x=x.transpose(1,2)
    knn_1 = knn(x,2)
    #print(knn_1.size())
    #
    #x=(torch.arange(64,0,-1).view(4,4,4))
    #x=x.transpose(1,2)
    #
    #ggf = get_graph_feature_DA(x,2,knn_1)
    #ggf_1 = get_graph_feature_DA(x,2)
    #
    #print(ggf_1.size())
    #
    #x = ggf_1.permute(0,2,3,1)
    #pointnet = PointNet2_DA()
    #x=(torch.arange(1,61).view(5,4,3)).type(torch.float32)
    #x=x.transpose(1,2)
    #x=(torch.rand(40, 3, 1024))
    #print("x",x)
    #print(knn_1,"\n",ggf.permute(0,2,3,1)==ggf_1.permute(0,2,3,1))
    #print(ggf_1.permute(0,2,3,1))
    x = dgcnn(x)
    print("--------------------------------------------------------------------\nRESULT")
    print( x.shape)