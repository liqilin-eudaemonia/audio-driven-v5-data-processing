from numpy import mod
# from face_modules.model import ResnetModel
import torch
from collections import OrderedDict
import numpy as np
import cv2
import os
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResBlock(Module):
    def __init__(self, c_in, c_out, kernel=3, stride=1, padding=1, bias=True, identity=False):
        super(ResBlock, self).__init__()
        self.bn1 = BatchNorm2d(c_in)
        self.conv1 = Conv2d(c_in, c_out, kernel_size=kernel, stride=1, padding=padding, bias=bias)
        self.prelu = PReLU(c_out)
        self.conv2 = Conv2d(c_out, c_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias)
        # self.conv1 = Sequential(BatchNorm2d(c_in),
        #                         Conv2d(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
        #                         PReLU(),
        #                         Conv2d(c_out, c_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
        
        self.identity = identity
        if self.identity:
            self.conv3 =Conv2d(c_in, c_out, kernel_size=1, stride=2, padding=0, bias=bias)
    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        if self.identity:
            out = out + self.conv3(x)
        else:
            out = out + x
        return out

class ResnetModel(Module):
    def __init__(self):
        super(ResnetModel, self).__init__()
        self.conv_0 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.prelu_0 = PReLU(64)
        channels = [64, 128, 256, 512]
        layer_num= [3,4,14,3]
        self.layer1 = self.make_layer(channels[0],channels[0],layer_num[0])
        self.layer2 = self.make_layer(channels[0],channels[1],layer_num[1])
        self.layer3 = self.make_layer(channels[1],channels[2],layer_num[2])
        self.layer4 = self.make_layer(channels[2],channels[3],layer_num[3])
        self.bn2 = BatchNorm2d(channels[-1])
        self.Flatten = Flatten()
        self.fc = Linear(512 * 7 * 7, 512)
        self.features = BatchNorm1d(512)

    def forward(self, x):
        out = self.conv_0(x)
        out = self.prelu_0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn2(out)
        out = self.Flatten(out)
        out = self.fc(out)
        out = self.features(out)
        return out

    
    def make_layer(self, c_in, c_out, res_num):
        module_list = []
        module_list.append(ResBlock(c_in, c_out, stride=2, identity=True))
        for i in range(1,res_num):
            module_list.append(ResBlock(c_out, c_out))
        return Sequential(*module_list)