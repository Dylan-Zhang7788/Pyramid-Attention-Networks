import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import utils as vutils
from model import common
from utils.tools import extract_image_patches,\
    reduce_mean, reduce_sum, same_padding

class PyramidAttention(nn.Module):
    def __init__(self, level=5, res_scale=1, channel=64, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True, conv=common.default_conv):
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = [1-i/10 for i in range(level)]
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = common.BasicBlock(conv,channel,channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = common.BasicBlock(conv,channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv,channel, channel,1,bn=False, act=nn.PReLU())

    def forward(self, input):
        res = input # shape:[1,256,50,50]
        #theta
        match_base = self.conv_match_L_base(input) # [1,32,50,50] 通道数减少为原来的1/8
        shape_base = list(res.size())
        # 把输入分开
        input_groups = torch.split(match_base,1,dim=0) # [1,32,50,50](因为只有一张图)
        # patch size for matching 
        kernel = self.ksize
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        #build feature pyramid
        for i in range(len(self.scale)): # [1.0, 0.9, 0.8, 0.7, 0.6]
            ref = input
            if self.scale[i]!=1:
                ref  = F.interpolate(input, scale_factor=self.scale[i], mode='bicubic')
            #feature transformation function f
            base = self.conv_assembly(ref) # 通道数不变 
            shape_input = base.shape
            #sampling
            # raw_w_i：[N, C*k*k, L] [batchsize，单个卷积核的参数量，卷积核的数目]
            raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                      strides=[self.stride,self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
            # raw_w_i：[N,C,k,k,L] [batchsize，通道数，k，k，卷积核个数]
            raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
            # raw_w_i的形状为 [batchsize，卷积核个数，通道数，k，k]
            raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
            raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
            raw_w.append(raw_w_i_groups)

            #feature transformation function g
            ref_i = self.conv_match(ref) # 通道数减少为原来的1/8
            shape_ref = ref_i.shape
            #sampling
            w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
            w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
            w_i = w_i.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
            w_i_groups = torch.split(w_i, 1, dim=0)
            w.append(w_i_groups)

        y = []
        for idx, xi in enumerate(input_groups): #input_groups n个 [1,32,50,50]
            #idx 的值是1-n
            #group in a filter 
            wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))],dim=0)  # [L, ··, k, k]
            #normalize 对每个维度都平方求和，然后和NAN比较，避免0的出现
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi/ max_wi
            #matching
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            yi = yi.view(1,wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax matching score
            yi = F.softmax(yi*self.softmax_scale, dim=1)
            
            if self.average == False:
                yi = (yi == yi.max(dim=1,keepdim=True)[0]).float()
            
            # deconv for patch pasting
            raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))],dim=0)
            # 这里的转置卷积目的是把输入和输出的通道数目换一下，因为raw_i的通道数是[8250,256,3,3]
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride,padding=1)
            yi=yi/4.
            y.append(yi)
      
        y = torch.cat(y, dim=0)+res*self.res_scale  # back to the mini-batch
        return y