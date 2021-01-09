import torch
import torch.nn as nn
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
sys.path.append(os.getcwd()) #把mmdetection-master这个folder，也就是工作路径加入模块搜索路径
from mmdet.ops import DeformConv


channel = 1
kernal = (1,9)
padiding = (0,4)
offset_num = kernal[0]*kernal[1]*2
base_offset = torch.tensor([0,-4, 0,-3, 0,-2, 0,-1, 0,0, 0,1, 0,2, 0,3, 0,4]).float().cuda()
target_offset = torch.tensor([-1,-1 , -1,0, -1,1,
                0,-1, 0,0, 0,1,
                1,-1, 1,0, 1,1]).float().cuda()
target_offset -=  base_offset   
in_w = 10
in_h = 10
out_w = in_w+padiding[1]*2 - kernal[1] + 1     
out_h = in_h+padiding[0]*2 - kernal[0] + 1      

ones = torch.Tensor(np.ones([channel,channel,kernal[0],kernal[1]])).cuda() # 先创建一个自定义权值的Tensor，这里为了方便将所有权值设为1
dcn_layer = DeformConv(channel, channel,kernal , 1, padiding).cuda()
dcn_layer.weight = torch.nn.Parameter(ones)

input = torch.range(1,in_w).cuda()
input = input[..., None].repeat(1, in_w)
input = input[None, None, ...]

# offset = torch.zeros((1,offset_num,out_h,out_w)).float().cuda()

offset = target_offset
offset = offset[None,...,None, None]
offset = offset.repeat(1,1,out_h,out_w)

print('input', input.size())
print('offset', offset.size())
output = dcn_layer(input, offset)
# output = output[:,:,padiding:-(padiding),:]
# output = cn_layer(input)

print('input', input)
print('output', output)
print('offset', offset[0,:,0,0])
print('input', input.size())
print('output', output.size())
# print('weight', dcn_layer.weight)