
import torch 
import torch.nn as nn
from utils.tools import extract_image_patches,\
    reduce_mean, reduce_sum, same_padding
    
inp = torch.tensor([[[[1.0,  2,  3,  4,  5,  6],
                      [ 7,  8,  9, 10, 11, 12],
                      [13, 14, 15, 16, 17, 18],
                      [19, 20, 21, 22, 23, 24],
                      [25, 26, 27, 28, 29, 30],
                      ]]])
print('inp=')
print(inp.size())
 
unfold = nn.Unfold(kernel_size=(3, 3), dilation=1, padding=0, stride=(1, 1))
inp_unf = unfold(inp)
inp_unf = inp_unf.view(inp.shape[0], inp.shape[1], 3, 3, -1)
inp_unf = inp_unf.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
w_i_groups = torch.split(inp_unf, 1, dim=0)
max_wi = torch.pow(w_i_groups[0][0], 2)
max_wi = reduce_sum(max_wi,axis=[1, 2, 3],keepdim=True)
max_wi = torch.sqrt(max_wi)
max_wi = torch.max(max_wi,torch.FloatTensor([1e-4]))

print('inp_unf=')
print(max_wi)