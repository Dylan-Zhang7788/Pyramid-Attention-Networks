import torch
import torch.nn as nn
inp = torch.tensor([[[[1.0, 2,  3,  4,  5,  6],
                      [ 7,  8,  9, 10, 11, 12],
                      [13, 14, 15, 16, 17, 18],
                      [19, 20, 21, 22, 23, 24],
                      [25, 26, 27, 28, 29, 30],
                      ],
                      [[1.0, 21,  31,  4,  5,  6],
                      [ 72,  8,  9, 10, 11, 12],
                      [13, 14, 15, 16, 17, 18],
                      [19, 20, 21, 22, 23, 24],
                      [25, 26, 27, 28, 29, 30],
                      ]]])
print('inp=')
print(inp.shape)
 
unfold = nn.Unfold(kernel_size=(3, 3), dilation=(1,1), padding=0, stride=(1, 1))
inp_unf = unfold(inp)
print('inp_unf=')
print(inp_unf.shape)
print(inp_unf)