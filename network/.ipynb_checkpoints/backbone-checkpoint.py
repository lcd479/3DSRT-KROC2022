from network.SwinTransformer import *

import torch
import torch.nn.functional as F
import torch.nn as nn


class backbone(nn.Module):
    
    def __init__(self, img_size = (640,480)):
        super(backbone, self).__init__()
        
        self.img_size = img_size
        
        self.ST = SwinTransformer(self.img_size)
        
        in_c = [384, 192, 96]
        out_c = [80, 40, 24]
    
        self.inner0 = nn.Conv2d(in_c[1], out_c[0], 1,bias = True)s
        self.inner1 = nn.Conv2d(in_c[2], out_c[0], 1,bias = True)

        self.out0 = nn.Conv2d(in_c[0], out_c[0], 1, bias = False)
        self.out1 = nn.Conv2d(out_c[0], out_c[1], 1, bias = False)
        self.out2 = nn.Conv2d(out_c[0], out_c[2], 1, bias = False)

    def forward(self, x):
        
        out_list = self.ST(x)
        
        T_out0 = out_list[0]
        T_out1 = out_list[1]
        T_out2 = out_list[2]
        
        intra_feat = T_out2
        outputs = []
        out = self.out0(intra_feat)
        outputs.append(out.transpose(2,3))
        

        intra_feat = F.interpolate(out, scale_factor=2, mode='nearest') + self.inner0(T_out1)
        out = self.out1(intra_feat)
        outputs.append(out.transpose(2,3))
        
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode='nearest') + self.inner1(T_out0)
        out = self.out2(intra_feat)
        outputs.append(out.transpose(2,3))

        
        return outputs[::-1]