# https://github.com/BM-AI-Lab/LSSeg/blob/main/models/lsseg.py

import torch
from torch import nn
from einops import rearrange

from utils import Smish


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class RFCAConv(nn.Module):
    def __init__(self, inp, oup,kernel_size,stride, reduction=32):
        super(RFCAConv, self).__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(inp,inp * (kernel_size**2),kernel_size,padding=kernel_size//2,
                                                stride=stride,groups=inp,
                                                bias =False),
                                      nn.BatchNorm2d(inp * (kernel_size**2)),
                                      nn.ReLU()
                                      )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp,oup,kernel_size,stride=kernel_size))
        

    def forward(self, x):
        b,c = x.shape[0:2]
        generate_feature = self.generate(x)
        h,w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b,c,self.kernel_size**2,h,w)
        
        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)
        
        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        h,w = generate_feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return self.conv(generate_feature * a_w * a_h)


class RFCADenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        """
        Dense block with RFCA added in the first layer
        num_convs:      Number of convolutional blocks in the dense block
        input_channels: Number of input channels
        num_channels:   Number of output channels per convolutional block
        """
        super(RFCADenseBlock, self).__init__()
        layers = [RFCAConv(input_channels, num_channels, kernel_size=3, stride=1)]
        for i in range(1, num_convs):
            layers.append(self.conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layers)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            Smish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat([X, Y], dim=1)    # BxCxHxW
        return X


class DWSConv(nn.Module):
    """
    Depthwise Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DWSConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                        stride=stride, padding=kernel_size // 2, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, X):
        return self.pointwise_conv(self.depthwise_conv(X))


class TransBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        """
        Reduce height and width by half
        input_channels:  Number of input channels to the transition layer
        output_channels: Number of output channels from the transition layer
        """
        super(TransBlock, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            Smish(),
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, X):
        return self.net(X)


class DownSample(nn.Module):
    """
    TImE downsampling block, output height and width reduced by half
    num_convs: Number of convolutional layers in the dense block
    """
    def __init__(self, num_convs, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.dense_block = RFCADenseBlock(num_convs, in_channels, num_channels=out_channels)
        self.trans_block = TransBlock(num_convs * out_channels + in_channels, out_channels)
    
    def forward(self, X):
        return self.trans_block(self.dense_block(X))


class UpSample(nn.Module):
    """
    TImE upsampling block, output height and width doubled
    num_channels: Number of output channels for DWS convolution, must be a multiple of 4
    """
    def __init__(self, in_channels, out_channels, upscale=2):
        super(UpSample, self).__init__()
        self.dws_conv = DWSConv(in_channels, out_channels * upscale**2, kernel_size=3)
        self.ps = nn.PixelShuffle(upscale_factor=upscale)
        self.af = Smish()
    
    def forward(self, X):
        return self.ps(self.dws_conv(self.af(X)))


# class FoL(nn.Module):
#     """
#     Focus Locally (FoL) block
#     """
#     def __init__(self, in_channels):
#         super(FoL, self).__init__()
#         assert in_channels % 8 == 0, 'in_channels must be a multiple of eight.'
#         self.lkp = LKP(in_channels, lks=7, sks=3, groups=8)
#         self.ska = SKA()
#         self.bn = nn.BatchNorm2d(in_channels)

#     def forward(self, x):
#         return self.bn(self.ska(x, self.lkp(x))) + x


class GSC(nn.Module):
    """
    the Gated Skip Connection
    """
    def __init__(self, in_channels):
        super(GSC, self).__init__()
        self.gate_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.smish = Smish()

    def forward(self, X, Y):
        merged = torch.cat([X, Y], dim=1)
        gate = self.sigmoid(self.gate_conv(merged))
        Xgsc = self.smish(X * gate + Y * (1 - gate))
        return Xgsc


class FDM(nn.Module):
    """
    the Feature Decimation Module
    """
    def __init__(self, num_convs, in_channels, out_channels):
        super(FDM, self).__init__()
        self.dense_block = RFCADenseBlock(num_convs, in_channels, num_channels=out_channels)
        self.trans_block = TransBlock(num_convs * out_channels + in_channels, out_channels)
    
    def forward(self, X):
        return self.trans_block(self.dense_block(X))


class RM(nn.Module):
    """
    the Reconstruction Module
    """
    def __init__(self, in_channels, out_channels, upscale=2):
        super(RM, self).__init__()
        # self.fol = FoL(in_channels)
        # self.dpuu1 = UpSample(in_channels, out_channels, upscale)
        self.dpuu2 = UpSample(in_channels, out_channels, upscale)
        # self.fuse = nn.Conv2d(2 * out_channels, out_channels, kernel_size=5, padding=2)
    
    def forward(self, X):
        # X1 = self.dpuu1(self.fol(X))    # path 1
        X2 = self.dpuu2(X)              # path 2
        # return self.fuse(torch.cat([X1, X2], dim=1))
        return X2



class LSSeg(nn.Module):
    """
    the architecture of Line-like Structures Segmentation Network
    Params:
        in_channels: the no. channels of input images.
        len(in_channels): the no. FDM and RM pairs.
    """
    def __init__(self, in_channels=[3, 8, 8]):
        super(LSSeg, self).__init__()
        self.K = len(in_channels)
        self.FDMs = nn.ModuleList()
        self.RMs = nn.ModuleList()
        self.GSCs = nn.ModuleList()
        
        for i in range(self.K):
            # build FDMs
            if i != self.K - 1:
                self.FDMs.append(FDM(4, in_channels=in_channels[i], out_channels=in_channels[i + 1]))
            else:
                self.FDMs.append(FDM(4, in_channels=in_channels[i], out_channels=in_channels[i]))
            
            # build RMs
            if i == 0:
                self.RMs.append(RM(in_channels=in_channels[i + 1], out_channels=1))
            elif i == self.K - 1:
                self.RMs.append(RM(in_channels=in_channels[i], out_channels=in_channels[i]))
            else:
                self.RMs.append(RM(in_channels=in_channels[i + 1], out_channels=in_channels[i]))
            
            # build GSCs
            if i != 0:
                self.GSCs.append(GSC(in_channels[i]))

        self.af = Smish()
        self.apply(self.init_weights)

    def forward(self, X):
        X_Fs = []
        for i in range(self.K):
            X = self.FDMs[i](X)
            X_Fs.append(X)

        for i in range(self.K - 1, -1, -1):
            if i == self.K - 1:
                X = self.RMs[i](X_Fs[i])
            else:
                X = self.RMs[i](self.GSCs[i](X_Fs[i], X))
                 
        # return self.af(X)
        return X

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in')