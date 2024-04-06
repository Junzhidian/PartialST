import torch.nn as nn
from torch import Tensor
import torch
from timm.models.layers import DropPath
import torch.nn.functional as F
from models.layer.encoders import Inceptionblock, DWblock, Convblock, Dilateblock, DeformConv2d, LayerNorm

class Partialtime(nn.Module):

    def __init__(self, dconf, mconf):
        super().__init__()
        self.dconf = dconf
        self.mconf = mconf
        self.K = self.mconf.K

        pc = self.mconf.transformer_dmodel // self.K
        self.dim_conv = pc
        self.dim_untouched = self.mconf.transformer_dmodel - pc
        encoder_layer = nn.TransformerEncoderLayer(d_model=pc, nhead=self.mconf.transformer_nhead, dropout=self.mconf.transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.mconf.transformer_nlayers)
        self.lstm = nn.LSTM(input_size=pc, hidden_size=pc//2,
                            num_layers=1 , bias=True, bidirectional=True)
        self.gru = nn.GRU(input_size=pc, hidden_size=pc//2,
                            num_layers=1 , bias=True, bidirectional=True)
        self.mlp = nn.Linear(in_features = pc, out_features = pc)
        
        self.split_indexes = (self.mconf.transformer_dmodel - 3 * pc, pc, pc, pc)

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=2)
        x1 = self.transformer_encoder(x1)
        #x1,_ = self.gru(x1)
        #x1,_ = self.lstm(x1)
        #x1 = self.mlp(x1)
        x = torch.cat((x1, x2), 2)
        x = channel_shuffle(x, self.K) 
        return x

class Partialspace(nn.Module):

    def __init__(self, dim, n_div, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv = DWblock(self.dim_conv3)
        #self.partial_conv = Convblock(self.dim_conv3)
        #self.partial_conv = Inceptionblock(self.dim_conv3)
        #self.partial_conv= DeformConv2d(self.dim_conv3,self.dim_conv3)
        #self.partial_conv = Dilateblock(self.dim_conv3)
        self.n_div = n_div

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat((x1, x2), 1)
        x = channel_shuffle(x, self.n_div)

        return x


class Partialblock(nn.Module):
    def __init__(self, dconf, mconf):
        super().__init__()
        self.dconf = dconf
        self.mconf = mconf
        self.L = self.mconf.res_repetation
        self.Conv1 = nn.Conv2d(self.dconf.dim_flow, self.mconf.res_nbfilter, kernel_size=3, stride=1, padding=1)
        self.ResUnits = self._stack_resunits(self.mconf.res_nbfilter)
        self.SeLu = nn.SELU(inplace=True)
        self.Conv2 = nn.Conv2d(self.mconf.res_nbfilter, self.dconf.dim_flow, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(in_features = self.dconf.dim_flow*self.dconf.dim_h*self.dconf.dim_w, out_features = self.mconf.transformer_dmodel)

    def _stack_resunits(self, filter_num):
        layers = []
        for i in range(0, self.L):
            layers.append(Partialspace(filter_num,n_div=self.mconf.K))
        return nn.Sequential(*layers)

    def forward(self, x, ext):
        x = self.Conv1(x)
        ext = ext.reshape(-1, self.mconf.res_nbfilter, self.dconf.dim_h, self.dconf.dim_w)
        out = x + ext
        out = self.ResUnits(out)
        out = self.SeLu(out)
        out = self.Conv2(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out

def channel_shuffle(x, groups):
    if len(x.shape) == 4:
 
        batchsize, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // groups
            
        # reshape
        x = x.view(batchsize, groups, 
            channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)
    
    else:
        len_seq, batchsize, num_channels = x.data.size()
        channels_per_group = num_channels // groups
            
        # reshape
        x = x.view(len_seq, batchsize, groups, 
            channels_per_group)

        x = torch.transpose(x, 2, 3).contiguous()

        # flatten
        x = x.view(len_seq, batchsize, -1)

    return x
    

