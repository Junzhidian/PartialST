import torch
import torch.nn as nn
from models.layer.partialblock import Partialblock,Partialtime
from models.ModelConfiguration import ModelConfigurationTaxiBJ, ModelConfigurationBikeNYC


class PartialST(nn.Module):
    def __init__(self, data_conf):
        super().__init__()

        # config
        self.dconf = data_conf
        if self.dconf.name == 'BikeNYC':
            self.mconf = ModelConfigurationBikeNYC()
        elif self.dconf.name == 'TaxiBJ':
            self.mconf = ModelConfigurationTaxiBJ()
        else:
            raise ValueError('The data set does not exist')

        self.resnns = nn.ModuleList()
        for i in range(self.dconf.len_seq):
            resnn = Partialblock(self.dconf, self.mconf)
            self.resnns.append(resnn)

        self.extnn_inter = nn.Sequential(
            nn.Linear(self.dconf.ext_dim, self.mconf.inter_extnn_inter_channels),
            nn.Dropout(self.mconf.inter_extnn_dropout),
            nn.SELU(),
            nn.Linear(self.mconf.inter_extnn_inter_channels, self.mconf.res_nbfilter*self.dconf.dim_h*self.dconf.dim_w),
            nn.SELU()
        )

        self.time = Partialtime(self.dconf,self.mconf)
        self.FC = nn.Linear(in_features=self.mconf.transformer_dmodel*(self.dconf.len_seq), out_features=self.dconf.dim_flow*self.dconf.dim_h*self.dconf.dim_w)

        self.extnn_last = nn.Sequential(
            nn.Linear(self.dconf.ext_dim, self.mconf.last_extnn_inter_channels),
            nn.SELU(),
            nn.Linear(self.mconf.last_extnn_inter_channels, self.dconf.dim_flow*self.dconf.dim_h*self.dconf.dim_w),
            nn.SELU()
        )

        self.model_name = str(type(self).__name__)

    def forward(self, X, X_ext, Y_ext):
        B,_,_,_ = X.shape
        inputs = torch.split(X, self.dconf.dim_flow, 1)
        ext_outputs = self.extnn_inter(X_ext)
        E_ems = torch.split(ext_outputs, 1, 1)
        
        temp_inputs = []
        for i in range(self.dconf.len_seq):
            X_em = self.resnns[i](inputs[i], E_ems[i].squeeze(1))
            temp_inputs.append(X_em)

        temp_inputs = torch.stack(temp_inputs, 0)
        temp_outputs = self.time(temp_inputs)
        temp_outputs = temp_outputs.transpose(0, 1)
        out = torch.flatten(temp_outputs, 1)
        out = self.FC(out)

        main_out = out.reshape(-1, self.dconf.dim_flow, self.dconf.dim_h, self.dconf.dim_w)
        ext_out = self.extnn_last(Y_ext)
        ext_out = ext_out.reshape(-1, self.dconf.dim_flow, self.dconf.dim_h, self.dconf.dim_w)
        main_out = main_out + ext_out

        main_out = torch.tanh(main_out)
        return main_out

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print("The training model was successfully loaded.")