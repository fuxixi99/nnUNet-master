import torch
import torch.nn as nn

class FrequencySelection3D(nn.Module):
    def __init__(self, 
                 in_channels,
                 k_list=[2],
                 lowfreq_att=True,
                 fs_feat='feat',
                 lp_type='freq',
                 act='sigmoid',
                 spatial='conv',
                 spatial_group=1,
                 spatial_kernel=3,
                 init='zero',
                 global_selection=False):
        super().__init__()
        self.k_list = k_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.in_channels = in_channels
        if spatial_group > 64: 
            spatial_group = in_channels
        self.spatial_group = spatial_group
        self.lowfreq_att = lowfreq_att
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att: 
                _n += 1
            for i in range(_n):
                freq_weight_conv = nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=self.spatial_group, 
                    stride=1,
                    kernel_size=spatial_kernel, 
                    groups=self.spatial_group,
                    padding=spatial_kernel // 2, 
                    bias=True
                )
                if init == 'zero':
                    # freq_weight_conv.weight.data.zero_()
                    # freq_weight_conv.bias.data.zero_()   
                    nn.init.kaiming_uniform_(freq_weight_conv.weight, mode='fan_in', nonlinearity='relu')  # He 初始化
                    freq_weight_conv.bias.data.uniform_(0.1, 0.2)  # 偏置初始化为小正值
                self.freq_weight_conv_list.append(freq_weight_conv)
        else:
            raise NotImplementedError
        
        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                    nn.ReplicationPad3d(padding=k // 2),
                    nn.AvgPool3d(kernel_size=k, padding=0, stride=1)
                ))
        elif self.lp_type == 'laplacian':
            pass
        elif self.lp_type == 'freq':
            pass
        else:
            raise NotImplementedError
        
        self.act = act
        
        self.global_selection = global_selection
        if self.global_selection:
            self.global_selection_conv_real = nn.Conv3d(
                in_channels=in_channels, 
                out_channels=self.spatial_group, 
                stride=1,
                kernel_size=1, 
                groups=self.spatial_group,
                padding=0, 
                bias=True
            )
            self.global_selection_conv_imag = nn.Conv3d(
                in_channels=in_channels, 
                out_channels=self.spatial_group, 
                stride=1,
                kernel_size=1, 
                groups=self.spatial_group,
                padding=0, 
                bias=True
            )
            if init == 'zero':
                self.global_selection_conv_real.weight.data.zero_()
                self.global_selection_conv_real.bias.data.zero_()  
                self.global_selection_conv_imag.weight.data.zero_()
                self.global_selection_conv_imag.bias.data.zero_()  

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            # freq_weight = freq_weight.sigmoid() * 2
            freq_weight = freq_weight.sigmoid()
        elif self.act == 'softmax':
            # freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
            freq_weight = freq_weight.softmax(dim=1)
        else:
            raise NotImplementedError
        return freq_weight

    def forward(self, x, att_feat=None):
        if att_feat is None: 
            att_feat = x
        x_list = []
        if self.lp_type == 'avgpool':
            pre_x = x
            b, _, d, h, w = x.shape
            for idx, avg in enumerate(self.lp_list):
                low_part = avg(x)
                high_part = pre_x - low_part
                pre_x = low_part
                freq_weight = self.freq_weight_conv_list[idx](att_feat)
                freq_weight = self.sp_act(freq_weight)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, d, h, w) * high_part.reshape(b, self.spatial_group, -1, d, h, w)
                x_list.append(tmp.reshape(b, -1, d, h, w))
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, d, h, w) * pre_x.reshape(b, self.spatial_group, -1, d, h, w)
                x_list.append(tmp.reshape(b, -1, d, h, w))
            else:
                x_list.append(pre_x)
        elif self.lp_type == 'freq':
            pre_x = x.clone()
            b, _, d, h, w = x.shape
            x_fft = torch.fft.fftshift(torch.fft.fftn(x, dim=(2, 3, 4), norm='ortho'))
            if self.global_selection:
                x_real = x_fft.real
                x_imag = x_fft.imag
                global_att_real = self.global_selection_conv_real(x_real)
                global_att_real = self.sp_act(global_att_real).reshape(b, self.spatial_group, -1, d, h, w)
                global_att_imag = self.global_selection_conv_imag(x_imag)
                global_att_imag = self.sp_act(global_att_imag).reshape(b, self.spatial_group, -1, d, h, w)
                x_real = x_real.reshape(b, self.spatial_group, -1, d, h, w)
                x_imag = x_imag.reshape(b, self.spatial_group, -1, d, h, w)
                x_fft_real_updated = x_real * global_att_real
                x_fft_imag_updated = x_imag * global_att_imag
                x_fft_updated = torch.complex(x_fft_real_updated, x_fft_imag_updated)
                x_fft = x_fft_updated.reshape(b, -1, d, h, w)

            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :, :], device=x.device)
                mask[:, :, 
                     round(d/2 - d/(2 * freq)):round(d/2 + d/(2 * freq)), 
                     round(h/2 - h/(2 * freq)):round(h/2 + h/(2 * freq)), 
                     round(w/2 - w/(2 * freq)):round(w/2 + w/(2 * freq))] = 1.0
                low_part = torch.fft.ifftn(torch.fft.ifftshift(x_fft * mask), dim=(2, 3, 4), norm='ortho').real
                high_part = pre_x - low_part
                pre_x = low_part
                freq_weight = self.freq_weight_conv_list[idx](att_feat)
                freq_weight = self.sp_act(freq_weight)
                freq_weight = freq_weight * self.attention(att_feat)  # 使用注意力引导
                freq_weight_loss = torch.mean(freq_weight ** 2)  # 添加正则化
                # print(f"freq_weight at layer {idx}: {freq_weight.mean().item()}")
                tmp = freq_weight.reshape(b, self.spatial_group, -1, d, h, w) * high_part.reshape(b, self.spatial_group, -1, d, h, w)
                x_list.append(tmp.reshape(b, -1, d, h, w))
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
                freq_weight = self.sp_act(freq_weight)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, d, h, w) * pre_x.reshape(b, self.spatial_group, -1, d, h, w)
                x_list.append(tmp.reshape(b, -1, d, h, w))
            else:
                x_list.append(pre_x)
        x = sum(x_list)
        # print()
        # output_string="freq_weight"+str(freq_weight)+"\n"
        # with open(f"freq_test.txt", "a") as file:
        #     file.write(output_string)
        output_string = f"freq_weight mean: {freq_weight.mean().item()}\n"
        with open("freq_test11.txt", "a") as file:
            file.write(output_string)

        return x
if __name__ == '__main__':
    x = torch.rand(2, 4, 16,16, 16).cuda()
    # m = AdaptiveDilatedConv(in_channels=4, out_channels=8, kernel_size=3).cuda()
    m=FrequencySelection3D(in_channels=4).cuda()
    m.eval()
    y = m(x)
    print(y.shape)