import torch
import torch.nn as nn
import copy
from mamba_ssm import Mamba
from baseline.module.MLCAM import MLCAM
from baseline.module.LSTM_Layer import LSTMModel
from baseline.module.wavelet import MulCausalTCN
from baseline.module.decom import series_decomp_multi
from baseline.module.CNN_MOE import NOEExpertSystem
import torch.nn.functional as F
# from baseline.module.MOE import MoE_MLP


class AHCMOE(nn.Module):
    def __init__(self, nl, ga, hlkc, ker, moe, bmss, t, com_factor, dim, num_classes):
        super(AHCMOE, self).__init__()
        # (args.nl, args.ga, args.hlkc, args.moe, args.bmss, args.t, args.com_factor, args.features_dim, args.num_class)
        # AHCMOE(args.nl, args.ga, args.hlkc, args.ker, args.moe, args.bmss, args.t, args.com_factor, args.features_dim, args.num_class)
        self.ga = ga
        self.hlkc = hlkc
        self.bmss = bmss
        self.t = t
        self.PatchEmbedding = PatchEmbedding(dim, com_factor, com_factor)
        self.classify = nn.Conv1d(com_factor , num_classes, 1)
        self.conv_out = nn.Conv1d(com_factor, com_factor, 1)
        if self.hlkc:
            self.LargeCNN_UNet = CNNMOE_UNet(com_factor, nl, num_classes, ker, moe)
        if self.bmss:
            self.layers = nn.ModuleList([SingleStageModel(com_factor, com_factor, com_factor) for i in range(2)])
        if self.hlkc and self.bmss:
            self.fc = nn.Linear(com_factor * 2, com_factor)
        if self.t:
            self.at = SMTrans(com_factor, block_size=256)
        if self.ga:
            self.acts = nn.ModuleList([Activation(com_factor) for i in range(2)])

    def forward(self, x):
        out_list_all = []
        x_ = self.PatchEmbedding(x)

        x0 = self.conv_out(x_)

        #gated activate
        if self.ga:
            for act in self.acts:
                x0 = act(x0) + x0

        if self.hlkc:
            out_list, x0 = self.LargeCNN_UNet(x0)
            out_list_all = out_list

        # bmss
        if self.bmss:
            x_m = x_
            for layer in self.layers:
                x_m = layer(x_m)
            out = self.classify(x_m)
            out_list_all.append(out)

        # fusion
        if self.hlkc and self.bmss:
            x_a = torch.cat([x0, x_m], 1)
            x_a = self.fc(x_a.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.hlkc and not self.bmss:
            x_a = x0
        elif not self.hlkc and self.bmss:
            x_a = x_m
        else:
            print("Please configure the parameters correctly!")

        # transformer
        if self.t:
            x_a = self.at(x_a, x_a)

        out = self.classify(x_a)
        out_list_all.append(out)


        return out_list_all

# decomposition

class SMTrans(nn.Module):
    def __init__(self, embed_dim, block_size=256):
        super(SMTrans, self).__init__()
        self.block_size = block_size
        self.attention_module = MLCAM(embed_dim, 8, 2)

    def forward(self, x_a, x0, mask_flag=False):
        B, C, T = x_a.size()

        # 计算需要填充的长度
        pad_len = (self.block_size - (T % self.block_size)) % self.block_size
        if pad_len > 0:
            x_a = F.pad(x_a, (0, pad_len), 'constant', 0)
            x0 = F.pad(x0, (0, pad_len), 'constant', 0)

        # 更新时间序列长度
        T = x_a.size(2)
        num_blocks = T // self.block_size

        # 拆分序列
        x_a_blocks = x_a.view(B, C, num_blocks, self.block_size).permute(0, 2, 1, 3).contiguous().view(B * num_blocks,
                                                                                                       C,
                                                                                                       self.block_size)
        x0_blocks = x0.view(B, C, num_blocks, self.block_size).permute(0, 2, 1, 3).contiguous().view(B * num_blocks, C,
                                                                                                     self.block_size)

        # 应用注意力模块
        attn_output = self.attention_module(x_a_blocks.permute(0, 2, 1), x0_blocks.permute(0, 2, 1),
                                            x0_blocks.permute(0, 2, 1), mask_flag=mask_flag).permute(0, 2, 1)

        # 重塑回原始形状
        attn_output = attn_output.view(B, num_blocks, C, self.block_size).permute(0, 2, 1, 3).contiguous().view(B, C, T)

        # 如果有填充，去掉填充部分
        if pad_len > 0:
            attn_output = attn_output[:, :, :-pad_len]

        return attn_output

class CNNMOE_UNet(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, com_factor, BN, num_classes, ker, moe=1):
        super(CNNMOE_UNet, self).__init__()
        self.BN = BN
        ####
        self.EBS1_2 = nn.ModuleList([copy.deepcopy(CNNMOE(com_factor, com_factor*2**s, com_factor*2**s, ker, moe)) for s in range(self.BN)])
        self.Downsamples_2 = nn.ModuleList([copy.deepcopy(Downsample(3, com_factor*2**s, com_factor*2**(s+1), stride = 3)) for s in range(self.BN-1)])
        self.EBS2_2 = nn.ModuleList(
            [copy.deepcopy(CNNMOE(com_factor, com_factor * 2 ** (self.BN - 1 - s), com_factor * 2 ** (self.BN - 1 - s), ker, moe)) for s in
             range(self.BN)])
        self.Upsamples_2 = nn.ModuleList([copy.deepcopy(Upsample(com_factor * 2 ** (self.BN - 1 - s), com_factor * 2 ** (self.BN - 2 - s))) for s in range(self.BN-1)])

        self.classifys = nn.ModuleList([copy.deepcopy(nn.Conv1d(com_factor*2**(self.BN - 1 - s), num_classes, 1)) for s in range(self.BN)])

    def forward(self, x):
        x_list = []
        out_list = []
        x0 = self.EBS1_2[0](x)
        x_list.append(x0)
        for i in range(self.BN-1):
            x0 = self.Downsamples_2[i](x0)
            x0 = self.EBS1_2[i+1](x0)
            x_list.append(x0)

        x_list1_2 = []
        x1 = x0
        x1 = self.EBS2_2[0](x1)
        x_list1_2.append(x1)
        out = self.classifys[0](x1)
        out_list.append(out)
        for i in range(self.BN-1):
            x1 = self.Upsamples_2[i](x1, x_list[self.BN - 2 - i])
            x1 = self.EBS2_2[i + 1](x1)
            x_list1_2.append(x1)
            out = self.classifys[i+1](x1)
            out_list.append(out)
        return out_list, x1


#######  MoE
class MLPExpert(nn.Module):
    """
    A single expert in the Mixture of Experts (MoE) system.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MLPExpert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GatingNetwork(nn.Module):
    """
    The gating network in the Mixture of Experts (MoE) system.
    """

    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


class MoE_MLP(nn.Module):
    """
    Mixture of Experts (MoE) system.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, dropout_rate=0.5):
        super(MoE_MLP, self).__init__()
        self.experts = nn.ModuleList(
            [MLPExpert(input_dim, hidden_dim, output_dim, dropout_rate) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        # x: [batch_size, feature_dim, seq_len]
        batch_size, feature_dim, seq_len = x.size()

        # Flatten the input for the gating network
        x_flat = x.permute(0, 2, 1).contiguous().view(batch_size * seq_len, feature_dim)

        # Get the weights from the gating network
        weights = self.gating_network(x_flat)  # [batch_size * seq_len, num_experts]

        # Get the outputs from each expert
        expert_outputs = [expert(x_flat) for expert in self.experts]  # List of [batch_size * seq_len, output_dim]

        # Combine the outputs using the weights
        combined_output = torch.zeros_like(expert_outputs[0])
        for i, expert_output in enumerate(expert_outputs):
            combined_output += weights[:, i].unsqueeze(-1) * expert_output

        # Reshape back to the original shape
        combined_output = combined_output.view(batch_size, seq_len, -1).permute(0, 2, 1).contiguous()

        return combined_output



class Downsample(nn.Module):
    def __init__(self, kernel_size, in_f_maps, out_f_maps, stride=3):
        super(Downsample, self).__init__()
        self.conv = nn.Conv1d(in_f_maps, out_f_maps, 1)
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        # self.avgpool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        # x = self.avgpool(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_f_maps, out_f_maps):
        super(Upsample, self).__init__()
        self.conv = nn.Conv1d(in_f_maps, out_f_maps, 1)
        self.para = nn.Parameter(torch.zeros(1, requires_grad=True))

    def _upsample_add(self, x, y):
        _,_,W = y.size()
        return F.interpolate(x, size=W, mode='linear')*self.para + y

    def forward(self, x, x0):
        x = self.conv(x)
        x = self._upsample_add(x, x0)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_f_maps, com_factor, out_f_maps):
        super(PatchEmbedding, self).__init__()
        self.channel_dropout = nn.Dropout2d()
        self.conv_in = nn.Conv1d(in_f_maps, com_factor, 1)
        # self.conv_out = nn.Conv1d(com_factor, out_f_maps, 1)
        # self.layers = nn.ModuleList([Activation(com_factor) for i in range(2)])

    def forward(self, x):
        x = x.unsqueeze(3)  # of shape (bs, c, l, 1)
        x = self.channel_dropout(x)
        x = x.squeeze(3)
        out = self.conv_in(x)
        # out0 = out
        # for layer in self.layers:
        #     out = layer(out) + out
        # out = self.conv_out(out)
        return out


class Activation(nn.Module):
    def __init__(self, com_factor):
        super(Activation, self).__init__()
        # self.mamba = Mamba(d_model=com_factor, d_state=16, d_conv=4, expand=2)
        base_channels = com_factor // 8
        # Bottleneck
        self.conv_1 = nn.Conv1d(com_factor, com_factor*4, 1)
        self.conv_2 = nn.Conv1d(com_factor*4, com_factor, 1)
        self.conv1 = nn.Conv1d(com_factor, base_channels, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=4, dilation=4)
        self.conv3 = nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=8, dilation=8)
        self.conv_1x1_output = nn.Conv1d(com_factor + base_channels * 3, com_factor, 1)
        # self.LSTMModel = LSTMModel(com_factor, com_factor*4, 2, com_factor)
        self.ActivationBlock = Activation_Block(com_factor, com_factor * 4, 2, com_factor)


    def forward(self, x):
        # FCTF
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([x, x1, x2, x3], dim=1)
        x = self.conv_1x1_output(out)
        x = self.ActivationBlock(x.permute(0, 2, 1)).permute(0, 2, 1)  # 错误
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


########################################################################################################################
class Activation_Block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, bias: bool = True):
        super(Activation_Block, self).__init__()
        # 定义输出层
        self.fc = nn.Linear(input_size, output_size)
        self.num_directions = 1
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            # 第一层的输入维度是input_size，其他层是hidden_size * num_directions
            layer_input_size = input_size
            # 添加前向层
            self.layers.append(Activation_Layer(layer_input_size, hidden_size, bias))


    def forward(self, x):
        current_input = x
        for layer_idx in range(self.num_layers):

            forward_layer = self.layers[layer_idx * self.num_directions]
            forward_output = forward_layer(current_input)         #  第二个是：torch.Size([1, 1409, 320])
            layer_output = forward_output


        # 应用dropout（除了最后一层）
            if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            current_input = layer_output

        out = self.fc(current_input)

        return out



class Activation_Layer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = None, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else input_size

        self.linear = nn.Linear(input_size, 4 * input_size, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 最简洁的方法，完全避免 reshape
        gates = self.linear(x)  # [B, T, 4*C]
        gates = torch.sigmoid(gates)

        # 在最后一个维度上分割
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=2)

        c_t = f_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t

########################################################################################################################


class CNNMOE(nn.Module):
    def __init__(self, com_factor, in_f_maps, out_f_maps, ker, moe = 1):
        super(CNNMOE, self).__init__()
        self.moe = moe
        self.channel_dropout = nn.Dropout2d()
        # Bottleneck
        self.conv_in = nn.Conv1d(in_f_maps, com_factor, 1)
        self.conv_out = nn.Conv1d(com_factor, out_f_maps, 1)
        self.conv_m = nn.Conv1d(com_factor, com_factor, kernel_size=3, padding=2, dilation=2)  # 非因果推断

        self.LocalBlock = LocalBlock(com_factor, ker)
        self.Fu = torch.nn.Linear(2*com_factor, com_factor)
        if self.moe:
            self.MoE = MoE_MLP(com_factor, com_factor*4, out_f_maps, 4)
        else:
            self.mlp = MLPExpert(com_factor, com_factor*4, out_f_maps, dropout_rate=0.5)


    def forward(self, x):
        x = x.unsqueeze(3)  # of shape (bs, c, l, 1)
        x = self.channel_dropout(x)
        x = x.squeeze(3)

        out = self.conv_in(x)
        out = self.LocalBlock(out)
        out = self.conv_m(out) + out
        if self.moe:
            out = self.MoE(out)
        else:
            out = self.mlp(out.permute(0, 2, 1)).permute(0, 2, 1)
        return out


class SingleStageModel(nn.Module):
    def __init__(self, com_factor, dim, out_f_maps):
        super(SingleStageModel, self).__init__()
        base_channels = com_factor // 8

        # Bottleneck
        self.conv_1x1 = nn.Conv1d(dim, com_factor, 1)
        self.conv_out = nn.Conv1d(com_factor, out_f_maps, 1)

        # FCTF
        self.conv1 = nn.Conv1d(com_factor, base_channels, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=4, dilation=4)
        self.conv3 = nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=8, dilation=8)
        self.conv_1x1_output = nn.Conv1d(com_factor + base_channels * 3, com_factor, 1)

        self.layers = nn.ModuleList([Mamba(d_model=com_factor, d_state=16, d_conv=4, expand=2) for i in range(1)])
        # self.LocalBlock = LocalBlock(com_factor)

    def forward(self, x):
        # Bottleneck
        out = self.conv_1x1(x)

        # FCTF
        x1 = self.conv1(out)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([out, x1, x2, x3], dim=1)
        out = self.conv_1x1_output(out)
        # out = self.LocalBlock(out)
        # Mamba
        out = out.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        for layer in self.layers:
            out = layer(out)

        # Bottleneck
        out = self.conv_out(out.permute(0, 2, 1))
        return out


class CNNKernelBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3,
                 padding='same', use_norm=True, activation='relu'):
        """
        大核卷积块（包含卷积、归一化和激活函数）

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            padding: 填充方式
            use_norm: 是否使用归一化
            activation: 激活函数类型
        """
        super(CNNKernelBlock, self).__init__()

        # 计算填充
        if padding == 'same':
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=1  # 固定为1，不使用空洞卷积
        )

        # 归一化层
        # self.norm = nn.BatchNorm1d(out_channels) if use_norm else nn.Identity()
        self.norm = CustomBatchNorm1d(out_channels) if use_norm else nn.Identity()
        # nn.LayerNorm(d)

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        # x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.activation(x)
        return x

class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(CustomBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

        # 运行时的均值和方差
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1))

        # 重置参数
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        assert x.shape[1] == self.num_features, f"Expected {self.num_features} features, got {x.shape[1]}"

        if self.training:
            # 计算当前mini-batch的均值和方差
            mean = x.mean(dim=(0, 2), keepdim=True)
            var = x.var(dim=(0, 2), unbiased=False, keepdim=True)

            # 更新运行时的均值和方差
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 在评估模式下使用运行时的统计量
            mean = self.running_mean
            var = self.running_var

        # mean = x.mean(dim=(0, 2), keepdim=True)
        # var = x.var(dim=(0, 2), unbiased=False, keepdim=True)

        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 缩放和偏移
        result = self.weight * x_norm + self.bias

        return result


class LocalBlock(nn.Module):
    def __init__(self, com_factor, ker):
        super(LocalBlock, self).__init__()
        com_factor1 = int(com_factor//8)
        self.conv_in = nn.Conv1d(com_factor, com_factor1, 1)
        self.conv_l = CNNKernelBlock(com_factor1, com_factor1, kernel_size = ker)
        self.conv_out = nn.Conv1d(com_factor1, com_factor, 1)


    def forward(self, x):
        x = self.conv_in(x)
        x_s = x
        x_s = self.conv_l(x_s)
        out = self.conv_out(x_s)
        return out




# if __name__ == "__main__":
#     from utils import parameter_count_table
#
#     model = MultiStageModel(3, 64, 1000, 1)
#     print(parameter_count_table(model))


