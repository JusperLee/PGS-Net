import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .utils import overlap_and_add
from .base_model import BaseModel

EPS = 1e-8


def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class av_convtasnet_causal(BaseModel):
    def __init__(self, N=256, L=40, B=256, H=512, P=3, X=8, R=4, C=2, sample_rate=16000):
        super(av_convtasnet_causal, self).__init__(sample_rate=sample_rate)

        self.encoder = Encoder(L, N)
        self.separator = TemporalConvNet(N, B, H, P, X, R, C)
        self.decoder = Decoder(N, L)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, audio_mixture, mouth_embedding):
        mixture_w = self.encoder(audio_mixture)    #mixture_w: [1, 256, 799]    mixture: [1, 16000]    
        est_mask = self.separator(mixture_w, mouth_embedding)   # est_mask: [1, 256, 799]  visual: [1, 25, 512]
        est_source = self.decoder(mixture_w, est_mask)  #est_source: [1, 16000]

        # T changed after conv1d in encoder, fix it here
        T_origin = audio_mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        est_source = est_source.view(est_source.size(0), 1, est_source.size(1))
        return est_source
    
    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args


class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1,
                                  N,
                                  kernel_size=L,   #L=40
                                  stride=L // 2,   #L=40//2=20
                                  bias=False)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T] [1, 1, 16000]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]  [1, 256, 799]]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        est_source = mixture_w * est_mask  # [M,  N, K]
        est_source = torch.transpose(est_source, 2, 1)  # [M,  K, N]
        est_source = self.basis_signals(est_source)  # [M,  K, L]
        est_source = overlap_and_add(est_source, self.L // 2)  # M x C x T
        return est_source


class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C):
        super(TemporalConvNet, self).__init__()
        self.C = C
        self.layer_norm = ChannelWiseLayerNorm(N)
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)

        # Audio TCN
        tcn_blocks = []
        tcn_blocks += [nn.Conv1d(B * 2, B, 1, bias=False)]
        for x in range(X):
            dilation = 2**x
            padding = (P - 1) * dilation // 2
            tcn_blocks += [
                TemporalBlock(B,   
                              H,
                              P,
                              stride=1,
                              padding=padding,
                              dilation=dilation)
            ]
        self.tcn = _clones(nn.Sequential(*tcn_blocks), R)

        # visual blocks
        ve_blocks = []
        for x in range(5):
            ve_blocks += [VisualConv1D()]
        self.visual_conv = nn.Sequential(*ve_blocks)

        # Audio and visual seprated layers before concatenation
        self.ve_conv1x1 = _clones(nn.Conv1d(512, B, 1, bias=False), R)

        # Mask generation layer
        self.mask_conv1x1 = nn.Conv1d(B, N, 1, bias=False)

    def forward(self, x, visual):
        visual = self.visual_conv(visual)  # [1, 512, 25]

        x = self.layer_norm(x)          # [1, 256, 799]
        x = self.bottleneck_conv1x1(x)  # [1, 256, 799]

        mixture = x

        batch, B, K = x.size()       #batch: 1, B: 256, K: 799

        for i in range(len(self.tcn)):
            v = self.ve_conv1x1[i](visual)   
            v = F.interpolate(v, (32 * v.size()[-1]), mode='linear')  
            v = F.pad(v, (0, K - v.size()[-1]))  
            x = torch.cat((x, v), 1)  
            x = self.tcn[i](x)  
        x = self.mask_conv1x1(x)  # [1, 256, 799]
        x = F.relu(x)
        return x

class VisualConv1D(nn.Module):
    def __init__(self):
        super(VisualConv1D, self).__init__()
        relu = nn.ReLU()
        norm_1 = nn.BatchNorm1d(512)
        dsconv = nn.Conv1d(512,
                           512,
                           3,
                           stride=1,
                           padding=1,
                           dilation=1,
                           groups=512,
                           bias=False)
        prelu = nn.PReLU()
        norm_2 = nn.BatchNorm1d(512)
        pw_conv = nn.Conv1d(512, 512, 1, bias=False)

        self.net = nn.Sequential(relu, norm_1, dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x


class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size,
                                               1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2,
                                                keepdim=True)  #[M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1,
                                            keepdim=True).mean(dim=2,
                                                               keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation):
        super(TemporalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal_padding = (kernel_size - 1) * dilation
        self.dilation = dilation

        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        #norm = GlobalLayerNorm(out_channels)  
        self.norm = nn.LayerNorm(out_channels)   #causal使用层归一化
        self.dsconv = DepthwiseSeparableConv(
                                            self.out_channels, 
                                            self.in_channels, 
                                            self.kernel_size,
                                            self.stride,
                                            padding=self.causal_padding, 
                                            dilation=self.dilation
                                            )  

    def forward(self, x):

        residual = x
        x = self.conv1x1(x)
        x = self.prelu(x)
        x = x.transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous()
        x = self.dsconv(x)
        out = x[:, :, :-self.causal_padding]
        return out + residual  # look like w/o F.relu is better than w/ F.relu


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation):
        super(DepthwiseSeparableConv, self).__init__()
        depthwise_conv = nn.Conv1d(in_channels,
                                   in_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=False)

        prelu = nn.PReLU()
        norm = GlobalLayerNorm(in_channels)
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    model = av_convtasnet_causal_snr()
    print(model(torch.randn(1, 16000), torch.randn(1, 25, 512)).shape)
