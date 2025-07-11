import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from .espnet2.complex_utils import is_complex, new_complex_like
from .espnet2.abs_separator import AbsSeparator
from .espnet2.get_layer_from_string import get_layer

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


from .espnet2.stft_decoder import STFTDecoder
from .espnet2.stft_encoder import STFTEncoder
from .base_model import BaseModel

EPS = 1e-8
import copy
def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class av_TFGridNetV3_causal(BaseModel):
    """Offline TFGridNetV3.

    On top of TFGridNetV2, TFGridNetV3 slightly modifies the internal architecture
    to make the model sampling-frequency-independent (SFI). This is achieved by
    making all network layers independent of the input time and frequency dimensions.

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in TASLP, 2023.
    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in ICASSP, 2023.

    NOTES:
    As outlined in the Reference, this model works best when trained with variance
    normalized mixture input and target, e.g., with mixture of shape [batch, samples,
    microphones], you normalize it by dividing with torch.std(mixture, (1, 2)). You
    must do the same for the target signals. It is encouraged to do so when not using
    scale-invariant loss functions such as SI-SDR.
    Specifically, use:
        std_ = std(mix)
        mix = mix / std_
        tgt = tgt / std_

    Args:
        input_dim: placeholder, not used
        n_srcs: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of TFGridNetV3 blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_attn_qk_output_channel: output channels of point-wise conv2d for getting
            key and query
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole TFGridNetV3 model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        eps: small epsilon for normalization layers.
        use_builtin_complex: whether to use builtin complex type or not.
    """

    def __init__(
            self,
            n_fft=256,
            stride=128,
            window="hann",
            use_builtin_complex=False,
            n_srcs =1,
            n_imics=1,
            n_layers=6,
            lstm_hidden_units=192,
            attn_n_head=4,
            attn_qk_output_channel=4,
            emb_dim=48,
            emb_ks=4,
            emb_hs=1,
            activation="prelu",
            emb_size=512,
            eps=1.0e-5,
            causal=False,
            sample_rate=16000,
    ):
        super().__init__(sample_rate=sample_rate)
        
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert self.n_imics == 1, self.n_imics


        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        # self.ref_channel = ref_channel

        self.enc = STFTEncoder(
            n_fft, n_fft, stride, window=window, use_builtin_complex=use_builtin_complex
        )
        self.dec = STFTDecoder(n_fft, n_fft, stride, window=window)


        t_ksize = 3
        if causal:
            ks, padding = (t_ksize, 3), (t_ksize - 1, 1)
            self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            SliceLayer2d(t_ksize),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        else:
            ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
            self.conv = nn.Sequential(
                nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
                nn.GroupNorm(1, emb_dim, eps=eps),
            )

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetV3Block(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=attn_qk_output_channel,
                    activation=activation,
                    eps=eps,
                    causal=causal,
                )
            )

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)



        # visual
        self.av_conv = _clones(nn.Linear(emb_dim+emb_size, emb_dim),n_layers)

    def forward(self, audio_mixture, mouth_embedding):

        n_samples = audio_mixture.shape[1]
        if self.n_imics == 1:
            assert len(audio_mixture.shape) == 2
            audio_mixture = audio_mixture[..., None]  # [B, N, M]  [1, 32000, 1]

        # ilens = torch.ones(input.shape[0], dtype=torch.int8).cuda()
        # for i in range(input.shape[0]):
        #     ilens[i] = int(input.shape[1])
        ilens = [audio_mixture.shape[1] for i in range(audio_mixture.shape[0])]
        ilens = torch.Tensor(ilens).int()

        # mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        # if mix_std_ != 0:
        #     input = input / mix_std_  # RMS normalization

        batch = self.enc(audio_mixture, ilens)[0]  # [B, T, M, F]
        batch0 = batch.transpose(1, 2)  # [B, M, T, F]  [1, 1, 251, 129]
        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape


        batch = self.conv(batch)  # [B, -1, T, F]  [1, 48, 251, 129]

        # visual
        mouth_embedding = F.interpolate(mouth_embedding, (n_frames), mode='linear')  #[1, 512, 251]
        mouth_embedding = torch.repeat_interleave(mouth_embedding.unsqueeze(-1), n_freqs, -1) #[1, 512, 251, 129]

        for ii in range(self.n_layers):
            batch = torch.cat((mouth_embedding,batch), axis=1)   #[1, 560, 251, 129]
            batch = self.av_conv[ii](batch.transpose(1,3)).transpose(1,3)
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]   
        batch = F.interpolate(batch, size=(251, 129), mode="nearest")

        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))

        batch = self.dec(batch.view(-1, n_frames, n_freqs), ilens)[0]  # [B, n_srcs, -1]
        batch = self.pad2(batch.view([n_batch, self.num_spk, -1]), n_samples)

        # if mix_std_ != 0:
        #     batch = batch * mix_std_  # reverse the RMS normalization

        batch = batch.squeeze(1)

        return batch
    
    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args

    @property
    def num_spk(self):
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor

class SliceLayer2d(nn.Module):
    def __init__(self, kernel_size):
        super(SliceLayer2d, self).__init__()
        self.kernel_size = kernel_size
    
    def forward(self, x):
        return x[:, :, :-(self.kernel_size - 1), :]

class GridNetV3Block(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        hidden_channels,
        n_head=4,
        qk_output_channel=4,
        activation="prelu",
        eps=1e-5,
        causal=False,
    ):
        super().__init__()
        assert activation == "prelu"

        in_channels = emb_dim * emb_ks
        self.padding = emb_ks-1
        self.causal = causal

        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs,
            )

        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        if causal:
            self.inter_rnn = nn.LSTM(
                in_channels, hidden_channels, 1, batch_first=True, bidirectional=False
            )
            if emb_ks == emb_hs:
                self.inter_linear = nn.Linear(hidden_channels, in_channels)
            else:
                self.inter_linear = nn.ConvTranspose1d(
                    hidden_channels, emb_dim, emb_ks, stride=emb_hs, padding=self.padding,
                )
        else:
            self.inter_rnn = nn.LSTM(
                in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
            )
            if emb_ks == emb_hs:
                self.inter_linear = nn.Linear(hidden_channels * 2, in_channels)
            else:
                self.inter_linear = nn.ConvTranspose1d(
                    hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs,
                )

        # use constant E not to be dependent on the number of frequency bins
        E = qk_output_channel
        assert emb_dim % n_head == 0

        self.add_module("attn_conv_Q", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_Q",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )

        self.add_module("attn_conv_K", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_K",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )

        self.add_module(
            "attn_conv_V", nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1)
        )
        self.add_module(
            "attn_norm_V",
            AllHeadPReLULayerNormalization4DC((n_head, emb_dim // n_head), eps=eps),
        )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """GridNetV2Block Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape

        olp = self.emb_ks - self.emb_hs
        T = (
            math.ceil((old_T + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )
        Q = (
            math.ceil((old_Q + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )

        x = x.permute(0, 2, 3, 1)  # [B, old_T, old_Q, C]
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, olp, T - old_T - olp))  # [B, T, Q, C]

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, T, Q, C]
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([B * T, -1, self.emb_ks * C])  # [BT, Q//I, I*C]
            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, Q//I, H]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, Q//I, I*C]
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B * T, Q, C])  # [BT, Q, C]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, C, Q]
            intra_rnn = F.unfold(
                intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BT, C*I, -1]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*I]

            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]

            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)  # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]

        intra_rnn = intra_rnn.transpose(1, 2)  # [B, Q, T, C]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, Q, T, C]
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self.emb_ks * C])  # [BQ, T//I, I*C]
            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, T//I, H]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, T//I, I*C]
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B * Q, T, C)  # [BQ, T, C]    [135, 257, 48]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, C, T]
            inter_rnn = F.unfold(
                inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BQ, C*I, -1]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*I]  [135, 254, 192]
            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, -1, H]  [135, 254, 384]

            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]  [135, 384, 254]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]   [135, 48, 257] [135, 48, 251]
            inter_rnn = F.interpolate(inter_rnn, size=257, mode="nearest")

            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)  # [B, Q, T, C]
        inter_rnn = inter_rnn + input_  # [B, Q, T, C]

        inter_rnn = inter_rnn.permute(0, 3, 2, 1)  # [B, C, T, Q]

        inter_rnn = inter_rnn[..., olp : olp + old_T, olp : olp + old_Q]
        batch = inter_rnn

        Q = self["attn_norm_Q"](self["attn_conv_Q"](batch))  # [B, n_head, C, T, Q]
        K = self["attn_norm_K"](self["attn_conv_K"](batch))  # [B, n_head, C, T, Q]
        V = self["attn_norm_V"](self["attn_conv_V"](batch))  # [B, n_head, C, T, Q]
        Q = Q.view(-1, *Q.shape[2:])  # [B*n_head, C, T, Q]
        K = K.view(-1, *K.shape[2:])  # [B*n_head, C, T, Q]
        V = V.view(-1, *V.shape[2:])  # [B*n_head, C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]

        K = K.transpose(2, 3)
        K = K.contiguous().view([B * self.n_head, -1, old_T])  # [B', C*Q, T]

        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)  # [B', T, T]

        if self.causal :
            T = Q.size(1)
            mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool))
            attn_mat = attn_mat.masked_fill(mask == 0, float('-inf'))

        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.contiguous().view(
            [B, self.n_head * emb_dim, old_T, old_Q]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class LayerNormalization(nn.Module):
    def __init__(self, input_dim, dim=1, total_dim=4, eps=1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if ii != self.dim else input_dim for ii in range(total_dim)]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if x.ndim - 1 < self.dim:
            raise ValueError(
                f"Expect x to have {self.dim + 1} dimensions, but got {x.ndim}"
            )
        if x.dtype in HALF_PRECISION_DTYPES:
            dtype = x.dtype
            x = x.float()
        else:
            dtype = None
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat.to(dtype=dtype) if dtype else x_hat


class AllHeadPReLULayerNormalization4DC(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2, input_dimension
        H, E = input_dimension
        param_size = [1, H, E, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, F = x.shape
        x = x.view([B, self.H, self.E, T, F])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2,)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x


class VisualConv1D(nn.Module):
    def __init__(self, V=256, H=512):
        super(VisualConv1D, self).__init__()
        relu_0 = nn.ReLU()
        norm_0 = GlobalLayerNorm(V)
        conv1x1 = nn.Conv1d(V, H, 1, bias=False)
        relu = nn.ReLU()
        norm_1 = GlobalLayerNorm(H)
        dsconv = nn.Conv1d(H, H, 3, stride=1, padding=1,dilation=1, groups=H, bias=False)
        prelu = nn.PReLU()
        norm_2 = GlobalLayerNorm(H)
        pw_conv = nn.Conv1d(H, V, 1, bias=False)
        self.net = nn.Sequential(relu_0, norm_0, conv1x1, relu, norm_1 ,dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
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
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y