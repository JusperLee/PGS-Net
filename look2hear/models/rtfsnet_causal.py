import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sru import SRU
from .base_model import BaseModel
from timm.models.layers import DropPath
from ..layers import normalizations, activations

class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_chan: int = 1,
        out_chan: int = 1,
        kernel_size: int = -1,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = None,
        pre_norm_type: str = None,
        pre_act_type: str = None,
        norm_type: str = None,
        act_type: str = None,
        xavier_init: bool = False,
        bias: bool = True,
        is2d: bool = False,
        causal: bool = False,  # 添加causal参数
    ):
        super(ConvNormAct, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan if kernel_size > 0 else self.in_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.pre_norm_type = pre_norm_type
        self.pre_act_type = pre_act_type
        self.norm_type = norm_type
        self.act_type = act_type
        self.xavier_init = xavier_init
        self.bias = bias
        self.is2d = is2d
        self.causal = causal  # 保存causal参数

        if self.padding is None:
            self.padding = dilation * (kernel_size - 1) // 2
        
        if self.causal:
            self.causal_padding = dilation * (kernel_size - 1)

            if kernel_size > 0:
                conv_class = nn.Conv2d if is2d else nn.Conv1d
                
                self.conv = conv_class(
                    in_channels=self.in_chan,
                    out_channels=self.out_chan,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.causal_padding if not self.is2d else (self.causal_padding, self.padding),
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=self.bias,
                )
                if self.xavier_init:
                    nn.init.xavier_uniform_(self.conv.weight)
            else:
                self.conv = nn.Identity()
        else:
            if kernel_size > 0:
                conv_class = nn.Conv2d if is2d else nn.Conv1d
                
                self.conv = conv_class(
                    in_channels=self.in_chan,
                    out_channels=self.out_chan,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=self.bias,
                )
                if self.xavier_init:
                    nn.init.xavier_uniform_(self.conv.weight)
            else:
                self.conv = nn.Identity()

        self.pre_norm = normalizations.get(self.pre_norm_type)(self.in_chan)
        self.pre_act = activations.get(self.pre_act_type)()
        self.norm = normalizations.get(self.norm_type)(self.out_chan)
        self.act = activations.get(self.act_type)()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): B, C, T, (F)
        """
        x = self.pre_norm(x)
        x = self.pre_act(x)
        x = self.conv(x)
        if self.causal:
            if self.is2d:
                x = x[:, :, :-self.causal_padding, :]
            else:
                x = x[:, :, :-self.causal_padding]
        x = self.norm(x)
        x = self.act(x)
        return x
    
class ConvActNorm(nn.Module):
    def __init__(
        self,
        in_chan: int = 1,
        out_chan: int = 1,
        kernel_size: int = -1,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = None,
        norm_type: str = None,
        act_type: str = None,
        n_freqs: int = -1,
        xavier_init: bool = False,
        bias: bool = True,
        is2d: bool = False,
        causal: bool = False,  # 新增参数causal
    ):
        super(ConvActNorm, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.norm_type = norm_type
        self.act_type = act_type
        self.n_freqs = n_freqs
        self.xavier_init = xavier_init
        self.bias = bias
        self.is2d = is2d
        self.causal = causal  # 保存causal参数

        if self.padding is None:
            self.padding = dilation * (kernel_size - 1) // 2
        
        if self.causal:
            self.causal_padding = dilation * (kernel_size - 1)

            if kernel_size > 0:
                conv_class = nn.Conv2d if is2d else nn.Conv1d
                
                self.conv = conv_class(
                    in_channels=self.in_chan,
                    out_channels=self.out_chan,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.causal_padding if not self.is2d else (self.causal_padding, self.padding),
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=self.bias,
                )
                if self.xavier_init:
                    nn.init.xavier_uniform_(self.conv.weight)
            else:
                self.conv = nn.Identity()
        else:
            if kernel_size > 0:
                conv_class = nn.Conv2d if is2d else nn.Conv1d
                
                self.conv = conv_class(
                    in_channels=self.in_chan,
                    out_channels=self.out_chan,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=self.bias,
                )
                if self.xavier_init:
                    nn.init.xavier_uniform_(self.conv.weight)
            else:
                self.conv = nn.Identity()

        self.act = activations.get(self.act_type)()
        self.norm = normalizations.get(self.norm_type)(
            (self.out_chan, self.n_freqs) if self.norm_type == "LayerNormalization4D" else self.out_chan
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): B, C, T, (F)
        """
        output = self.conv(x)
        if self.causal:
            if self.is2d:
                output = output[:, :, :-self.causal_padding, :]
            else:
                output = output[:, :, :-self.causal_padding]
        output = self.act(output)
        output = self.norm(output)
        return output
    

def causal_adaptive_avg_pool1d_segment(x: torch.Tensor, output_size: int) -> torch.Tensor:
    """
    用“分段”思路实现因果自适应平均池化：
    - 将 T 均匀分割成 S 段
    - 第 i 段只使用过去的信息
    """
    import math
    B, C, T = x.shape
    out = x.new_zeros((B, C, output_size))

    for i in range(output_size):
        start_i = math.floor(i * T / output_size)
        end_i   = math.floor((i + 1) * T / output_size)
        
        # 如果这里要严格“因果”，那就确保 end_i 不超出现实的 “当前可用的过去”。
        # 如果你在一个在线场景，第 i 步真的还没跑到 end_i，那么只能取 min(end_i, 当前时刻)。
        # 这里先假设“离线”场景，end_i 不会越界。
        # 标准 adaptive avg pooling 分段:
        #      bin_i = [start_i, end_i)
        # 这里就直接用 bin_i 做平均即可
        if end_i <= start_i:
            out[:, :, i] = 0
        else:
            out[:, :, i] = x[:, :, start_i:end_i].mean(dim=-1)

    return out

class InjectionMultiSum(nn.Module):
    def __init__(
        self,
        in_chan: int,
        kernel_size: int,
        norm_type: str = "gLN",
        is2d: bool = False,
        causal: bool = False,  # 添加参数causal
    ):
        super(InjectionMultiSum, self).__init__()
        self.in_chan = in_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.is2d = is2d
        self.causal = causal  # 保存causal参数

        self.local_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=self.kernel_size,
            groups=self.in_chan,
            norm_type=self.norm_type,
            bias=False,
            is2d=self.is2d,
            causal=self.causal,  # 传递causal参数
        )
        self.global_embedding = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=self.kernel_size,
            groups=self.in_chan,
            norm_type=self.norm_type,
            bias=False,
            is2d=self.is2d,
            causal=self.causal,  # 传递causal参数
        )
        self.global_gate = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=self.kernel_size,
            groups=self.in_chan,
            norm_type=self.norm_type,
            act_type="Sigmoid",
            bias=False,
            is2d=self.is2d,
            causal=self.causal,  # 传递causal参数
        )

    def forward(self, local_features: torch.Tensor, global_features: torch.Tensor):
        old_shape = global_features.shape[-(len(local_features.shape) // 2) :]
        new_shape = local_features.shape[-(len(local_features.shape) // 2) :]

        local_emb = self.local_embedding(local_features)
        if torch.prod(torch.tensor(new_shape)) > torch.prod(torch.tensor(old_shape)):
            global_emb = F.interpolate(self.global_embedding(global_features), size=new_shape, mode="nearest")
            gate = F.interpolate(self.global_gate(global_features), size=new_shape, mode="nearest")
        else:
            g_interp = F.interpolate(global_features, size=new_shape, mode="nearest")
            global_emb =  F.interpolate(self.global_embedding(g_interp), size=new_shape, mode="nearest")
            gate = F.interpolate(self.global_gate(g_interp), size=new_shape, mode="nearest")

        injection_sum = local_emb * gate + global_emb

        return injection_sum
    
class DualPathRNN(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        dim: int,
        kernel_size: int = 8,
        stride: int = 1,
        rnn_type: str = "LSTM",
        num_layers: int = 1,
        norm_type: str = "LayerNormalization4D",
        act_type: str = "Tanh",
        bidirectional: bool = True,
        causal: bool = False,  # 添加参数causal
    ):
        super(DualPathRNN, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.act_type = act_type
        self.bidirectional = bidirectional
        self.causal = causal  # 保存causal参数

        self.num_direction = int(bidirectional) + 1
        self.unfolded_chan = self.in_chan * self.kernel_size
        self.rnn_out_chan = self.hid_chan * self.num_direction if self.rnn_type != "Attn" else self.unfolded_chan
        
        self.norm = normalizations.get(self.norm_type)((self.in_chan, 1) if self.norm_type == "LayerNormalization4D" else self.in_chan)
        self.unfold = nn.Unfold((self.kernel_size, 1), stride=(self.stride, 1))

        if self.rnn_type == "SRU":
            self.rnn = SRU(
                input_size=self.unfolded_chan,
                hidden_size=self.hid_chan,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
            )
        else:
            self.rnn = getattr(nn, self.rnn_type)(
                input_size=self.unfolded_chan,
                hidden_size=self.hid_chan,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
            )

        self.linear = nn.ConvTranspose1d(self.rnn_out_chan, self.in_chan, self.kernel_size, stride=self.stride)

    def forward(self, x: torch.Tensor):
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        B, C, old_T, old_F = x.shape
        new_T = math.ceil((old_T - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        new_F = math.ceil((old_F - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        x = F.pad(x, (0, new_F - old_F, 0, new_T - old_T))

        residual = x
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous().view(B * new_F, C, new_T, 1)
        if self.causal :
            x = F.pad(x, (0, 0, self.kernel_size - 1, 0))
            x = self.unfold(x)  # B * new_F, C * kernel_size, unfolded_T
            x = x[:,:,:-(self.kernel_size - 1)]
        else:
            x = self.unfold(x)  # B * new_F, C * kernel_size, unfolded_T
        x = x.permute(2, 0, 1)  # unfolded_T, B * new_F, C * kernel_size
        x = self.rnn(x)[0] if self.rnn_type != "Attn" else self.rnn(x)
        x = x.permute(1, 2, 0)
        if self.causal:
            x = F.pad(x, (self.kernel_size - 1, 0))
            x = self.linear(x)
            x = x[:,:,:-(self.kernel_size - 1)]
        else:
            x = self.linear(x)
        x = x.view([B, new_F, C, new_T])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x + residual
        x = x[..., :old_T, :old_F]

        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        return x

class MultiHeadSelfAttention2D(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_freqs: int,
        n_head: int = 4,
        hid_chan: int = 4,
        act_type: str = "PReLU",
        norm_type: str = "LayerNormalization4D",
        dim: int = 3,
        causal: bool = False,  # 添加参数causal
    ):
        super(MultiHeadSelfAttention2D, self).__init__()
        self.in_chan = in_chan
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hid_chan = hid_chan
        self.act_type = act_type
        self.norm_type = norm_type
        self.dim = dim
        self.causal = causal  # 保存causal参数

        assert self.in_chan % self.n_head == 0

        self.Queries = nn.ModuleList()
        self.Keys = nn.ModuleList()
        self.Values = nn.ModuleList()

        for _ in range(self.n_head):
            self.Queries.append(
                ConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                    causal=False
                )
            )
            self.Keys.append(
                ConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                    causal=False
                )
            )
            self.Values.append(
                ConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.in_chan // self.n_head,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                    causal=False
                )
            )

        self.attn_concat_proj = ConvActNorm(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.n_freqs,
            is2d=True,
            causal=False
        )

    def forward(self, x: torch.Tensor):
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        batch_size, _, time, freq = x.size()
        residual = x
        
        all_Q = [q(x) for q in self.Queries]  # [B, E, T, F]
        all_K = [k(x) for k in self.Keys]  # [B, E, T, F]
        all_V = [v(x) for v in self.Values]  # [B, C/n_head, T, F]

        Q = torch.cat(all_Q, dim=0)  # [B', E, T, F]    B' = B*n_head
        K = torch.cat(all_K, dim=0)  # [B', E, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C/n_head, T, F]

        Q = Q.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        K = K.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        V = V.transpose(1, 2)  # [B', T, C/n_head, F]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*F/n_head]
        emb_dim = Q.shape[-1]  # C*F/n_head

        #attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        #attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]

        if self.causal:
            # 在流式情况下使用对角线掩码
            T = Q.size(1)
            mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
        attn_mat = F.softmax(attn_scores, dim=-1)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*F/n_head]
        V = V.reshape(old_shape)  # [B', T, C/n_head, F]
        V = V.transpose(1, 2)  # [B', C/n_head, T, F]
        emb_dim = V.shape[1]  # C/n_head

        x = V.view([self.n_head, batch_size, emb_dim, time, freq])  # [n_head, B, C/n_head, T, F]
        x = x.transpose(0, 1).contiguous()  # [B, n_head, C/n_head, T, F]

        x = x.view([batch_size, self.n_head * emb_dim, time, freq])  # [B, C, T, F]
        x = self.attn_concat_proj(x)  # [B, C, T, F]

        x = x + residual

        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, channels: int, max_len: int = 10000, *args, **kwargs):
        super(PositionalEncoding, self).__init__()
        self.channels = channels
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.channels, requires_grad=False)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.channels, 2).float() * -(torch.log(torch.tensor(self.max_len).float()) / self.channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_head: int = 8,
        dropout: int = 0.1,
        positional_encoding: bool = True,
        batch_first=True,
        causal: bool = False,  # 添加参数causal
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_chan = in_chan
        self.n_head = n_head
        self.dropout = dropout
        self.positional_encoding = positional_encoding
        self.batch_first = batch_first
        self.causal = causal  # 保存causal参数

        assert self.in_chan % self.n_head == 0, "In channels: {} must be divisible by the number of heads: {}".format(
            self.in_chan, self.n_head
        )

        self.norm1 = nn.LayerNorm(self.in_chan)
        self.pos_enc = PositionalEncoding(self.in_chan) if self.positional_encoding else nn.Identity()
        self.attention = nn.MultiheadAttention(self.in_chan, self.n_head, self.dropout, batch_first=self.batch_first)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.in_chan)
        self.drop_path_layer = DropPath(self.dropout)

    def forward(self, x: torch.Tensor):
        res = x
        if self.batch_first:
            x = x.transpose(1, 2)  # B, C, T -> B, T, C

        x = self.norm1(x)
        x = self.pos_enc(x)
        residual = x

        if self.causal:
            # 在流式情况下，使用causal mask
            seq_len = x.size(1)
            attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
        else:
            attn_mask = None

        #x = self.attention(x, x, x)[0]
        x = self.attention(x, x, x, attn_mask=attn_mask)[0]    
        x = self.dropout_layer(x) + residual
        x = self.norm2(x)

        if self.batch_first:
            x = x.transpose(2, 1)  # B, T, C -> B, C, T

        x = self.drop_path_layer(x) + res
        return x 
    
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        norm_type: str = "gLN",
        act_type: str = "ReLU",
        dropout: float = 0,
        is2d: bool = False,
        causal: bool = False,  # 添加参数causal
    ):
        super(FeedForwardNetwork, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.act_type = act_type
        self.dropout = dropout
        self.is2d = is2d
        self.causal = causal  # 保存causal参数

        self.encoder = ConvNormAct(self.in_chan, self.hid_chan, 1, norm_type=self.norm_type, bias=False, is2d=self.is2d, causal=False)  # FC 1
        self.refiner = ConvNormAct(
            self.hid_chan,
            self.hid_chan,
            self.kernel_size,
            groups=self.hid_chan,
            act_type=self.act_type,
            is2d=self.is2d,
            causal=self.causal,  # 传递causal参数
        )  # DW seperable conv
        self.decoder = ConvNormAct(self.hid_chan, self.in_chan, 1, norm_type=self.norm_type, bias=False, is2d=self.is2d, causal=False)  # FC 2
        self.dropout_layer = DropPath(self.dropout)

    def forward(self, x: torch.Tensor):
        res = x
        x = self.encoder(x)
        x = self.refiner(x)
        x = self.dropout_layer(x)
        x = self.decoder(x)
        x = self.dropout_layer(x) + res
        return x

class GlobalAttention(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int = None,
        kernel_size: int = 5,
        n_head: int = 8,
        dropout: float = 0.1,
        pos_enc: bool = True,
        norm_type: str = "gLN",
        causal: bool = False,  # 添加参数causal
    ):
        super(GlobalAttention, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan if hid_chan is not None else 2 * self.in_chan
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout
        self.pos_enc = pos_enc
        self.causal = causal  # 保存causal参数

        self.MHSA = MultiHeadSelfAttention(self.in_chan, self.n_head, self.dropout, self.pos_enc, causal=self.causal,)
        self.FFN = FeedForwardNetwork(self.in_chan, self.hid_chan, self.kernel_size, dropout=self.dropout, causal=self.causal, norm_type=norm_type)

    def forward(self, x: torch.Tensor):
        x = self.MHSA(x)
        x = self.FFN(x)
        return x

class AudioBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        layers: dict = dict(),
        is2d: bool = False,
        causal: bool = False,
    ):
        super(AudioBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.layers = layers
        self.is2d = is2d
        self.causal = causal  # 保存causal参数

        self.pool = F.adaptive_avg_pool2d if self.is2d else F.adaptive_avg_pool1d

        self.gateway = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            groups=self.in_chan,
            act_type=self.act_type,
            is2d=self.is2d,
            causal=False,  # 传递causal参数
        )
        self.projection = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=1,
            norm_type=self.norm_type,
            act_type=self.act_type,
            is2d=self.is2d,
            causal=False,  # 传递causal参数
        )
        # Downsample layers
        self.downsample_layers = nn.ModuleList([])
        for i in range(self.upsampling_depth):
            self.downsample_layers.append(
                ConvNormAct(
                    in_chan=self.hid_chan,
                    out_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    stride=1 if i == 0 else self.stride,
                    groups=self.hid_chan,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                    causal=self.causal,  # 传递causal参数
                )
            )
        # Global process
        self.process_f = DualPathRNN(
            in_chan=self.hid_chan,
            **self.layers["layer_1"]
        )
        self.process_t = DualPathRNN(
            in_chan=self.hid_chan,
            causal = self.causal,
            **self.layers["layer_2"]
        )
        self.process_att = MultiHeadSelfAttention2D(
            in_chan=self.hid_chan,
            **self.layers["layer_3"]
        )
        
        # top down fusion 
        self.fusion_layers = nn.ModuleList([])
        for _ in range(self.upsampling_depth):
            self.fusion_layers.append(
                InjectionMultiSum(
                    in_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                    causal=self.causal,  # 传递causal参数
                )
            )
        
        # Upsampling layers
        self.concat_layers = nn.ModuleList([])
        for _ in range(self.upsampling_depth):
            self.concat_layers.append(
                InjectionMultiSum(
                    in_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                    causal=self.causal,  # 传递causal参数
                )
            )        
        
        
        self.residual_conv = ConvNormAct(
            in_chan=self.hid_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            is2d=self.is2d,
            causal=False,  # 传递causal参数
        )

    def forward(self, x: torch.Tensor):
        # x: B, C, T, (F)
        residual = self.gateway(x)
        x_enc = self.projection(residual)

        # bottom-up
        downsampled_outputs = [self.downsample_layers[0](x_enc)]
        for i in range(1, self.upsampling_depth):
            downsampled_outputs.append(self.downsample_layers[i](downsampled_outputs[-1]))

        # global pooling
        shape = downsampled_outputs[-1].shape
        global_features = sum(self.pool(features, output_size=shape[-(len(shape) // 2) :]) for features in downsampled_outputs)
        
        # global attention module
        global_features = self.process_f(global_features)
        global_features = self.process_t(global_features)
        global_features = self.process_att(global_features)

        # Gather them now in reverse order
        x_fused = [self.fusion_layers[i](downsampled_outputs[i], global_features) for i in range(self.upsampling_depth)]

        # fuse them into a single vector
        expanded = self.concat_layers[-1](x_fused[-2], x_fused[-1]) + downsampled_outputs[-2]
        for i in range(self.upsampling_depth - 3, -1, -1):
            expanded = self.concat_layers[i](x_fused[i], expanded) + downsampled_outputs[i]

        expanded = F.interpolate(expanded, size=residual.shape[-(len(residual.shape) // 2) :], mode="nearest")
        out = self.residual_conv(expanded) + residual

        return out
    
class VideoBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        layers: dict = dict(),
        is2d: bool = False,
        causal: bool = False,  # 添加参数causal
    ):
        super(VideoBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.layers = layers
        self.is2d = is2d
        self.causal = causal  # 保存causal参数

        self.pool = causal_adaptive_avg_pool1d_segment

        self.gateway = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            groups=self.in_chan,
            act_type=self.act_type,
            is2d=self.is2d,
            causal=False,  # 传递causal参数
        )
        self.projection = ConvNormAct(
            in_chan=self.in_chan,
            out_chan=self.hid_chan,
            kernel_size=1,
            norm_type=self.norm_type,
            act_type=self.act_type,
            is2d=self.is2d,
            causal=False,  # 传递causal参数
        )
        # Downsample layers
        self.downsample_layers = nn.ModuleList([])
        for i in range(self.upsampling_depth):
            self.downsample_layers.append(
                ConvNormAct(
                    in_chan=self.hid_chan,
                    out_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    stride=1 if i == 0 else self.stride,
                    groups=self.hid_chan,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                    causal=self.causal,  # 传递causal参数
                )
            )
        # Global process
        self.process_att = GlobalAttention(
            in_chan=self.hid_chan,
            **self.layers["layer_1"]
        )
        
        # top down fusion 
        self.fusion_layers = nn.ModuleList([])
        for _ in range(self.upsampling_depth):
            self.fusion_layers.append(
                InjectionMultiSum(
                    in_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                    causal=self.causal,  # 传递causal参数
                )
            )
        
        # Upsampling layers
        self.concat_layers = nn.ModuleList([])
        for _ in range(self.upsampling_depth):
            self.concat_layers.append(
                InjectionMultiSum(
                    in_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    norm_type=self.norm_type,
                    is2d=self.is2d,
                    causal=self.causal,  # 传递causal参数
                )
            )        
        
        
        self.residual_conv = ConvNormAct(
            in_chan=self.hid_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            is2d=self.is2d,
            causal=False,  # 传递causal参数
        )

    def forward(self, x: torch.Tensor):
        # x: B, C, T, (F)
        residual = self.gateway(x)
        x_enc = self.projection(residual)

        # bottom-up
        downsampled_outputs = [self.downsample_layers[0](x_enc)]
        for i in range(1, self.upsampling_depth):
            downsampled_outputs.append(self.downsample_layers[i](downsampled_outputs[-1]))
        
        # global pooling
        shape = downsampled_outputs[-1].shape
        global_features = sum(self.pool(features, output_size=int(shape[-(len(shape) // 2) :][0])) for features in downsampled_outputs)
        # global attention module
        global_features = self.process_att(global_features)

        # Gather them now in reverse order
        x_fused = [self.fusion_layers[i](downsampled_outputs[i], global_features) for i in range(self.upsampling_depth)]

        # fuse them into a single vector
        expanded = self.concat_layers[-1](x_fused[-2], x_fused[-1]) + downsampled_outputs[-2]
        for i in range(self.upsampling_depth - 3, -1, -1):
            expanded = self.concat_layers[i](x_fused[i], expanded) + downsampled_outputs[i]

        out = self.residual_conv(expanded) + residual

        return out
    
class FusionBasemodule(nn.Module):
    def __init__(self, ain_chan: int, vin_chan: int, kernel_size: int, video_fusion: bool, is2d: bool, causal: bool = False):
        super(FusionBasemodule, self).__init__()
        self.ain_chan = ain_chan
        self.vin_chan = vin_chan
        self.kernel_size = kernel_size
        self.video_fusion = video_fusion
        self.is2d = is2d
        self.causal = causal  # 保存causal参数

    def forward(self, audio, video):
        raise NotImplementedError

    def wrangle_dims(self, audio: torch.Tensor, video: torch.Tensor):
        T1 = audio.shape[-(len(audio.shape) // 2) :]
        T2 = video.shape[-(len(video.shape) // 2) :]

        self.x = len(T1) > len(T2)
        self.y = len(T2) > len(T1)

        video = video.unsqueeze(-1) if self.x else video
        audio = audio.unsqueeze(-1) if self.y else audio

        return audio, video

    def unwrangle_dims(self, audio: torch.Tensor, video: torch.Tensor):
        video = video.squeeze(-1) if self.x else video
        audio = audio.squeeze(-1) if self.y else audio

        return audio, video

class ATTNFusionCell(nn.Module):
    def __init__(
        self,
        in_chan_a: int,
        in_chan_b: int,
        kernel_size: int = 1,
        is2d: bool = False
    ):
        super(ATTNFusionCell, self).__init__()
        self.in_chan_a = in_chan_a
        self.in_chan_b = in_chan_b
        self.kernel_size = kernel_size
        self.is2d = is2d

        self.key_embed = ConvNormAct(
            self.in_chan_a,
            self.in_chan_a,
            1,
            groups=self.in_chan_a,
            norm_type="BatchNorm2d",
            act_type="ReLU",
            bias=False,
            is2d=self.is2d
        )
        self.value_embed = ConvNormAct(
            self.in_chan_a,
            self.in_chan_a,
            1,
            groups=self.in_chan_a,
            norm_type="BatchNorm2d",
            bias=False,
            is2d=self.is2d
        )
        self.attention_embed = ConvNormAct(
            self.in_chan_b,
            self.kernel_size * self.in_chan_a,
            1,
            groups=self.in_chan_a,
            norm_type="cLN"
        )

        self.resize = ConvNormAct(
            self.in_chan_b,
            self.in_chan_a,
            1,
            groups=self.in_chan_a,
            norm_type="cLN"
        )
    def forward(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor):
        batch_size, _, time_steps, _ = tensor_a.shape

        b_transformed = F.interpolate(self.resize(tensor_b), size=time_steps, mode="nearest")
        if self.is2d:
            b_transformed = b_transformed.unsqueeze(-1)

        k1 = self.key_embed(tensor_a) * b_transformed  # bs,c,h,w
        v = self.value_embed(tensor_a)  # bs,c,h,w

        att = self.attention_embed(tensor_b)  # bs,c*k*k,h,w
        att = att.reshape(batch_size, self.in_chan_a, self.kernel_size, -1)

        att = att.mean(2, keepdim=False).view(batch_size, self.in_chan_a, -1)  # bs,c,h*w
        att = F.interpolate(torch.softmax(att, -1), size=time_steps, mode="nearest")

        if self.is2d:
            att = att.unsqueeze(-1)
        k2 = att * v
        # Fusion
        fused_tensor = k1 + k2

        return fused_tensor

class ATTNFusion(FusionBasemodule):
    def __init__(
        self,
        ain_chan: int,
        vin_chan: int,
        kernel_size: int,
        video_fusion: bool = True,
        is2d=True,
    ):
        super(ATTNFusion, self).__init__(ain_chan, vin_chan, kernel_size, video_fusion, is2d)
        if video_fusion:
            self.video_lstm = ATTNFusionCell(self.vin_chan, self.ain_chan, self.kernel_size, self.is2d)
        self.audio_lstm = ATTNFusionCell(self.ain_chan, self.vin_chan, self.kernel_size, self.is2d)

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        if self.video_fusion:
            video_fused = self.video_lstm(video, audio)
        else:
            video_fused = video

        audio_fused = self.audio_lstm(audio, video)

        return audio_fused, video_fused


class MultiModalFusion(nn.Module):
    def __init__(
        self,
        audio_bn_chan: int,
        video_bn_chan: int,
        kernel_size: int = 1,
        fusion_repeats: int = 3,
        fusion_type: str = "ConcatFusion",
        fusion_shared: bool = False,
        is2d: bool = False
    ):
        super(MultiModalFusion, self).__init__()
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        self.kernel_size = kernel_size
        self.fusion_repeats = fusion_repeats
        self.fusion_type = fusion_type
        self.fusion_shared = fusion_shared
        self.is2d = is2d
        
        fusion_class = ATTNFusion if self.fusion_repeats > 0 else nn.Identity
        if self.fusion_shared:
            self.fusion_module = fusion_class(
                ain_chan=self.audio_bn_chan,
                vin_chan=self.video_bn_chan,
                kernel_size=self.kernel_size,
                video_fusion=self.fusion_repeats > 1,
                is2d=self.is2d
            )
        else:
            self.fusion_module = nn.ModuleList()
            for i in range(self.fusion_repeats):
                self.fusion_module.append(
                    fusion_class(
                        ain_chan=self.audio_bn_chan,
                        vin_chan=self.video_bn_chan,
                        kernel_size=self.kernel_size,
                        video_fusion=i != self.fusion_repeats - 1,
                        is2d=self.is2d
                    )
                )

    def get_fusion_block(self, i: int):
        if self.fusion_shared:
            return self.fusion_module
        else:
            return self.fusion_module[i]

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio_residual = audio
        video_residual = video

        for i in range(self.fusion_repeats):
            if i == 0:
                audio_fused, video_fused = self.get_fusion_block(i)(audio, video)
            else:
                audio_fused, video_fused = self.get_fusion_block(i)(audio_fused + audio_residual, video_fused + video_residual)

        return audio_fused
    
class Separator(nn.Module):
    def __init__(
        self,
        audio_params: dict,
        video_params: dict,
        audio_bn_chan: int,
        video_bn_chan: int,
        fusion_params: dict,
    ):
        super(Separator, self).__init__()
        self.audio_params = audio_params
        self.video_params = video_params
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        self.fusion_params = fusion_params
        
        self.fusion_repeats = self.video_params["repeats"]
        self.audio_repeats = self.audio_params["repeats"] - self.fusion_repeats
        self.video_repeat = self.video_params["repeats"]
        
        self.audio_shared = self.audio_params["shared"]
        self.video_shared = self.video_params["shared"]
        
        self.audio_params.pop("repeats", None)
        self.audio_params.pop("shared", None)
        self.video_params.pop("repeats", None)
        self.video_params.pop("shared", None)
        
        # Video net and audio net
        if self.audio_shared:
            self.audio_net = AudioBlock(
                **self.audio_params,
                in_chan=self.audio_bn_chan
            )
        else:
            self.audio_net = nn.ModuleList([AudioBlock(
                **self.audio_params,
                in_chan=self.audio_bn_chan
            ) for _ in range(self.fusion_repeats + self.audio_repeats)])
            
        if self.video_shared:
            self.video_net = VideoBlock(
                **self.video_params,
                in_chan=self.video_bn_chan
            )
        else:
            self.video_net = nn.ModuleList([VideoBlock(
                **self.video_params,
                in_chan=self.video_bn_chan
            ) for _ in range(self.video_repeat)])
            
        # CAF
        # print(self.audio_bn_chan)
        self.caf = MultiModalFusion(
            **self.fusion_params, 
            audio_bn_chan=self.audio_bn_chan, 
            video_bn_chan=self.video_bn_chan,
            fusion_repeats=self.fusion_repeats)
        
    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        """
        Args:
            audio (torch.Tensor): B, C, T, F
            video (torch.Tensor): B, C, T

        """
        audio_residual = audio
        video_residual = video
        # CAF
        for i in range(self.fusion_repeats):
            if self.audio_shared:
                audio = self.audio_net(audio + audio_residual if i > 0 else audio)
            else:
                audio = self.audio_net[i](audio + audio_residual if i > 0 else audio)
            if self.video_shared:
                video = self.video_net(video + video_residual if i > 0 else video)
            else:
                video = self.video_net[i](video + video_residual if i > 0 else video)
                
            audio, video = self.caf.get_fusion_block(i)(audio, video)
        
        # RTFSNet
        for j in range(self.audio_repeats):
            i = j + self.fusion_repeats
            if self.audio_shared:
                audio = self.audio_net(audio + audio_residual if i > 0 else audio)
            else:
                audio = self.audio_net[i](audio + audio_residual if i > 0 else audio)
                
        return audio

class STFTEncoder(nn.Module):
    def __init__(
        self,
        win: int,
        hop_length: int,
        out_chan: int = 2,
        kernel_size: int = -1,
        stride: int = 1,
        act_type: str = "ReLU",
        norm_type: str = "gLN",
        bias: bool = False,
        causal: bool = False,  # 添加causal参数
    ):
        super(STFTEncoder, self).__init__()

        self.win = win
        self.hop_length = hop_length
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.act_type = act_type
        self.norm_type = norm_type
        self.bias = bias
        self.causal = causal  # 保存causal参数

        self.conv = ConvNormAct(
            in_chan=2,
            out_chan=self.out_chan,
            kernel_size=self.kernel_size,
            stride=self.stride,
            act_type=self.act_type,
            norm_type=self.norm_type,
            xavier_init=True,
            bias=self.bias,
            is2d=True,
            causal=self.causal,  # 传递causal参数
        )

        self.register_buffer("window", torch.hann_window(self.win), False)

    def unsqueeze_to_2D(self, x: torch.Tensor):
        if x.ndim == 1:
            return x.reshape(1, -1)
        elif len(s := x.shape) == 3:
            assert s[1] == 1
            return x.reshape(s[0], -1)
        else:
            return x
        
    def get_out_chan(self):
        return self.out_chan
    
    def forward(self, x: torch.Tensor):
        x = self.unsqueeze_to_2D(x)

        spec = torch.stft(
            x,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            return_complex=True,
        )

        spec = torch.stack([spec.real, spec.imag], 1).transpose(2, 3).contiguous()  # B, 2, T, F
        spec_feature_map = self.conv(spec)  # B, C, T, F

        return spec_feature_map
    
class STFTDecoder(nn.Module):
    def __init__(
        self,
        win: int,
        hop_length: int,
        in_chan: int,
        n_src: int,
        kernel_size: int = -1,
        stride: int = 1,
        bias: bool = False,
        causal: bool = False,  # 添加causal参数
        *args,
        **kwargs,
    ):
        super(STFTDecoder, self).__init__()
        self.win = win
        self.hop_length = hop_length
        self.in_chan = in_chan
        self.n_src = n_src
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.stride = stride
        self.bias = bias
        self.causal = causal  # 保存causal参数

        if self.kernel_size > 0:
            self.decoder = nn.ConvTranspose2d(
                in_channels=self.in_chan,
                out_channels=2,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            )
            torch.nn.init.xavier_uniform_(self.decoder.weight)
        else:
            self.decoder = nn.Identity()

        self.register_buffer("window", torch.hann_window(self.win), False)

    def forward(self, x: torch.Tensor, input_shape: torch.Size):
        # B, n_src, N, T, F

        batch_size, length = input_shape[0], input_shape[-1]

        x = x.view(batch_size * self.n_src, self.in_chan, *x.shape[-2:])  # B, n_src, N, T, F -> # B * n_src, N, T, F

        if self.causal:
            # 流式情况下手动padding
            pad = (0, 0, self.kernel_size - 1, 0)
            x = F.pad(x, pad)

        decoded_separated_audio = self.decoder(x)  # B * n_src, N, T, F - > B * n_src, 2, T, F

        if self.causal:
            # 只保留有用的信息
            decoded_separated_audio = decoded_separated_audio[:, :, :-self.padding, :]

        spec = torch.complex(decoded_separated_audio[:, 0], decoded_separated_audio[:, 1])  # B*n_src, T, F
        # spec = torch.stack([spec.real, spec.imag], dim=-1)  # B*n_src, T, F
        spec = spec.transpose(1, 2).contiguous()  # B*n_src, F, T
        output = torch.istft(
            spec,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            length=length,
        )  # B*n_src, L

        output = output.view(batch_size, self.n_src, length)  # B, n_src, L

        return output
    
class MaskGenerator(nn.Module):
    def __init__(
        self,
        n_src: int,
        audio_emb_dim: int,
        bottleneck_chan: int,
        kernel_size: int = 1,
        mask_act: str = "ReLU",
        RI_split: bool = False,
        output_gate: bool = False,
        dw_gate: bool = False,
        direct: bool = False,
        is2d: bool = False,
        causal: bool = False,  # 添加causal参数
        *args,
        **kwargs,
    ):
        super(MaskGenerator, self).__init__()
        self.n_src = n_src
        self.in_chan = audio_emb_dim
        self.bottleneck_chan = bottleneck_chan
        self.kernel_size = kernel_size
        self.mask_act = mask_act
        self.output_gate = output_gate
        self.dw_gate = dw_gate
        self.RI_split = RI_split
        self.direct = direct
        self.is2d = is2d
        self.causal = causal  # 保存causal参数

        if not self.direct:
            mask_output_chan = self.n_src * self.in_chan

            self.mask_generator = nn.Sequential(
                nn.PReLU(),
                ConvNormAct(
                    self.bottleneck_chan,
                    mask_output_chan,
                    self.kernel_size,
                    act_type=self.mask_act,
                    is2d=self.is2d,
                    causal=self.causal,  # 传递causal参数
                ),
            )

            if self.output_gate:
                groups = mask_output_chan if self.dw_gate else 1
                self.output = ConvNormAct(mask_output_chan, mask_output_chan, 1, act_type="Tanh", is2d=self.is2d, groups=groups, causal=self.causal,)
                self.gate = ConvNormAct(mask_output_chan, mask_output_chan, 1, act_type="Sigmoid", is2d=self.is2d, groups=groups, causal=self.causal,)

    def __apply_masks(self, masks: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        batch_size = audio_mixture_embedding.size(0)
        dims = audio_mixture_embedding.shape[-(len(audio_mixture_embedding.shape) // 2) :]
        if self.RI_split:
            masks = masks.view(batch_size, self.n_src, 2, self.in_chan // 2, *dims)
            audio_mixture_embedding = audio_mixture_embedding.view(batch_size, 2, self.in_chan // 2, *dims)

            mask_real = masks[:, :, 0]  # B, n_src, C/2, T, (F)
            mask_imag = masks[:, :, 1]  # B, n_src, C/2, T, (F)
            emb_real = audio_mixture_embedding[:, 0].unsqueeze(1)  # B, 1, C/2, T, (F)
            emb_imag = audio_mixture_embedding[:, 1].unsqueeze(1)  # B, 1, C/2, T, (F)

            est_spec_real = emb_real * mask_real - emb_imag * mask_imag  # B, n_src, C/2, T, (F)
            est_spec_imag = emb_real * mask_imag + emb_imag * mask_real  # B, n_src, C/2, T, (F)

            separated_audio_embedding = torch.cat([est_spec_real, est_spec_imag], 2)  # B, n_src, C, T, (F)
        else:
            masks = masks.view(batch_size, self.n_src, self.in_chan, *dims)
            separated_audio_embedding = masks * audio_mixture_embedding.unsqueeze(1)

        return separated_audio_embedding

    def forward(self, refined_features: torch.Tensor, audio_mixture_embedding: torch.Tensor):
        if self.direct:
            return refined_features
        else:
            masks = self.mask_generator(refined_features)
            if self.output_gate:
                masks = self.output(masks) * self.gate(masks)
                
            separated_audio_embedding = self.__apply_masks(masks, audio_mixture_embedding)

            return separated_audio_embedding

class RTFSNetCausal(BaseModel):
    def __init__(
        self,
        n_src: int,
        enc_dec_params: dict,
        audio_bn_params: dict,
        audio_params: dict,
        mask_generation_params: dict,
        pretrained_vout_chan: int = -1,
        video_bn_params: dict = dict(),
        video_params: dict = dict(),
        fusion_params: dict = dict(),
        sample_rate=16000,
        causal: bool = False,  # 添加causal参数
    ):
        super(RTFSNetCausal, self).__init__(sample_rate=sample_rate)
        self.n_src = n_src
        self.pretrained_vout_chan = pretrained_vout_chan
        self.audio_bn_params = audio_bn_params
        self.video_bn_params = video_bn_params
        self.enc_dec_params = enc_dec_params
        self.audio_params = audio_params
        self.video_params = video_params
        self.fusion_params = fusion_params
        self.mask_generation_params = mask_generation_params
        self.causal = causal  # 保存causal参数
        
        self.enc_dec_params["causal"] = self.causal  # 传递causal参数
        self.encoder = STFTEncoder(**self.enc_dec_params)
        encoder_out_chan = self.encoder.get_out_chan()
        
        self.audio_bottleneck = ConvNormAct(**self.audio_bn_params, in_chan=encoder_out_chan)
        self.video_bottleneck = ConvNormAct(**self.video_bn_params, in_chan=self.pretrained_vout_chan)

        self.audio_params['causal'] = self.causal  # 传递causal参数
        self.video_params['causal'] = self.causal  # 传递causal参数
        
        self.separator = Separator(
            audio_params=self.audio_params,
            video_params=self.video_params,
            audio_bn_chan=self.audio_bn_params["out_chan"],
            video_bn_chan=self.pretrained_vout_chan,
            fusion_params=self.fusion_params,
        )
        
        self.mask_generator = MaskGenerator(
            **self.mask_generation_params,
            n_src=self.n_src,
            audio_emb_dim=encoder_out_chan,
            bottleneck_chan=self.audio_bn_params["out_chan"],
        )
        
        self.decoder = STFTDecoder(
            **self.enc_dec_params,
            in_chan=encoder_out_chan * self.n_src,
            n_src=self.n_src,
        )
        
    def forward(self, audio_mixture: torch.Tensor, mouth_embedding: torch.Tensor = None):
        audio_mixture_embedding = self.encoder(audio_mixture)  # B, 1, L -> B, N, T, (F)

        audio = self.audio_bottleneck(audio_mixture_embedding)  # B, C, T, (F)
        video = self.video_bottleneck(mouth_embedding)  # B, N2, T2, (F2) -> B, C2, T2, (F2)

        refined_features = self.separator(audio, video)  # B, C, T, (F)

        separated_audio_embeddings = self.mask_generator(refined_features, audio_mixture_embedding)  # B, n_src, N, T, (F)
        separated_audios = self.decoder(separated_audio_embeddings, audio_mixture.shape)  #  B, n_src, L

        return separated_audios
    
    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args
