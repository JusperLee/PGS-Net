import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import normalizations, activations
from .base_model import BaseModel
from rich import print
import math

def cal_padding(input_size, kernel_size=1, stride=1, dilation=1):
    return (kernel_size - input_size + (kernel_size-1)*(dilation-1) \
        + stride*(input_size-1)) // 2
    
class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, in_chan, out_chan, kernel_size=1, stride=1, norm_type="gLN", act_type="PReLU"):
        super().__init__()
        #padding = int((kernel_size - 1) / 2)
        padding = kernel_size - 1
        if in_chan == out_chan:
            groups = in_chan
        else:
            groups = 1
        self.conv = nn.Conv1d(
            in_chan, out_chan, kernel_size, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = normalizations.get(norm_type)(out_chan)
        self.act = activations.get(act_type)()
        self.padding = padding

    def forward(self, input):
        output = self.conv(input)
        if self.padding > 0:
            output = output[:, :, :-self.padding]
        output = self.norm(output)
        return self.act(output)

class ConvNorm(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride=1,
        groups=1,
        dilation=1,
        padding=0,
        norm_type="gLN",
        bias=True,
    ):
        super(ConvNorm, self).__init__()
        self.padding = padding
        self.conv = nn.Conv1d(
            in_chan, out_chan, kernel_size, stride, padding, dilation, bias=bias, groups=groups
        )
        self.norm = normalizations.get(norm_type)(out_chan)

    def forward(self, x):
        output = self.conv(x)
        if self.padding > 0:
            output = output[:, :, :-self.padding]
        return self.norm(output)

class VisualSubnetwork(nn.Module):
    """[summary]

                   [spp_dw_3] --------\        ...-->\ 
                        /              \             \ 
                   [spp_dw_2] --------> [c] -PC.N.A->\ 
                        /              /             \ 
                   [spp_dw_1] -DC.N.--/        ...-->\ 
                        /                            \ 
    x -> [proj] -> [spp_dw_0]                  ...--> [c] -PC.N.A.PC--[(dropout)]--> 
     \                                                                           / 
      \-------------------------------------------------------------------------/

    Args:
        in_chan (int, optional): [description]. Defaults to 128.
        out_chan (int, optional): [description]. Defaults to 512.
        depth (int, optional): [description]. Defaults to 4.
        norm_type (str, optional): [description]. Defaults to "BatchNorm1d".
        act_type (str, optional): [description]. Defaults to "PReLU".
    """
    def __init__(
            self, 
            in_chan=128, 
            out_chan=512, 
            depth=4, 
            dropout=-1,
            norm_type="BatchNorm1d", 
            act_type="PReLU",
            dilation=1,
            kernel_size=5
        ):
        super(VisualSubnetwork, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.depth = depth
        self.norm_type = norm_type
        self.act_type = act_type
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.proj = ConvNormAct(in_chan, out_chan, kernel_size=1, norm_type=norm_type, act_type=act_type)
        self.spp_dw = self._build_downsample_layers()
        self.fuse_layers = self._build_fusion_layers()
        self.concat_layer = self._build_concat_layers()
        self.last = nn.Sequential(
            ConvNormAct(out_chan*depth, out_chan, kernel_size=1, norm_type=norm_type, act_type=act_type),
            nn.Conv1d(out_chan, in_chan, 1)
        )
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None


    def _build_downsample_layers(self):
        out = nn.ModuleList()
        out.append(
            ConvNormAct(self.out_chan, self.out_chan, kernel_size=self.kernel_size, 
                norm_type=self.norm_type, act_type=None
            )
        )
        # ----------Down Sample Layer----------
        for _ in range(1, self.depth):
            out.append(
                ConvNormAct(self.out_chan, self.out_chan, kernel_size=self.kernel_size, stride=2,
                    norm_type=self.norm_type, act_type=None
                )
            )
        return out


    def _build_fusion_layers(self):
        out = nn.ModuleList()
        for i in range(self.depth):
            fuse_layer = nn.ModuleList()
            for j in range(self.depth):
                if i == j or (j - i == 1):
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(
                        ConvNormAct(self.out_chan, self.out_chan, kernel_size=self.kernel_size, stride=2,
                            norm_type=self.norm_type, act_type=None
                        )
                    )
            out.append(fuse_layer)
        return out


    def _build_concat_layers(self):
        out = nn.ModuleList()
        for i in range(self.depth):
            if i == 0 or i == self.depth - 1:
                out.append(
                    ConvNormAct(self.out_chan*2, self.out_chan, kernel_size=1, 
                        norm_type=self.norm_type, act_type=self.act_type
                    )
                )
            else:
                out.append(
                    ConvNormAct(self.out_chan*3, self.out_chan, kernel_size=1,
                        norm_type=self.norm_type, act_type=self.act_type
                    )
                )
        return out


    def forward(self, x):
        # x: shape (B, C, T)
        res = x
        x = self.proj(x)
        
        # bottom-up
        output = [self.spp_dw[0](x)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # lateral connection
        x_fuse = []
        for i in range(self.depth):
            T = output[i].shape[-1]
            y = torch.cat(
                (
                    self.fuse_layers[i][0](output[i - 1]) if i - 1 >= 0 \
                        else torch.Tensor().to(x.device),
                    output[i],
                    F.interpolate(output[i + 1], size=T, mode="nearest") if i + 1 < self.depth
                        else torch.Tensor().to(x.device),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layer[i](y))

        # resize to T
        T = output[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(x_fuse[i], size=T, mode="nearest")

        # concat and shortcut
        x = self.last(torch.cat(x_fuse, dim=1))
        # dropout
        if self.dropout_layer:
            x = self.dropout_layer(x)

        return res + x

class VideoBlock(nn.Module):
    """[summary]

    x --> videoBlock --> Conv1d|PReLU -> videoBlock -> Conv1d|PReLU -> videoBlock -> ...
     \               /                             /                             /
      \-------------/-----------------------------/-----------------------------/

    Args:
        in_chan (int, optional): [description]. Defaults to 128.
        out_chan (int, optional): [description]. Defaults to 512.
        iter (int, optional): [description]. Defaults to 4.
        shared (bool, optional): [description]. Defaults to False.
    """

    def __init__(self,
                 in_chan=128, 
                 out_chan=512, 
                 depth=4, 
                 iter=4, 
                 dilations=None,
                 shared=False, 
                 dropout=-1,
                 norm_type="BatchNorm1d",
                 act_type="PReLU",
                 kernel_size=5):
        super(VideoBlock, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.depth = depth
        self.iter = iter
        self.dilations = [1,]*iter if dilations is None else dilations
        self.shared = shared
        self.dropout = dropout
        self.norm_type = norm_type
        self.act_type = act_type
        self.kernel_size = kernel_size

        self.video = self._build_video()
        self.concat_block = self._build_concat_block()


    def get_video_block(self, i):
        if self.shared:
            return self.video
        else:
            return self.video[i]

    def get_concat_block(self, i):
        if self.shared:
            return self.concat_block
        else:
            return self.concat_block[i]

    def _build_video(self):
        if self.shared:
            return self._build_videoblock(self.in_chan, self.out_chan, self.depth, self.dropout,
                        self.norm_type, self.act_type, dilation=self.dilations,
                        kernel_size=self.kernel_size)
        else:
            out = nn.ModuleList()
            for i in range(self.iter):
                out.append(
                    self._build_videoblock(self.in_chan, self.out_chan, self.depth, self.dropout,
                        self.norm_type, self.act_type, dilation=self.dilations[i],
                        kernel_size=self.kernel_size))
            return out


    def _build_videoblock(self, *args, dilation=1, **kwargs):
        return VisualSubnetwork(*args, **kwargs)


    def _build_concat_block(self):
        if self.shared:
            return nn.Sequential(
                nn.Conv1d(self.in_chan, self.in_chan,
                          1, 1, groups=self.in_chan),
                nn.PReLU())
        else:
            out = nn.ModuleList([None])
            for _ in range(self.iter-1):
                out.append(nn.Sequential(
                    nn.Conv1d(self.in_chan, self.in_chan,
                              1, 1, groups=self.in_chan),
                    nn.PReLU()))
            return out

    def forward(self, x):
        # x: shape (B, C, T)
        res = x
        for i in range(self.iter):
            video = self.get_video_block(i)
            concat_block = self.get_concat_block(i)
            if i == 0:
                x = video(x)
            else:
                x = video(concat_block(res + x))
        return x

class AudioSubnetwork(nn.Module):
    def __init__(
        self, in_chan=128, out_chan=512, upsampling_depth=4, norm_type="gLN", act_type="prelu"
    ):
        super().__init__()
        self.proj_1x1 = ConvNormAct(
            in_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
        )
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([])
        self.spp_dw.append(
            ConvNorm(
                out_chan,
                out_chan,
                kernel_size=5,
                stride=1,
                groups=out_chan,
                dilation=1,
                padding=4,  # padding=((5 - 1) // 2) * 1, causal padding: kernel_size - 1
                norm_type=norm_type,
            )
        )
        # ----------Down Sample Layer----------
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                ConvNorm(
                    out_chan,
                    out_chan,
                    kernel_size=5,
                    stride=2,
                    groups=out_chan,
                    dilation=1,
                    padding=4,  # padding=((5 - 1) // 2) * 1, causal padding: kernel_size - 1
                    norm_type=norm_type,
                )
            )
        # ----------Fusion Layer----------
        self.fuse_layers = nn.ModuleList([])
        for i in range(upsampling_depth):
            fuse_layer = nn.ModuleList([])
            for j in range(upsampling_depth):
                if i == j:
                    fuse_layer.append(None)
                elif j - i == 1:
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(
                        ConvNorm(
                            out_chan,
                            out_chan,
                            kernel_size=5,
                            stride=2,
                            groups=out_chan,
                            dilation=1,
                            padding=4,  # padding=((5 - 1) // 2) * 1, causal padding: kernel_size - 1
                            norm_type=norm_type,
                        )
                    )
            self.fuse_layers.append(fuse_layer)
        self.concat_layer = nn.ModuleList([])
        # ----------Concat Layer----------
        for i in range(upsampling_depth):
            if i == 0 or i == upsampling_depth - 1:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 2, out_chan, 1, 1, norm_type=norm_type, act_type=act_type
                    )
                )
            else:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 3, out_chan, 1, 1, norm_type=norm_type, act_type=act_type
                    )
                )
        self.last_layer = nn.Sequential(
            ConvNormAct(
                out_chan * upsampling_depth, out_chan, 1, 1, norm_type=norm_type, act_type=act_type
            )
        )
        self.res_conv = nn.Conv1d(out_chan, in_chan, 1)
        # ----------parameters-------------
        self.depth = upsampling_depth

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            wav_length = output[i].shape[-1]
            y = torch.cat(
                (
                    self.fuse_layers[i][0](output[i - 1])
                    if i - 1 >= 0
                    else torch.Tensor().to(output1.device),
                    output[i],
                    F.interpolate(output[i + 1], size=wav_length, mode="nearest")
                    if i + 1 < self.depth
                    else torch.Tensor().to(output1.device),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layer[i](y))

        wav_length = output[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(x_fuse[i], size=wav_length, mode="nearest")

        concat = self.last_layer(torch.cat(x_fuse, dim=1))
        expanded = self.res_conv(concat)
        return expanded + residual
    
class ConcatFC2(nn.Module):
    def __init__(self, ain_chan=128, vin_chan=128, norm_type="gLN"):
        super(ConcatFC2, self).__init__()
        self.W_wav = ConvNorm(ain_chan+vin_chan, ain_chan, 1, 1, norm_type=norm_type)
        self.W_video = ConvNorm(ain_chan+vin_chan, vin_chan, 1, 1, norm_type=norm_type)

    def forward(self, a, v):
        sa = F.interpolate(a, size=v.shape[-1], mode='nearest')
        sv = F.interpolate(v, size=a.shape[-1], mode='nearest')
        xa = torch.cat([a, sv], dim=1)
        xv = torch.cat([sa, v], dim=1)
        a = self.W_wav(xa)
        v = self.W_video(xv)
        return a, v
    
class Seaprator(nn.Module):
    """
    a(B, N, T) -> [pre_a] ----> [av_part] -> [post] -> (B, n_src, out_chan, T)
                                            /
    v(B, N, T) -> [pre_a]----------/

    [bottleneck]:   -> [layer_norm] -> [conv1d] ->
    [av_part]: Recurrent
    """

    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        an_repeats=4,
        fn_repeats=4,
        bn_chan=128,
        hid_chan=512,
        upsampling_depth=5,
        norm_type="gLN",
        mask_act="relu",
        act_type="prelu",
        # video
        vin_chan=256,
        vout_chan=256,
        vconv_kernel_size=3,
        vn_repeats=5,
        # fusion
        fout_chan=256,
        # video frcnn
        video_config=dict(),
        pretrain=None,
        fusion_shared=False,
        fusion_levels=None,
        fusion_type="ConcatFC2"
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.an_repeats = an_repeats
        self.fn_repeats = fn_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.upsampling_depth = upsampling_depth
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.act_type = act_type
        # video part
        self.vin_chan = vin_chan
        self.vout_chan = vout_chan
        self.vconv_kernel_size = vconv_kernel_size
        self.vn_repeats = vn_repeats
        # fusion part
        self.fout_chan = fout_chan

        self.fusion_shared = fusion_shared
        self.fusion_levels = fusion_levels if fusion_levels is not None \
            else list(range(self.an_repeats))
        self.fusion_type = fusion_type

        # pre and post processing layers
        self.pre_a = nn.Sequential(normalizations.get(norm_type)(in_chan), nn.Conv1d(in_chan, bn_chan, 1, 1))
        self.pre_v = nn.Conv1d(self.vout_chan, video_config["in_chan"], kernel_size=3, padding=2) #causal, padding=kernel_size-1
        self.post = nn.Sequential(nn.PReLU(), 
                        nn.Conv1d(fout_chan, n_src * in_chan, 1, 1),
                        activations.get(mask_act)())
        # main modules
        self.video_frcnn = VideoBlock(**video_config)
        self.audio_frcnn = AudioSubnetwork(bn_chan, hid_chan, upsampling_depth, norm_type, act_type)
        self.audio_concat = nn.Sequential(nn.Conv1d(bn_chan, hid_chan, 1, 1, groups=bn_chan), nn.PReLU())
        self.crossmodal_fusion = self._build_crossmodal_fusion(bn_chan, video_config["in_chan"])
        self.init_from(pretrain)

        print("Fusion levels", self.fusion_levels, type(self.fusion_levels))


    def _build_crossmodal_fusion(self, ain_chan, vin_chan):
        module = globals()[self.fusion_type]
        print("Using fusion:\n", module)
        if self.fusion_shared:
            return module(ain_chan, vin_chan, self.norm_type)
        else:
            return nn.ModuleList([
                module(ain_chan, vin_chan, self.norm_type) \
                    for _ in range(self.an_repeats)])


    def get_crossmodal_fusion(self, i):
        if self.fusion_shared:
            return self.crossmodal_fusion
        else:
            return self.crossmodal_fusion[i]


    def init_from(self, pretrain):
        if pretrain is None:
            return 
        print("Init pre_v and video_frcnn from", pretrain)
        state_dict = torch.load(pretrain, map_location="cpu")["model_state_dict"]

        frcnn_state_dict = dict()
        for k, v in state_dict.items():
            if k.startswith("module.head.frcnn"):
                frcnn_state_dict[k[18:]] = v
        self.video_frcnn.load_state_dict(frcnn_state_dict)

        pre_v_state_dict = dict(
            weight = state_dict["module.head.proj.weight"],
            bias = state_dict["module.head.proj.bias"])
        self.pre_v.load_state_dict(pre_v_state_dict)


    def fuse(self, a, v):
        """
            /----------\------------\ -------------\ 
        a --> [frcnn*] --> [frcnn*] --> ... ----[]---> [frcnn*] -> ... -> 
                                               /
        v --> [frcnn] ---> [frcnn] ---> ... --/
            \----------/------------/ 
        """
        res_a = a
        res_v = v

        # iter 0
        # a = self.audio_frcnn[0](a)
        a = self.audio_frcnn(a)
        v = self.video_frcnn.get_video_block(0)(v)
        if 0 in self.fusion_levels:
            # print("fusion", 0)
            a, v = self.get_crossmodal_fusion(0)(a, v)

        # iter 1 ~ self.an_repeats
        # assert self.an_repeats <= self.video_frcnn.iter
        for i in range(1, self.an_repeats):
            # a = self.audio_frcnn[i](self.audio_concat[i](res_a + a))
            a = self.audio_frcnn(self.audio_concat(res_a + a))
            frcnn = self.video_frcnn.get_video_block(i)
            concat_block = self.video_frcnn.get_concat_block(i)
            v = frcnn(concat_block(res_v + v))
            if i in self.fusion_levels:
                # print("fusion", i)
                a, v = self.get_crossmodal_fusion(i)(a, v)

        # audio decoder
        for _ in range(self.fn_repeats):
            a = self.audio_frcnn(self.audio_concat(res_a + a))
        return a

    def forward(self, a, v):
        # a: [4, 512, 3280], v: [4, 512, 50]
        B, _, T = a.size()
        # print(a.shape)
        # import pdb; pdb.set_trace()
        a = self.pre_a(a)
        v = self.pre_v(v)
        a = self.fuse(a, v)
        a = self.post(a)
        return a.view(B, self.n_src, self.out_chan, T)


class CTCNet_Causal(BaseModel):
    def __init__(
        self,
        n_src=1,
        out_chan=None,
        an_repeats=4,
        fn_repeats=4,
        bn_chan=128,
        hid_chan=512,
        norm_type="gLN",
        act_type="prelu",
        mask_act="sigmoid",
        upsampling_depth=5,
        # video
        vin_chan=256,
        vout_chan=256,
        vconv_kernel_size=3,
        vn_repeats=5,
        # fusion
        fout_chan=256,
        # enc_dec
        fb_name="free",
        kernel_size=16,
        n_filters=512,
        stride=8,
        encoder_activation=None,
        sample_rate=8000,
        video_config=dict(),
        pretrain=None,
        fusion_levels=None,
        fusion_type="ConcatFC2"
    ):
        super(CTCNet_Causal, self).__init__(sample_rate=sample_rate)
        self.enc_kernel_size = kernel_size
        self.enc_num_basis = n_filters
        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(
            n_filters // 2 * 2 ** upsampling_depth
        ) // math.gcd(n_filters // 2, 2 ** upsampling_depth)

        # Front end
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        self.sm = Seaprator(
            in_chan=n_filters,
            n_src=n_src,
            out_chan=out_chan,
            an_repeats=an_repeats,
            fn_repeats=fn_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            upsampling_depth=upsampling_depth,
            norm_type=norm_type,
            mask_act=mask_act,
            act_type=act_type,
            vin_chan=vin_chan,
            vout_chan=vout_chan,
            vconv_kernel_size=vconv_kernel_size,
            vn_repeats=vn_repeats,
            fout_chan=fout_chan,
            video_config=video_config,
            pretrain=pretrain,
            fusion_levels=fusion_levels,
            fusion_type=fusion_type
        )

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=n_filters * n_src,
            out_channels=n_src,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=(kernel_size // 2) - 1,
            groups=1,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.decoder.weight)

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, window - stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    # Forward pass
    def forward(self, audio_mixture, mouth_embedding):
        # input shape: (B, T)
        was_one_d = False
        if audio_mixture.ndim == 1:
            was_one_d = True
            audio_mixture = audio_mixture.unsqueeze(0)
        if audio_mixture.ndim == 2:
            audio_mixture = audio_mixture
        if audio_mixture.ndim == 3:
            audio_mixture = audio_mixture.squeeze(1)

        x = self.pad_to_appropriate_length(audio_mixture)
        # Front end
        x = self.encoder(x.unsqueeze(1))
        s = x.clone()
        # Separation module
        x = self.sm(x, mouth_embedding)
        x = x * s.unsqueeze(1)
        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        estimated_waveforms = self.remove_trailing_zeros(estimated_waveforms, audio_mixture)
        if was_one_d:
            return estimated_waveforms.squeeze(0)
        return estimated_waveforms

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1]) +
                [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=torch.float32).to(x.device)
            padded_x[..., :x.shape[-1]] = x
            return padded_x
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]
    
    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args

if __name__ == '__main__':
    model = CTCNet_Causal()
    print(model(torch.randn(1, 16000), torch.randn(1, 25, 512)).shape)