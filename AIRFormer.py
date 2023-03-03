import torch.nn.functional as F
from functools import partial
from DWT import *
from Transformer import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class EncoderTransformer(nn.Module):
    def __init__(self, in_channel=3,
                 out_channels=(64, 128, 320, 512),
                 freq_channels=(64*4, 128*4, 320*4, 512*4),
                 mlp_ratio=4,
                 beta=(0.1, 0.4, 0.7, 1),
                 norm_layer=GroupNorm,
                 depths=(2, 2, 2, 2),
                 sub_depths=(1, 1, 1),
                 drop_rate=0.1):
        super(EncoderTransformer, self).__init__()

        self.patch_embed1 = ChannelPatchEmbed(in_channel, out_channels[0],
                                              kernel_szie=7, stride=4, padding=3, norm_layer=norm_layer)
        self.patch_embed2 = ChannelPatchEmbed(out_channels[0], out_channels[1],
                                              kernel_szie=3, stride=2, padding=1, norm_layer=norm_layer)
        self.patch_embed3 = ChannelPatchEmbed(out_channels[1], out_channels[2],
                                              kernel_szie=3, stride=2, padding=1, norm_layer=norm_layer)
        self.patch_embed4 = ChannelPatchEmbed(out_channels[2], out_channels[3],
                                              kernel_szie=3, stride=2, padding=1, norm_layer=norm_layer)

        self.freq_patch_embed1 = ChannelPatchEmbed(freq_channels[0], freq_channels[1],
                                                   kernel_szie=3, stride=2, padding=1, norm_layer=norm_layer)
        self.freq_patch_embed2 = ChannelPatchEmbed(freq_channels[1], freq_channels[2],
                                                   kernel_szie=3, stride=2, padding=1, norm_layer=norm_layer)
        self.freq_patch_embed3 = ChannelPatchEmbed(freq_channels[2], freq_channels[3],
                                                   kernel_szie=3, stride=2, padding=1, norm_layer=norm_layer)

        dpr = [x.item() for x in torch.linspace(0, drop_rate, sum(depths))]
        cur = 0

        self.encoder1 = nn.ModuleList([PoolFormerBlock(out_channels[0], mlp_ratio=mlp_ratio, beta=beta[0], drop_rate=dpr[cur+i],
                                                       norm_layer=norm_layer) for i in range(depths[0])])
        self.encoder2 = nn.ModuleList([PoolFormerBlock(out_channels[1], mlp_ratio=mlp_ratio, beta=beta[1], drop_rate=dpr[cur+i],
                                                       norm_layer=norm_layer) for i in range(depths[1])])
        self.encoder3 = nn.ModuleList([PoolFormerBlock(out_channels[2], mlp_ratio=mlp_ratio, beta=beta[2], drop_rate=dpr[cur+i],
                                                       norm_layer=norm_layer) for i in range(depths[2])])
        self.encoder4 = nn.ModuleList([PoolFormerBlock(out_channels[3], mlp_ratio=mlp_ratio, beta=beta[3], drop_rate=dpr[cur+i],
                                                       norm_layer=norm_layer) for i in range(depths[3])])
        # self.freq_encoder1 = ConvLayer(freq_channels[1], freq_channels[1], 3, 1, 1, groups=freq_channels[1])
        # self.freq_encoder2 = ConvLayer(freq_channels[2], freq_channels[2], 3, 1, 1)
        # self.freq_encoder3 = ConvLayer(freq_channels[3], freq_channels[3], 3, 1, 1)

        self.norm1 = norm_layer(out_channels[0])
        self.norm2 = norm_layer(out_channels[1])
        self.norm3 = norm_layer(out_channels[2])
        self.norm4 = norm_layer(out_channels[3])

        self.pnorm1 = norm_layer(out_channels[1])
        self.pnorm2 = norm_layer(out_channels[2])
        self.pnorm3 = norm_layer(out_channels[3])

        self.ff = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # self.ff = nn.Conv2d(kernel_size=3, stride=1, padding=1, bias=False) # for comparison

        self.dwt = DWT_2D(wave='haar')

        self.refinement1 = nn.ConvTranspose2d(freq_channels[1], freq_channels[1]//4, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.refinement2 = nn.ConvTranspose2d(freq_channels[2], freq_channels[2]//4, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.refinement3 = nn.ConvTranspose2d(freq_channels[3], freq_channels[3]//4, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        # self.idwt = IDWT_2D(wave='haar')  # for comparison
        self.process_tail1 = nn.Conv2d(out_channels[1], out_channels[1], 3, 1, 1, groups=4)
        self.process_tail2 = nn.Conv2d(out_channels[2], out_channels[2], 3, 1, 1, groups=8)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        outs = []

        # stage 1
        x1 = self.patch_embed1(x)  # 64, 32, 32
        x2 = self.dwt(x1)
        x2 = self.ff(x2)
        x2 = self.freq_patch_embed1(x2)  # 512, 8, 8
        for blk in self.encoder1:
            x1 = blk(x1)
        x1 = self.norm1(x1)

        # for blk in self.freq_encoder1:
        #     x2 = blk(x2) + x2
        # x2 = self.freq_encoder1(x2)
        x2 = self.refinement1(x2)
        # x2 = self.idwt(x2)
        x2 = self.pnorm1(x2)  # 128, 16, 16

        outs.append(x1)

        # stage 2
        x1 = self.patch_embed2(x1) + x2  # 128, 16, 16
        x2 = self.dwt(x1)
        x2 = self.ff(x2)
        x2 = self.freq_patch_embed2(x2)  # 1280, 4, 4
        for blk in self.encoder2:
            x1 = blk(x1)
        x1 = self.norm2(x1)

        # for blk in self.freq_encoder2:
        #     x2 = blk(x2) + x2
        # x2 = self.freq_encoder2(x2)
        # x2 = self.idwt(x2)
        x2 = self.refinement2(x2)
        x2 = self.pnorm2(x2)  # 320, 8, 8

        outs.append(x1)

        x1 = self.process_tail1(x1)

        # stage 3
        x1 = self.patch_embed3(x1) + x2  # 320, 8, 8
        x2 = self.dwt(x1)
        x2 = self.ff(x2)
        x2 = self.freq_patch_embed3(x2)  # 2048, 2, 2
        for blk in self.encoder3:
            x1 = blk(x1)
        x1 = self.norm3(x1)

        # for blk in self.freq_encoder3:
        #     x2 = blk(x2) + x2
        # x2 = self.freq_encoder3(x2)
        # x2 = self.idwt(x2)
        x2 = self.refinement3(x2)
        x2 = self.pnorm3(x2)  # 512, 4, 4

        outs.append(x1)

        x1 = self.process_tail2(x1)

        # stage 4
        x1 = self.patch_embed4(x1) + x2  # 512, 8, 8
        # print(x1.shape)
        for blk in self.encoder4:
            x1 = blk(x1)
        x1 = self.norm4(x1)
        outs.append(x1)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, dim=152, num_head=8, mlp_ratio=4, drop_rate=0., norm_layer=nn.LayerNorm, depth=3):
        super().__init__()

        self.patch_embed1 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=dim, embed_dim=dim)

        dpr = [x.item() for x in torch.linspace(0, drop_rate, depth)]
        self.block1 = nn.ModuleList([
            SpatialBlock(dim, num_head, mlp_ratio=mlp_ratio, drop_rate=dpr[0]),
            WaveBlock(dim, num_head, mlp_ratio=mlp_ratio, drop_rate=dpr[1]),
            ChannelBlock(dim, num_head, mlp_ratio=mlp_ratio, drop_rate=dpr[2])
        ])
        self.norm1 = norm_layer(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        x = x[-1]
        B = x.shape[0]
        outs = []

        x, H, W = self.patch_embed1(x)
        x_size = (H, W)
        for i, blk in enumerate(self.block1):
            x = blk(x, x_size)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class Encoder(EncoderTransformer):
    def __init__(self,
                 in_channel=3,
                 out_channels=(64, 128, 320, 512),
                 freq_channels=(64*4, 128*4, 320*4, 512*4),
                 mlp_ratio=2,
                 beta=(0.1, 0.4, 0.7, 1),
                 norm_layer=GroupNorm,
                 depths=(2, 2, 2, 2),
                 drop_rate=0.):
        super(Encoder, self).__init__(in_channel=in_channel,
                                      out_channels=out_channels,
                                      freq_channels=freq_channels,
                                      mlp_ratio=mlp_ratio,
                                      beta=beta,
                                      norm_layer=norm_layer,
                                      depths=depths,
                                      drop_rate=drop_rate)


class Decoder(DecoderTransformer):
    def __init__(self,
                 dim=512,
                 num_head=8,
                 mlp_ratio=4,
                 drop_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depth=3):
        super(Decoder, self).__init__(dim=dim,
                                      num_head=num_head,
                                      mlp_ratio=mlp_ratio,
                                      norm_layer=norm_layer,
                                      depth=depth,
                                      drop_rate=drop_rate)


class Reconstruction(nn.Module):
    def __init__(self, channels=(512, 320, 128, 64, 32)):
        super(Reconstruction, self).__init__()

        self.up_512_512 = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.up_512_320 = UpsampleConvLayer(512, 320, kernel_size=4, stride=2)
        self.up_320_128 = UpsampleConvLayer(320, 128, kernel_size=4, stride=2)
        self.up_128_64 = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.up_64_16 = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.output = nn.Sequential(UpsampleConvLayer(16, 8, kernel_size=4, stride=2),
                                    ConvLayer(8, 3, 3, 1, 1))

        # self.resblock5 = ResidualBlock(512)
        self.resblock4 = ResidualBlock(320)
        self.resblock3 = ResidualBlock(128)
        self.resblock2 = ResidualBlock(64)
        self.resblock1 = ResidualBlock(16)

        # one step contains upsample and output
        self.output_one_step = UpsampleOutput(32, 3, up_scale=2)

    def forward(self, x1, x2):

        x = self.up_512_512(x2[0]) + x1[3]
        # x = self.resblock5(x) + x

        x = self.up_512_320(x)
        x = self.resblock4(x) + x1[2]

        x = self.up_320_128(x)
        x = self.resblock3(x) + x1[1]

        x = self.up_128_64(x)
        x = self.resblock2(x) + x1[0]

        x = self.up_64_16(x)
        x = self.resblock1(x)
        x = self.output(x)

        return x


class AIRFormer(nn.Module):
    def __init__(self):
        super(AIRFormer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.tail = Reconstruction()
        self.act = nn.Tanh()

    def forward(self, x):
        input_ = x
        x = self.encoder(x)
        x_ = self.decoder(x)
        x = self.tail(x, x_)
        return x


if __name__ == '__main__':

    myNet = AIRFormer()
    myNet = myNet.cuda()
    x = torch.randn((1, 3, 256, 256)).cuda()
    y = myNet(x)
    print(y.shape)

    import time
    start = time.time()
    for _ in range(100):
        y = myNet(x)
    end = time.time()
    print((end-start)/100.)


