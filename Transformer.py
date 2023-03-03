import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from DWT import DWT_2D, IDWT_2D


class GroupNorm(nn.GroupNorm):
    def __init__(self, channel):
        super(GroupNorm, self).__init__(1, channel)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ChannelPatchEmbed(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_szie=7, stride=4, padding=1, norm_layer=GroupNorm):
        super().__init__()
        kernel_szie = to_2tuple(kernel_szie)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)

        self.proj = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_szie, stride=stride, padding=padding)
        self.norm = norm_layer(out_channel)

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

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channel=32, out_channel=16, kernel_size=3, stride=1, padding=1, bias=True, groups=1):
        super(ConvLayer, self).__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias, groups=groups)

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

    def forward(self, x):
        return self.conv(x)


############################################################
class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = in_features
        hidden_features = in_features * mlp_ratio
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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

    def forward(self, x, x_size):
        H, W  = x_size
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ChannelDWConv(nn.Module):
    def __init__(self, channel=768):
        super(ChannelDWConv, self).__init__()
        self.dwconv = ConvLayer(channel, channel, groups=channel)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class ChannelMlp(nn.Module):
    def __init__(self, in_channel, mlp_ratio=4, drop=0.):
        super(ChannelMlp, self).__init__()
        mid_channel = in_channel * mlp_ratio
        self.fc1 = ConvLayer(in_channel, mid_channel, 1, 1, 0)
        self.dwconv = ChannelDWConv(mid_channel)
        self.act = nn.GELU()
        self.fc2 = ConvLayer(mid_channel, in_channel, 1, 1, 0)
        self.drop = nn.Dropout(drop)

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

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Pooling(nn.Module):
    def __init__(self, pool_size=3, beta=1):
        super().__init__()
        self.beta = beta
        self.pool = nn.AvgPool2d(kernel_size=pool_size,
                                 stride=1,
                                 padding=pool_size // 2,
                                 count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - self.beta * x


class PoolFormerBlock(nn.Module):
    def __init__(self, channel, mlp_ratio=4, beta=1, drop_rate=0., norm_layer=GroupNorm, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = GroupNorm(channel)
        self.token_mixer = Pooling(pool_size=3, beta=beta)
        self.norm2 = norm_layer(channel)
        self.mlp = ChannelMlp(channel, mlp_ratio=mlp_ratio)
        self.drop_path = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((channel)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((channel)), requires_grad=True)

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

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.task_query = nn.Parameter(torch.randn(1, 64, dim))

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

    def forward(self, x, x_size):
        B, N, C = x.shape
        task_q = self.task_query

        if B > 1:
            task_q = task_q.unsqueeze(0).repeat(B, 1, 1, 1)
            task_q = task_q.squeeze(1)

        q = self.q(task_q).reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.interpolate(q, size=(v.shape[2], v.shape[3]))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class SpatialBlock(nn.Module):
    def __init__(self, dim=96, num_heads=8, window_size=8,
                 mlp_ratio=4, norm_layer=nn.LayerNorm, drop_rate=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = SpatialAttention(dim, num_heads)

        self.drop_path = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio)

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

    def forward(self, x, x_size):
        x = x + self.drop_path(self.attn(self.norm1(x), x_size))
        x = x + self.drop_path(self.mlp(self.norm2(x), x_size))

        return x


class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.k = nn.Linear(dim, dim)
        self.task_query = nn.Parameter(torch.randn(1, 64, dim))

        self.qv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

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

    def forward(self, x):
        B, N, C = x.shape

        qv = self.qv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, v = qv[0], qv[1]

        task_q = self.task_query

        if B > 1:
            task_q = task_q.unsqueeze(0).repeat(B, 1, 1, 1)
            task_q = task_q.squeeze(1)

        k = self.k(task_q).reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = torch.nn.functional.interpolate(k, size=(v.shape[2], v.shape[3]))

        attention = (k.transpose(-1, -2) @ v ) * self.scale
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):

    def __init__(self, dim=96, num_heads=8, mlp_ratio=4, norm_layer=nn.LayerNorm, drop_rate=0.):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads)
        self.drop_path = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio)

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

    def forward(self, x, x_size):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), x_size))
        return x


class WaveAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')

        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.filter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.ReLU(inplace=True),
        )

        self.kv_embed = nn.Identity()
        self.q = nn.Linear(dim, dim)
        self.task_query = nn.Parameter(torch.randn(1, 64, dim))
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )
        self.proj = nn.Linear(dim+dim//4, dim)

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

    def forward(self, x, x_size):
        H, W = x_size
        B, N, C = x.shape
        task_q = self.task_query

        if B > 1:
            task_q = task_q.unsqueeze(0).repeat(B, 1, 1, 1)
            task_q = task_q.squeeze(1)
        q = self.q(task_q).reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # print(q.shape)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x_dwt = self.dwt(self.reduce(x))
        x_dwt = self.filter(x_dwt)

        x_idwt = self.idwt(x_dwt)
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-2)*x_idwt.size(-1)).transpose(1, 2)

        kv = self.kv_embed(x_dwt).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # 8, 4, 64
        q = torch.nn.functional.interpolate(q, size=(v.shape[2]*4, v.shape[3]))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(torch.cat([x, x_idwt], dim=-1))
        return x


class WaveBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WaveAttention(dim, num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

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

    def forward(self, x, x_size):
        x = x + self.drop_path(self.attn(self.norm1(x), x_size))
        x = x + self.drop_path(self.mlp(self.norm2(x), x_size))
        return x