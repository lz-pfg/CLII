# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************


from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import torchvision.transforms as T
import cv2
from .network import KernelConv
from . import utils as kpn_utils
from .dm import *
import math
import torch.nn.functional as F
import torchvision.models as models

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # self.positional_embedding = PositionEmbeddingSine(spacial_dim ** 2 + 1, num_pos_feats=embed_dim // 2)
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # self.positional_embedding = PositionEmbeddingSine(spacial_dim ** 2, num_pos_feats=1024)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        #self.positional_embedding = nn.Parameter(torch.randn(4097, 512) / 512 ** 0.5)
        # self.positional_embedding = PositionEmbeddingSine(spacial_dim ** 2, num_pos_feats=1024)
        #self.k_proj = nn.Linear(512, 512)
        #self.q_proj = nn.Linear(512, 512)
        #self.v_proj = nn.Linear(512, 512)
        #self.c_proj = nn.Linear(512, 384)
        #self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC (256, 2, 1024)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC (257,2,1024)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, att_maps = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )

        return x, att_maps


class ResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension 2048=64*32  512/32=16
        #self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        self.attnpool = AttentionPool2d(16, embed_dim//2, heads, output_dim)
        #self.attnpool2 = AttentionPool2d(64, 512, heads, output_dim)
        self.attnpool2 = AttentionPool2d(32, 512, heads, output_dim)
        #self.attnpool2 = AttentionPool2d(32, 1024, heads, output_dim)
        #self.attnpool2 = AttentionPool2d(32, 512, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x, att_maps = self.attnpool(x)
        

        return x, att_maps 


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, mask):
        mask = mask.to(device=x.device) if mask is not None else None
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=mask, attn_mask=self.attn_mask)[0]
        
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x: list):
        x, mask = x
        x = x + self.attention(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return [x, mask]


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class ResidualAttentionBlockDecoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q, k, v, im_m):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, attn_mask=self.attn_mask, key_padding_mask=im_m)


    def forward(self, x: list):
        if len(x) == 4:
            q, k, v, im_m = x
        else:
            q, k, v, im_m, m = x
        q_, m = self.attention(q, k, v, im_m)
        q = q + self.ln_1(q_)
        q = q + self.mlp(self.ln_2(q))
        return [q, k, v, im_m, m]


class ResidualAttentionBlockDecoder2(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q, k, v, im_m):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, attn_mask=self.attn_mask, key_padding_mask=im_m)


    def forward(self, x: list):
        if len(x) == 4:
            q, k, v, im_m = x
        else:
            q, k, v, im_m, m = x

        q_, m = self.attention(q, k, v, im_m)
        q = q + self.ln_1(q_)
        q = q + self.mlp(self.ln_2(q))
        return [q, k, v, im_m, m]

class TransformerDecoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockDecoder(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)

class TransformerDecoder2(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width 
        self.layers = layers 
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockDecoder2(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)

class TransformerDecoder3(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width #384
        self.layers = layers #=6
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockDecoder2(width, heads, attn_mask) for _ in range(layers)])


    def forward(self, x):
        return self.resblocks(x)



class UpsampleOneStep(nn.Sequential):
    
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class LayerNorm2(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm2, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm2(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention(nn.Module):
    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, patchsize, d_model):
        super().__init__()

        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d( d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d( d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def forward(self, x):
        b, c, h, w = x.size()
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1),
                                                      torch.chunk(_key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)
                                                    ):
            out_w, out_h = w // width, h // height

            # 1) embedding and reshape
            query = query.view(b, d_k, out_h, height, out_w, width)
            query = query.permute(0, 2, 4, 1, 3, 5).contiguous().view(b,  out_h*out_w, d_k*height*width)

            key = key.view(b, d_k, out_h, height, out_w, width)
            key = key.permute(0, 2, 4, 1, 3, 5).contiguous().view( b,  out_h*out_w, d_k*height*width)

            value = value.view(b, d_k, out_h, height, out_w, width)
            value = value.permute(0, 2, 4, 1, 3, 5).contiguous().view( b, out_h*out_w, d_k*height*width)

            y, _ = self.attention(query, key, value)

            y = y.view(b, out_h, out_w, d_k, height, width)
            y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k, h, w)

            output.append(y)

        output = torch.cat(output, 1)
        x = self.output_linear(output)

        return x


class MultiHeadedAttention2(nn.Module):
    def __init__(self, patchsize, d_model):
        super().__init__()

        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(d_model, 384, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d( d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d( d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()
        self.ln = nn.Linear(256, 384, bias=True)
        self.out = nn.Linear(384, 256, bias=True)
        self.out2 = nn.Conv2d(384, 256, kernel_size=1, padding=0)

    def forward(self, x, texts):
        b, c, h, w = x.size() #(2,256,64,64)
      
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x) #(2,384,64,64)
        _key = texts #(2,32,384)
        _value = texts
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1),
                                                      torch.chunk(_key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)
                                                    ): 
            out_w, out_h = w // width, h // height

            # 1) embedding and reshape
            query = query.view(b, 384, out_h, height, out_w, width) #(2,384,64,1,64,1)
            query = query.permute(0, 2, 4, 1, 3, 5).contiguous().view(b,  out_h*out_w, 384*height*width) #(2,4096,384)

            y, _ = self.attention(query, key, value) #(2,4096,384)

            y = y.view(b, out_h, out_w, 384, height, width) #(2,64,64,128,1,1)
            y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, 384, h, w) #(2,128,64,64)
            y = self.out2(y)
            output.append(y)
       
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        
        return x


class TransformerBlock(nn.Module):

    def __init__(self, patchsize=[], c=256, depth=4):
        super().__init__()
        self.depth = depth
        self.attention = MultiHeadedAttention(patchsize= patchsize, d_model=c)

    def forward(self, x):
        for i in range(self.depth):
            x = self.attention(x)

        return x

class TransformerBlock2(nn.Module):

    def __init__(self, patchsize=[], c=256, depth=4):
        super().__init__()
        self.depth = depth
        self.attention = MultiHeadedAttention2(patchsize= patchsize, d_model=c)

    def forward(self, x, texts):
        for i in range(self.depth):
            x = self.attention(x, texts)

        return x

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  #'bilinear'
        return self.conv(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class InpaintGenerator(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)

        self.kpn_model = kpn_utils.create_generator()

        if init_weights:
            self.init_weights()

        self.transformer33 = TransformerBlock2(patchsize=[(1, 1)], c=256, depth=6)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, track_running_stats=False),
            nn.ReLU(True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, track_running_stats=False),
            nn.ReLU(True)
        )
        self.attnpool = AttentionPool2d(16, 1024, 32, 384)
       

    def forward(self, x, texts): 
        inputs = x.clone()

        x = self.encoder0(x) 
        x = self.encoder1(x) 

        kernels, kernels_img = self.kpn_model(inputs, x)

        x = self.encoder2(x) 
        x = self.kernel_pred(x, kernels, white_level=1.0, rate=1)

        x = self.middle(x) 

        
        x_mid = x
        x_mid = self.encoder3(x_mid)  
        x_mid = self.encoder4(x_mid)  

        x_mid, att_maps = self.attnpool(x_mid)

        x3_1 = self.transformer33(x, texts)
        x = self.decoder(x3_1)
        

        x = self.kernel_pred(x, kernels_img, white_level=1.0, rate=1)

        x = (torch.tanh(x) + 1) / 2

        return x, x_mid, att_maps
        

    def save_feature(self, x, name):
        x = x.cpu().numpy()
        np.save('./result/{}'.format(name), x)

class img_en(nn.Module):
    def __init__(self, dim=1, dilation=1, use_spectral_norm=False):
        super(img_en, self).__init__()

        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, track_running_stats=False),
            nn.ReLU(True)
        )
        self.attnpool = AttentionPool2d(16, 1024, 32, 384)

    def forward(self, x):
        x = self.encoder4(x) 
        x, att = self.attnpool(x)
        return x, att

class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

class InpaintGenerator2(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8, init_weights=True):
        super(InpaintGenerator2, self).__init__()
        self.kernel_size = None #config.kernel_size
        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)
        self.kpn_model = kpn_utils.create_generator()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(256+256, 256, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256+256, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256+256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        
        self.transformer2 = TransformerBlock(patchsize=[(2,2)], c = 256, depth=2)
        self.transformer3 = TransformerBlock(patchsize=[(1,1), (2,2)], c = 256, depth=6)

        self.transformer33 = TransformerBlock2(patchsize=[(1, 1), (2, 2)], c=256, depth=6)
        

        self.decoder = nn.Sequential(
            deconv(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

        if init_weights:
            self.init_weights()


    def forward(self, x, texts):
        
        x = self.encoder(x) #(2, 256, 128, 128)

        # x1_1 = self.transformer1(x)
        # x1_2 = torch.cat([x1_1, x], dim=1)
        # x1_3 = self.conv1(x1_2)

        x2_1 = self.transformer2(x) 
        x2_2 = torch.cat([x, x2_1], dim=1) 
        x2_3 = self.conv2(x2_2) 
        
        #x2_3 = x #(2,256,64,64)
        #x3_1 = self.transformer3(x2_3) 
        x3_1 = self.transformer33(x2_3, texts)  
        x3_2 = torch.cat([x2_3, x3_1], dim=1)
        x3_3 = self.conv3(x3_2) 

        x = self.decoder(x3_3) 

        #x = (torch.tanh(x) + 1) / 2


        return x


class image_encode(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8, init_weights=True):
        super(image_encode, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)

        self.kpn_model = kpn_utils.create_generator()

        if init_weights:
            self.init_weights()

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, track_running_stats=False),
            nn.ReLU(True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, track_running_stats=False),
            nn.ReLU(True)
        )
        self.attnpool = AttentionPool2d(16, 1024, 32, 384)
        self.out = nn.Linear(1024, 384, bias=True)
        self.conv = nn.Conv2d(1024, 384, kernel_size=1)


    def forward(self, x):
        # def forward(self, x, texts):
        inputs = x.clone()

        x = self.encoder0(x)  
        x = self.encoder1(x) 

        kernels, kernels_img = self.kpn_model(inputs, x)

        x = self.encoder2(x) 
        x = self.kernel_pred(x, kernels, white_level=1.0, rate=1)

        x = self.middle(x) 
        x_mid = x
        x = self.encoder3(x) 
        x = self.encoder4(x)  

        x, att_maps = self.attnpool(x)
        #x = x.reshape([x.shape[0], 1024, -1]).permute(0, 2, 1) 
        #x = self.out(x) 

        #x = self.conv(x)
        #x = x.reshape([x.shape[0], 384, -1]).permute(0, 2, 1) 

        return x, x_mid, att_maps, kernels_img

class image_decode(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8, init_weights=True):
        super(image_decode, self).__init__()
        # self.filter_type = config.FILTER_TYPE
        # self.kernel_size = config.kernel_size

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)

        self.kpn_model = kpn_utils.create_generator()

        if init_weights:
            self.init_weights()

        self.transformer33 = TransformerBlock2(patchsize=[(1, 1)], c=256, depth=6)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    
    def forward(self, x, texts, kernels_img):
        

        x3_1 = self.transformer33(x, texts)  

        x = self.decoder(x3_1)  
        

        x = self.kernel_pred(x, kernels_img, white_level=1.0, rate=1)

        return x


class oCLIP(nn.Module):
    def __init__(self,
                 first_stage: bool,embed_dim: int,
                 image_resolution: int,vision_layers: Union[Tuple[int, int, int, int], int],vision_width: int,
                 vision_patch_size: int,               
                 context_length: int, vocab_size: int, transformer_width: int, transformer_heads: int, transformer_layers: int,
                 transformer_decoder_layers: int
                 ):
        super().__init__()

        self.context_length = context_length
        self.first_stage = first_stage

        vision_heads = vision_width * 32 // 64

        self.visual = ResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.transformer_heads = transformer_heads
        self.transformer_width = transformer_width
    
        if not self.first_stage:
            self.transformer_decoder = TransformerDecoder(
                width=embed_dim,
                layers=transformer_decoder_layers,
                heads=transformer_heads,
            )

            self.transformer_decoder2 = TransformerDecoder2(
                width=384, #384, #2048, #embed_dim,
                layers=6, #transformer_decoder_layers,
                heads=8 #8, #8, #32, #transformer_heads,
            )
            self.transformer_decoder3 = TransformerDecoder2(
                width=embed_dim,
                layers=transformer_decoder_layers,
                heads=transformer_heads,
            )


        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        
    
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if not self.first_stage:

            self.ln_final_decoder = LayerNorm(embed_dim)
            self.text_class = nn.Linear(embed_dim, vocab_size)

            self.image_pos = nn.Parameter(torch.randn((image_resolution // 32) ** 2, embed_dim) / embed_dim ** 0.5)
            
            self.image_pos2 = nn.Parameter(torch.randn(1025, 384) / 384 ** 0.5) 
            
            self.image_pos3 = nn.Parameter(torch.randn(1024, 384) / 384 ** 0.5)

        self.initialize_parameters()


        self.ln_final2 = LayerNorm(384)
        
        self.ln_in = nn.Linear(384, 1024, bias=True)
        
        self.tx_in = nn.Linear(384, 2048, bias=True)
        self.ln_final_decoder2 = LayerNorm(384)
        self.ln_final_decoder22 = LayerNorm(2048)

        self.ln_final_decoder3 = LayerNorm(1024)
        
        self.ln_out = nn.Linear(384, 256, bias=True)

        self.decoder_pred = nn.Linear(384, 2048, bias=True)
        self.resize = nn.Upsample(size=(512, 512), mode='nearest')

        self.conv3 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        mea = [0.48145466, 0.4578275, 0.40821073]
        self.mea = torch.Tensor(mea).view(3, 1, 1)
        std = [0.26862954, 0.26130258, 0.27577711]
        self.std = torch.Tensor(std).view(3, 1, 1)

        self.post_quant_conv = torch.nn.Conv2d(512, 384, 1)
        self.decode = Decoder(ch=128,out_ch=3,num_res_blocks=2,attn_resolutions=[], in_channels=512, resolution=256, z_channels=4)
        self.encode = Encoder(ch=128,out_ch=512,num_res_blocks=2,attn_resolutions=[], in_channels=3, resolution=256, z_channels=4)
        self.outconv = nn.Conv2d(384,512,1)

        self.generator = InpaintGenerator()
        #self.generator2 = InpaintGenerator2()
        self.image_encode = image_encode()
        self.image_decode = image_decode()
        self.img_en = img_en()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if isinstance(self.visual, ResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)


        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if not self.first_stage:
            for block in self.transformer_decoder.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            for block in self.transformer_decoder2.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            for block in self.transformer_decoder3.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        if not self.first_stage:
            nn.init.normal_(self.text_class.weight, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal

        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
        

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text, mask):
        
        batch_size, n_words, n_chars = text.shape
        text = text.reshape(batch_size * n_words, n_chars)
        x = self.token_embedding(text).type(self.dtype) 

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer([x, mask])[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        

        
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x.reshape(batch_size, n_words, x.shape[-1])
        

        return x

    def att_text_to_image(self, encoded_image, encoded_text, image_mask):
        x = encoded_text.permute(1, 0, 2)  # NLD -> LND
        tmp = self.transformer_decoder([x, encoded_image + self.image_pos[:, None, :].to(encoded_image.dtype), encoded_image, image_mask])
        
        x = tmp[0]
        m = tmp[4]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final_decoder(x).type(self.dtype)
        return x, m

    def img_decode(self, encoded_image, encoded_text, image_mask):
        encoded_image = self.ln_final2(encoded_image)  
        
        text = encoded_text.permute(1, 0, 2) 
        q = encoded_image + self.image_pos2[:, None, :].to(encoded_image.dtype)
       
        res = self.transformer_decoder3([q, text, text, image_mask])

        x = res[0] 
        x = x[1:] 
        x = self.ln_in(x) 
        
        x = x.permute(1, 0, 2)  
        
        x = self.ln_final_decoder3(x).type(self.dtype)
        x = x.permute(0, 2, 1)  
        
        x = x.reshape([x.shape[0], 256, 64, 64])
        
        imgs = self.res2(x)
        imgs = (torch.tanh(imgs) + 1) / 2
        
        return imgs

    def img_decode2(self, encoded_image, encoded_text, image_mask):
        encoded_image = self.post_quant_conv(encoded_image) 
        q = encoded_image.reshape([encoded_image.shape[0], encoded_image.shape[1], -1]) 
        q = q.permute(2,0,1) 
        q = q + self.image_pos3[:, None, :].to(encoded_image.dtype)
        #res = self.transformer_decoder3([q, q, q, image_mask])
        text = encoded_text.permute(1, 0, 2)
        res = self.transformer_decoder3([q, text, text, image_mask])
        x = res[0]
        x = x.permute(1,2,0)
        x = x.reshape([x.shape[0],x.shape[1],32, 32])
        x = self.outconv(x) #(2,512,32,32)

        return x

    def forward(self, image, text, image_mask, masked_imgs, masks):

        image_mask = image_mask.flatten(1, 2)
        encoded_texts = self.encode_text(text, None)
        logit_scale = self.logit_scale.exp()
        
        images_masked = image * (1 - masks)     
        inputs = torch.cat((images_masked, masks), dim=1) 
        outputs, encoded_image, att_maps = self.generator(inputs, encoded_texts)

        image_features = encoded_image[0]
        encoded_image = encoded_image[1:]

        text_features = torch.mean(encoded_texts, dim=1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if not self.first_stage:
            
            text_image_enc, char_mask = self.att_text_to_image(encoded_image, encoded_texts, image_mask)
            
            h = self.encode(masked_imgs) #(2,8,32,32)->(2,512,32,32) dm
            res = self.img_decode2(h, encoded_texts, None)
            outputs = self.decode(res) 
            
            text_logits = self.text_class(text_image_enc)

        if self.training:
            if not self.first_stage:
                return image_features, text_features, text_logits, logit_scale, outputs  #res_im
            else:
                return image_features, text_features, logit_scale, res_im
        else:
            if not self.first_stage:
                return image_features, text_features, text_logits, att_maps, char_mask, logit_scale, outputs
            else:
                return image_features, text_features, att_maps, logit_scale, res_im


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
        # for name in ["proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    # context_length = state_dict["positional_embedding"].shape[0]
    context_length = 256
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = oCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
