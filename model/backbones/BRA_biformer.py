"""
BiFormer impl.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg

from model.backbones.BRA_utils import BiLevelRoutingAttention,Attention, AttentionLePE, DWConv


def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    # if name == 'sum':
    #     return Summer(PositionalEncodingPermute2D(emb_dim))
    # elif name == 'npe.sin':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
    # elif name == 'npe.coord':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
    # elif name == 'hpe.conv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
    # elif name == 'hpe.dsconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
    # elif name == 'hpe.pointconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
    else:
        raise ValueError(f'PE name {name} is not surpported!')

class hfa(nn.Module):
    def __init__(self, in_dims=[96, 192, 384], global_channels=768, num_heads=8):
        super().__init__()
        self.global_channels = global_channels

        # Align each local feature dimension to global_channels
        self.align_layers = nn.ModuleList([
            nn.Conv2d(in_dim, global_channels, kernel_size=1)
            for in_dim in in_dims
        ])

        # MultiheadAttention
        self.attn = nn.MultiheadAttention(embed_dim=global_channels, num_heads=num_heads, batch_first=True)
        self.attn_patch = nn.MultiheadAttention(embed_dim=global_channels, num_heads=num_heads, batch_first=True)
        # Refinement after feature fusion
        self.refine = nn.Conv2d(2 * global_channels, global_channels, kernel_size=1)

    def forward(self, locals, global_feat):
        """
        local1: [B, 96, 49, 196] -> patch size 14x14
        local2: [B, 192, 49, 49] -> patch size 7x7
        local3: [B, 384, 49, 16] -> patch size 4x4
        global_feat: [B, 768, 12, 12]
        """
        B, C, H, W = global_feat.shape  # C = 768

        # locals = [local1, local2, local3]
        aligned = []
        aligned_patch = []
        for i in range(3):
            x = self.align_layers[i](locals[i])        # [B, 768, N, S]
            pt_x =  x.detach()
            pt_x = F.adaptive_avg_pool2d(pt_x, (pt_x.size(2), 1))  # [B, 768, N, 1]
            pt_x = pt_x.squeeze(-1)  # [B, 768, N]
            aligned_patch.append(pt_x)
            x_pooled = F.adaptive_avg_pool2d(x, (1, 1)) # [B, 768, 1, 1]
            aligned.append(x_pooled.squeeze(-1).squeeze(-1))  # → [B, 768]

        # [B, 3, 768]
        seq = torch.stack(aligned, dim=1)
        seq_patch = torch.stack(aligned_patch, dim=1) # [B, 3, 768, N]

        attn_out, _ = self.attn(seq, seq, seq)  # [B, 3, 768]
        fused = attn_out.mean(dim=1).view(B, C, 1, 1)  # [B, 768, 1, 1]

        # [B * N, 3, 768]
        B_patch, _, C_patch, N_patch = seq_patch.shape
        seq_patch = seq_patch.permute(0, 3, 1, 2).reshape(B_patch * N_patch, 3, C_patch)  # [B * N, 3, 768]

        attn_out_patch, _ = self.attn_patch(seq_patch, seq_patch, seq_patch)  # [B * N, 3, 768]

        # [B, 768, N]
        attn_out_patch = attn_out_patch.reshape(B_patch, N_patch, 3, C_patch).permute(0, 2, 3, 1)
        fused_patch = attn_out_patch.mean(dim=1)  # [B, 768, N]
        fused_up = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([global_feat, fused_up], dim=1)
        # out = fused_up + global_feat
        out = self.refine(out)  # [B, 768, 12, 12]
        return out, fused_up, global_feat, fused_patch

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=True,depth_j=None,depth_i=None):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad,depth_j=depth_j,depth_i=depth_i)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        out_patch = None

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                # 捕获BiLevelRoutingAttention返回的out_patch
                attn_output = self.attn(self.norm1(x))
                if isinstance(attn_output, tuple):  # 当返回(out, out_patch)时
                    attn_out, out_patch = attn_output
                else:
                    attn_out = attn_output
                x = x + self.drop_path(attn_out)
                x = x + self.drop_path(self.mlp(self.norm2(x)))

                # x = x + self.drop_path(self.attn(self.norm1(x), vclass_id=vclass_id))  # (N, H, W, C)
                # x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        if out_patch is not None:
            out_patch = out_patch.permute(0, 3, 1, 2)  # (n, p^2, w^2, m*c) -> (N, C, p^2, w^2)
        return x, out_patch


class BiFormer(nn.Module):
    def __init__(self, depth=[3, 4, 8, 3], in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=True,
                 # -----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # self.head2 = nn.Linear(num_classes, 768)
        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad,
                        depth_j=j,
                        depth_i=depth[i]) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        self.norm_fused_out = nn.BatchNorm2d(768)
        self.norm_global_feat = nn.BatchNorm2d(768)
        self.norm_local_feat = nn.BatchNorm1d(768)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.head1 = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.head2 = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        self.global_patch_fusion = hfa()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x, camera_id, view_id):

        stage_outputs = []

        for i in range(4):
            x = self.downsample_layers[i](x)
            for j, blk in enumerate(self.stages[i]):
                x, out_patch = blk(x)
                if j == len(self.stages[i]) - 1 and out_patch is not None:
                    # stage_outputs.append(out_patch.detach())
                    stage_outputs.append(out_patch)

        x, fused_out, global_feat, fused_patch = self.global_patch_fusion(stage_outputs,x)
        x= self.norm(x)
        fused_out = self.norm_fused_out(fused_out)
        global_feat = self.norm_global_feat(global_feat)
        fused_patch = self.norm_local_feat(fused_patch)
        x = self.pre_logits(x)

        return x, fused_out, global_feat, fused_patch


    def forward(self, x, cam_label=None, view_label=None):
        x, fused_out, global_feat, fused_patch = self.forward_features(x ,cam_label, view_label)
        x = x.flatten(2).mean(-1)
        fused_out = fused_out.flatten(2).mean(-1)
        global_feat = global_feat.flatten(2).mean(-1)
        x = self.head(x)
        fused_out = self.head1(fused_out)
        global_feat = self.head2(global_feat)

        return x, fused_out, global_feat, fused_patch


#################### model variants #######################


model_urls = {
    "biformer_tiny_in1k": "https://matix.li/e36fe9fb086c",
    "biformer_small_in1k": "https://matix.li/5bb436318902",
    "biformer_base_in1k": "https://matix.li/995db75f585d",
}


# https://github.com/huggingface/pytorch-image-models/blob/4b8cfa6c0a355a9b3cb2a77298b240213fb3b921/timm/models/_factory.py#L93

@register_model
def biformer_tiny(pretrained=False,model_path=None, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[2, 2, 8, 2],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint["model"])
        in_channel = model.head.in_features
        model.head = nn.Linear(in_channel, 768)

    return model


@register_model
def biformer_small(pretrained=False, model_path='', pretrained_cfg=None,
                   pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[4, 4, 18, 4],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint["model"])
        in_channel = model.head.in_features
        model.head = nn.Linear(in_channel, 768)

    return model


@register_model
def biformer_base(pretrained=False, model_path='',pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[4, 4, 18, 4],
        embed_dim=[96, 192, 384, 768], mlp_ratios=[3, 3, 3, 3],
        # use_checkpoint_stages=[0, 1, 2, 3],
        use_checkpoint_stages=[],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        # topks=[1, 4, 16, -2],
        topks=[4, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[96, 192, 384, 768],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,

        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint["model"],strict=False)
        in_channel = model.head.in_features
        model.head = nn.Linear(in_channel, 768)
        model.head1 = nn.Linear(in_channel, 768)
        model.head2 = nn.Linear(in_channel, 768)
    else:
        in_channel = model.head.in_features
        model.head = nn.Linear(in_channel, 768)


    return model

