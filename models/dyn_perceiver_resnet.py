import math

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from einops import rearrange, repeat
from torch import Tensor
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
# from .registry import register_model
from timm.models.registry import register_model
# from fairscale.nn import checkpoint_wrapper
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

from .perceiver_core import (
    CrossAttentionLayer,
    SelfAttentionBlock,
)
from .cnn_core.resnet import *


class DynPerceiver(nn.Module):
    def __init__(self,
                input_size: int=224,
                num_classes:int=100,
                cnn_arch: str="resnet18",
                num_SA_heads: list=[1,2,4,8],
                num_latents: int=32,
                num_latent_channels: int=None,
                dropout: float = 0.0,
                SA_widening_factor: int=4,
                activation_checkpointing: bool = False,
                spatial_reduction: bool=True,
                depth_factor: list=[1,1,1,3],
                output_dir: str='./',
                with_x2z=True,
                with_z2x=True,
                with_dwc=True,
                with_last_CA=True,
                with_isc=True,
                
                drop_rate=0.2,
                drop_path_rate=0.2,

                exit=-1,
                
                **kwargs):
        super().__init__()
        self.exit = exit
        if num_SA_heads is None:
            num_SA_heads = [1,2,4,8]
        self.num_classes = num_classes
        cnn = eval(f'{cnn_arch}')(drop_rate=drop_rate,
                                  drop_path_rate=drop_path_rate, num_classes=num_classes)
        self.cnn_stem = nn.Sequential(
            cnn.conv1,
            cnn.bn1,
            cnn.act1,
            cnn.maxpool
        )
        
        self.cnn_body_stage1 = cnn.layer1
        self.cnn_body_stage2 = cnn.layer2
        self.cnn_body_stage3 = cnn.layer3
        self.cnn_body_stage4 = cnn.layer4
        
        # self.cnn_body_last_conv1x1 = cnn.blocks[6]
        
        num_blocks_per_stage = [3*depth_factor[0], 3*depth_factor[1], 9*depth_factor[2], 3*depth_factor[3]]
        self.avgpool = cnn.global_pool
        # self.cnn_head_before_cls = nn.Sequential(
        #     cnn.conv_head,
        #     cnn.act2
        # )
        # self.flatten_cnn = cnn.flatten
        self.drop_rate_cnn = cnn.drop_rate
        self.classifier_cnn = cnn.fc
        
        print(cnn.drop_rate)
        
        self.spatial_reduction = spatial_reduction
        if spatial_reduction:
            self.ca_pooling = nn.AdaptiveAvgPool2d((7,7))
        def cross_attn(num_cross_attention_heads, q_input_channels, kv_input_channels, num_cross_attention_qk_channels, num_cross_attention_v_channels, cross_attention_widening_factor,
                       rpb=False,
                       feat_w=112,
                       feat_h=112):
            layer = CrossAttentionLayer(
                num_heads=num_cross_attention_heads,
                num_q_input_channels=q_input_channels,
                num_kv_input_channels=kv_input_channels,
                num_qk_channels=num_cross_attention_qk_channels,
                num_v_channels=num_cross_attention_v_channels,
                widening_factor=cross_attention_widening_factor,
                dropout=dropout,

                rpb=rpb,
                feat_w=feat_w,
                feat_h=feat_h,
            )
            return layer

        def self_attn(num_self_attention_layers_per_block, num_self_attention_heads, num_channels, num_self_attention_qk_channels, num_self_attention_v_channels, self_attention_widening_factor):
            return SelfAttentionBlock(
                num_layers=num_self_attention_layers_per_block,
                num_heads=num_self_attention_heads,
                num_channels=num_channels,
                num_qk_channels=num_self_attention_qk_channels,
                num_v_channels=num_self_attention_v_channels,
                widening_factor=self_attention_widening_factor,
                dropout=dropout,
                activation_checkpointing=activation_checkpointing,
            )
        

        # stage1
        x_channels_stage1in = cnn.layer1[0].conv1.in_channels
        x_channels_stage2in = cnn.layer2[0].conv1.in_channels
        x_channels_stage3in = cnn.layer3[0].conv1.in_channels
        x_channels_stage4in = cnn.layer4[0].conv1.in_channels
        x_channels_stage4out = cnn.layer4[-1].conv3.out_channels if isinstance(cnn.layer4[-1], Bottleneck) else cnn.layer4[-1].conv2.out_channels
        z_channels = [x_channels_stage1in, x_channels_stage2in, x_channels_stage3in, x_channels_stage4in]
        # print(z_channels)
        # assert(0==1)
        if num_latent_channels is None:
            num_latent_channels = x_channels_stage1in
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))

        self.with_x2z = with_x2z
        self.with_z2x = with_z2x
        self.with_dwc = with_dwc
        
        if with_dwc:
            self.dwc1_x2z = nn.Conv2d(in_channels=x_channels_stage1in, out_channels=x_channels_stage1in, kernel_size=7, 
                                  groups=x_channels_stage1in, stride=1, padding=3)
        feat_hw = 7 if spatial_reduction else input_size//2
        
        # essential
        self.cross_att1_x2z = cross_attn(num_cross_attention_heads=1,
                                        q_input_channels=x_channels_stage1in,
                                        kv_input_channels=x_channels_stage1in,                     
                                        num_cross_attention_qk_channels=None,       
                                        num_cross_attention_v_channels=None,        
                                        cross_attention_widening_factor=1,

                                        rpb=True,
                                        feat_w=feat_hw,
                                        feat_h=feat_hw,
        )
        self.self_att1 = self_attn(num_self_attention_layers_per_block=num_blocks_per_stage[0],                       
                                   num_self_attention_heads=num_SA_heads[0],                        
                                   num_channels=x_channels_stage1in,
                                   num_self_attention_qk_channels=None,                         
                                   num_self_attention_v_channels=None,                          
                                   self_attention_widening_factor=SA_widening_factor
        )
        
        # stage2
        if with_x2z:
            if with_dwc:
                self.dwc2_x2z = nn.Conv2d(in_channels=x_channels_stage2in, out_channels=x_channels_stage2in, kernel_size=7, groups=x_channels_stage2in, stride=1, padding=3)
            feat_hw = 7 if spatial_reduction else input_size//4
            self.cross_att2_x2z = cross_attn(num_cross_attention_heads=1,
                                        q_input_channels=x_channels_stage2in,
                                        kv_input_channels=x_channels_stage2in,                     
                                        num_cross_attention_qk_channels=None,       
                                        num_cross_attention_v_channels=None,                                      
                                        cross_attention_widening_factor=1,

                                        rpb=True,
                                        feat_w=feat_hw,
                                        feat_h=feat_hw,
            )

        if with_z2x:
            self.cross_att2_z2x = cross_attn(num_cross_attention_heads=1,
                                        q_input_channels=x_channels_stage2in,
                                        kv_input_channels=x_channels_stage2in,                     
                                        num_cross_attention_qk_channels=x_channels_stage2in//8,       
                                        num_cross_attention_v_channels=x_channels_stage2in//8,
                                        cross_attention_widening_factor=1
            )
        self.self_att2 = self_attn(num_self_attention_layers_per_block=num_blocks_per_stage[1], 
                                   num_self_attention_heads=num_SA_heads[1],                                  
                                   num_channels=x_channels_stage2in,
                                   num_self_attention_qk_channels=None,                         
                                   num_self_attention_v_channels=None,                          
                                   self_attention_widening_factor=SA_widening_factor
        )

        # stage3
        if with_x2z:
            if with_dwc:
                self.dwc3_x2z = nn.Conv2d(in_channels=x_channels_stage3in, out_channels=x_channels_stage3in, kernel_size=7, groups=x_channels_stage3in, stride=1, padding=3)
            feat_hw = 7 if spatial_reduction else input_size//8
            self.cross_att3_x2z = cross_attn(num_cross_attention_heads=1,
                                        q_input_channels=x_channels_stage3in,
                                        kv_input_channels=x_channels_stage3in,                     
                                        num_cross_attention_qk_channels=None,       
                                        num_cross_attention_v_channels=None,                                      
                                        cross_attention_widening_factor=1,

                                        rpb=True,
                                        feat_w=feat_hw,
                                        feat_h=feat_hw
            )

        if with_z2x:
            self.cross_att3_z2x = cross_attn(num_cross_attention_heads=1,
                                        q_input_channels=x_channels_stage3in,
                                        kv_input_channels=x_channels_stage3in,                     
                                        num_cross_attention_qk_channels=x_channels_stage3in//8,       
                                        num_cross_attention_v_channels=x_channels_stage3in//8,
                                        cross_attention_widening_factor=1
            )
        self.self_att3 = self_attn(num_self_attention_layers_per_block=num_blocks_per_stage[2],                       
                                   num_self_attention_heads=num_SA_heads[2],                                  
                                   num_channels=x_channels_stage3in,
                                   num_self_attention_qk_channels=None,                         
                                   num_self_attention_v_channels=None,                          
                                   self_attention_widening_factor=SA_widening_factor
        )

        # stage4
        if with_x2z:
            if with_dwc:
                self.dwc4_x2z = nn.Conv2d(in_channels=x_channels_stage4in, out_channels=x_channels_stage4in, kernel_size=7, groups=x_channels_stage4in, stride=1, padding=3)
            feat_hw = 7 if spatial_reduction else input_size//16
            self.cross_att4_x2z = cross_attn(num_cross_attention_heads=1,
                                        q_input_channels=x_channels_stage4in,
                                        kv_input_channels=x_channels_stage4in,                     
                                        num_cross_attention_qk_channels=None,       
                                        num_cross_attention_v_channels=None,                      
                                        cross_attention_widening_factor=1,

                                        rpb=True,
                                        feat_w=feat_hw,
                                        feat_h=feat_hw
            )

        if with_z2x:
            # print(x_channels_stage4in)
            self.cross_att4_z2x = cross_attn(num_cross_attention_heads=1,
                                        q_input_channels=x_channels_stage4in,
                                        kv_input_channels=x_channels_stage4in,                     
                                        num_cross_attention_qk_channels=x_channels_stage4in//8,       
                                        num_cross_attention_v_channels=x_channels_stage4in//8,                      
                                        cross_attention_widening_factor=1
            )
        # print(num_blocks_per_stage[3])
        self.self_att4 = self_attn(num_self_attention_layers_per_block=num_blocks_per_stage[3],                       
                                   num_self_attention_heads=num_SA_heads[3],                                  
                                   num_channels=x_channels_stage4in,
                                   num_self_attention_qk_channels=None,                         
                                   num_self_attention_v_channels=None,                          
                                   self_attention_widening_factor=SA_widening_factor
        )

        # last cross attention 
        # print(x_channels_stage4out//8)
        # print(x_channels_stage4out, x_channels_stage4in)
        self.last_cross_att_z2x = cross_attn(num_cross_attention_heads=1,
                                        q_input_channels=x_channels_stage4out,
                                        kv_input_channels=x_channels_stage4in,                     
                                        num_cross_attention_qk_channels=x_channels_stage4out//8,       
                                        num_cross_attention_v_channels=x_channels_stage4out//8,                                      
                                        cross_attention_widening_factor=1
        ) if with_last_CA else None

        # self.classifier_cnn = cnn.classifier
        # self.classifier_flops = cnn.classifier_flops
        # print(x_channels_stage1in, x_channels_stage2in, x_channels_stage3in, x_channels_stage4in)
        # self.early_classifier1 = nn.Linear(x_channels_stage1in, num_classes)
        # self.early_classifier2 = nn.Linear(x_channels_stage2in, num_classes)
        
        self.early_classifier3 = nn.Linear(x_channels_stage3in, num_classes)
        self.with_isc = with_isc
        
        if not with_isc:    
            self.classifier_att = nn.Linear(x_channels_stage4in, num_classes)            
            
            self.classifier_merge = nn.Sequential(
                nn.BatchNorm1d(cnn.num_features + x_channels_stage4in),
                nn.Linear(cnn.num_features + x_channels_stage4in, num_classes)
            )
        else:
            
            self.isc3 = nn.Sequential(nn.Linear(num_classes, x_channels_stage4in),
                                    nn.BatchNorm1d(x_channels_stage4in),
                                    nn.ReLU(inplace=True)
                                    )

            self.classifier_att = nn.Linear(2*x_channels_stage4in, num_classes)
            
            self.isc4 = nn.Sequential(nn.Linear(num_classes, x_channels_stage4in),
                                        nn.BatchNorm1d(x_channels_stage4in),
                                        nn.ReLU(inplace=True)
                                        )
            self.classifier_merge = nn.Sequential(
                nn.BatchNorm1d(cnn.num_features + 2*x_channels_stage4in),
                nn.Linear(cnn.num_features + 2*x_channels_stage4in, num_classes)
            )

        expander = []
        token_mixer = []

        num_latents_list = [num_latents, num_latents//2, num_latents//4, num_latents//8]
        for i in range(3):
            c_in = z_channels[i]
            c_out = z_channels[i+1]
            expander.append(nn.Sequential(
                nn.LayerNorm(c_in),
                nn.Linear(c_in, c_out)
            ))
            n_z_in = num_latents_list[i]
            n_z_out = num_latents_list[i+1]
            token_mixer.append(nn.Sequential(
                nn.LayerNorm(n_z_in),
                nn.Linear(n_z_in, n_z_out)
            ))

        self.token_expander = nn.ModuleList(expander)
        self.token_mixer = nn.ModuleList(token_mixer)
        self.output_dir = output_dir
        self._init_parameters()
        

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None):
        exit = self.exit
        #TODO
        b, c_in, _, _ = x.shape
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        x = self.cnn_stem(x)
        # print(x.shape)
        
        if self.with_dwc:
            x_kv = self.dwc1_x2z(x) + x
            x_kv = self.ca_pooling(x_kv)
        else:
            x_kv = self.ca_pooling(x)
        x_kv = rearrange(x_kv, "b c ... -> b (...) c")
        # print(x_latent.shape, x_kv.shape)
        x_latent = self.cross_att1_x2z(x_latent, x_kv, pad_mask)

        # stage1, conv and self attention
        x_latent = self.self_att1(x_latent)

        # y_early1 = torch.mean(x_latent, dim=1).squeeze(1)
        # y_early1 = self.early_classifier1(y_early1)

        
        x = self.cnn_body_stage1(x)

        # between stage1 and stage2
        _, n_tokens, c_in = x_latent.shape
        x_latent = x_latent.permute(0,2,1)
        x_latent = self.token_mixer[0](x_latent)
        x_latent = x_latent.permute(0,2,1)
        x_latent = self.token_expander[0](x_latent)
        
        # transformer to conv
        if self.with_z2x:
            _,_,h,w = x.shape
            x = rearrange(x, "b c ... -> b (...) c")
            x = self.cross_att2_z2x(x, x_latent, pad_mask)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        # conv to transformer
        # print(x.shape)
        if self.with_x2z:
            if self.with_dwc:
                x_kv = self.dwc2_x2z(x) + x
                x_kv = self.ca_pooling(x_kv)
            else:
                x_kv = self.ca_pooling(x)
            x_kv = rearrange(x_kv, "b c ... -> b (...) c")
            x_latent = self.cross_att2_x2z(x_latent, x_kv, pad_mask)
        
        # stage2
        x_latent = self.self_att2(x_latent)
        # y_early2 = torch.mean(x_latent, dim=1).squeeze(1)
        # y_early2 = self.early_classifier2(y_early2)

        x = self.cnn_body_stage2(x)

        # between stage2 and stage3
        _, n_tokens, c_in = x_latent.shape
        x_latent = x_latent.permute(0,2,1)
        x_latent = self.token_mixer[1](x_latent)
        x_latent = x_latent.permute(0,2,1)
        x_latent = self.token_expander[1](x_latent)
        
        # transformer to conv
        if self.with_z2x:
            _,_,h,w = x.shape
            x = rearrange(x, "b c ... -> b (...) c")
            x = self.cross_att3_z2x(x, x_latent, pad_mask)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        # conv to transformer
        # print(x.shape)
        if self.with_x2z:
            if self.with_dwc:
                x_kv = self.dwc3_x2z(x) + x
                x_kv = self.ca_pooling(x_kv)
            else:
                x_kv = self.ca_pooling(x)
            x_kv = rearrange(x_kv, "b c ... -> b (...) c")
            x_latent = self.cross_att3_x2z(x_latent, x_kv, pad_mask)

        # stage3
        x_latent = self.self_att3(x_latent)
        y_early3 = torch.mean(x_latent, dim=1).squeeze(1)
        y_early3 = self.early_classifier3(y_early3)
        if exit == 0:
            return y_early3

        x = self.cnn_body_stage3(x)

        # between stage3 and stage4
        _, n_tokens, c_in = x_latent.shape
        x_latent = x_latent.permute(0,2,1)
        x_latent = self.token_mixer[2](x_latent)
        x_latent = x_latent.permute(0,2,1)
        x_latent = self.token_expander[2](x_latent)

        # transformer to conv
        if self.with_z2x:
            _,_,h,w = x.shape
            x = rearrange(x, "b c ... -> b (...) c")
            # print(x_latent.shape, x.shape)
            x = self.cross_att4_z2x(x, x_latent, pad_mask)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


        # conv to transformer
        # print(x.shape)
        if self.with_x2z:
            
            if self.with_dwc:
                x_kv = self.dwc4_x2z(x) + x
                x_kv = self.ca_pooling(x_kv)
            else:
                x_kv = self.ca_pooling(x)
            x_kv = rearrange(x_kv, "b c ... -> b (...) c")
            x_latent = self.cross_att4_x2z(x_latent, x_kv, pad_mask)

        # stage4
        x_latent = self.self_att4(x_latent)

        x_latent_mean = torch.mean(x_latent, dim=1).squeeze(1)
        if self.with_isc:
            y3_ = self.isc3(y_early3)
            y_att = torch.cat((x_latent_mean, y3_), dim=1)
            y_att = self.classifier_att(y_att)
        else:
            y_att = self.classifier_att(x_latent_mean)
        if exit == 1:
            return y_att

        x = self.cnn_body_stage4(x)
        # print(x.shape)
        
        # x = self.cnn_body_last_conv1x1(x)
        x_mean = self.avgpool(x)
        # x_mean = self.cnn_head_before_cls(x_mean)
        # x_mean = self.flatten_cnn(x_mean)
        
        if self.drop_rate_cnn > 0.:
            x_mean = F.dropout(x_mean, p=self.drop_rate_cnn, training=self.training)
        
        # print(x_mean.shape)
        y_cnn = self.classifier_cnn(x_mean)
        if exit == 2:
            return y_cnn

        if self.last_cross_att_z2x is not None:
            _,_,h,w = x.shape
            x = rearrange(x, "b c ... -> b (...) c")
            x = self.last_cross_att_z2x(x, x_latent, pad_mask)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            x_mean = self.avgpool(x)
            # x_mean = self.cnn_head_before_cls(x_mean)
            # x_mean = self.flatten_cnn(x_mean)

        if self.with_isc:
            y4_ = self.isc4(y_att)
            x_merge = torch.cat((x_mean, x_latent_mean, y4_), dim=1)
            y_merge = self.classifier_merge(x_merge)
        else:
            x_merge = torch.cat((x_mean, x_latent_mean), dim=1)
            y_merge = self.classifier_merge(x_merge)

        return y_early3, y_att, y_cnn, y_merge


@register_model
def resnet18_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='resnet18', **kwargs)
    return model


@register_model
def resnet18_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='resnet18', **kwargs)
    return model


@register_model
def resnet18_perceiver_t512(**kwargs):
    model = DynPerceiver(num_latents=512, cnn_arch='resnet18', **kwargs)
    return model


@register_model
def resnet34_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='resnet34', **kwargs)
    return model


@register_model
def resnet34_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='resnet34', **kwargs)
    return model


@register_model
def resnet34_perceiver_t512(**kwargs):
    model = DynPerceiver(num_latents=512, cnn_arch='resnet34', **kwargs)
    return model


@register_model
def resnet50_perceiver_t64(**kwargs):
    model = DynPerceiver(num_latents=64, cnn_arch='resnet50', **kwargs)
    return model


@register_model
def resnet50_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='resnet50', **kwargs)
    return model


@register_model
def resnet50_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='resnet50', **kwargs)
    return model


@register_model
def resnet50_perceiver_t512(**kwargs):
    model = DynPerceiver(num_latents=512, cnn_arch='resnet50', **kwargs)
    return model


@register_model
def resnet50_050_perceiver_t64(**kwargs):
    model = DynPerceiver(num_latents=64, cnn_arch='resnet50_050', **kwargs)
    return model


@register_model
def resnet50_050_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='resnet50_050', **kwargs)
    return model


@register_model
def resnet50_050_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='resnet50_050', **kwargs)
    return model


@register_model
def resnet50_0375_perceiver_t64(**kwargs):
    model = DynPerceiver(num_latents=64, cnn_arch='resnet50_0375', **kwargs)
    return model


@register_model
def resnet50_0375_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='resnet50_0375', **kwargs)
    return model


@register_model
def resnet50_0375_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='resnet50_0375', **kwargs)
    return model


@register_model
def resnet50_0625_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='resnet50_0625', **kwargs)
    return model


@register_model
def resnet50_0625_perceiver_t160(**kwargs):
    model = DynPerceiver(num_latents=160, cnn_arch='resnet50_0625', **kwargs)
    return model


@register_model
def resnet50_0625_perceiver_t192(**kwargs):
    model = DynPerceiver(num_latents=192, cnn_arch='resnet50_0625', **kwargs)
    return model


@register_model
def resnet50_0625_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='resnet50_0625', **kwargs)
    return model


@register_model
def resnet50_075_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='resnet50_075', **kwargs)
    return model


@register_model
def resnet50_075_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='resnet50_075', **kwargs)
    return model


def compute_flops(model):
    x = torch.rand(1, 3, 224, 224)
    result = []
    for i in range(4,5):
        # model = resnet18_perceiver_t128(
        #     depth_factor=[1,1,1,1], SA_widening_factor=4, spatial_reduction=True,
        #     with_last_CA=True, with_x2z=True, with_dwc=True, with_z2x=True, exit=i)
        model.eval()
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x)
        result.append(flops.total() / 1e9)
    print('***************************')
    for flop in result:
        print(flop)
    print('***************************')
    model.train()


if __name__ == '__main__':

    # dyn_flops()

    # # Fourier-encodes pixel positions and flatten along spatial dimensions
    # input_adapter = ImageInputAdapter(
    # image_shape=(224, 224, 3),  # M = 224 * 224
    # num_frequency_bands=64,
    # )

    # # Projects generic Perceiver decoder output to specified number of classes
    # output_adapter = ClassificationOutputAdapter(
    # num_classes=1000,
    # num_output_query_channels=1024,  # F
    # )

    # # Generic Perceiver encoder
    # encoder = PerceiverEncoder(
    # input_adapter=input_adapter,
    # num_latents=512,  # N
    # num_latent_channels=1024,  # D
    # num_cross_attention_qk_channels=input_adapter.num_input_channels,  # C
    # num_cross_attention_heads=1,
    # num_self_attention_heads=4,
    # num_self_attention_layers_per_block=6,
    # num_self_attention_blocks=8,
    # dropout=0.0,
    # )

    # # Generic Perceiver decoder
    # decoder = PerceiverDecoder(
    # output_adapter=output_adapter,
    # num_latent_channels=1024,  # D
    # num_cross_attention_heads=1,
    # dropout=0.0,
    # )

    # # Perceiver IO image classifier
    # model = PerceiverIO(encoder, decoder)
    # model.eval()
    # print(model)
    # x = torch.rand(4,224,224,3)
    # with torch.no_grad():
    #     y = model(x)

    #     print(y.shape)





    # regnet = regnet_y_400mf()
    # print(regnet)
    # print(regnet.trunk_output.block1)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = resnet50_perceiver_t128(depth_factor=[1,1,1,3], SA_widening_factor=4, spatial_reduction=True, with_last_CA=True,
                                      with_x2z=True, with_dwc=True, with_z2x=True)
    model.eval()
    print(count_parameters(model)/1e6)
    x = torch.rand(1,3,224,224)
    with torch.no_grad():
        y = model(x)
        # print(y.shape)
        # print()
        # flops = FlopCountAnalysis(model, x)
        # print("FLOPs: ", flops.total()/1e9)
        # from fvcore.nn import flop_count_str
        # print(flop_count_str(flops))
        # # 分析parameters
        # print(parameter_count_table(model))