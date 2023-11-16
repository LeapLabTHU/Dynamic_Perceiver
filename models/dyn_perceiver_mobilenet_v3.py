import math

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from einops import rearrange, repeat
from torch import Tensor
import torch.nn as nn
# from fairscale.nn import checkpoint_wrapper
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
from timm.models.registry import register_model
from .perceiver_core import (
    CrossAttentionLayer,
    SelfAttentionBlock,
)

from .cnn_core import *
import numpy as np

class DynPerceiver(nn.Module):
    def __init__(self,
                input_size: int=224,
                num_classes:int=100,
                cnn_arch: str="mobilenet_v3_large",
                num_SA_heads: list=[1,2,4,8],
                num_latents: int=32,
                num_latent_channels: int=None,
                dropout: float = 0.0,
                SA_widening_factor: int=1,
                activation_checkpointing: bool = False,
                spatial_reduction: bool=False,
                depth_factor: list=[1,1,1,1],
                output_dir: str='./',
                with_x2z=True,
                with_z2x=True,
                with_dwc=True,
                with_last_CA=True,
                with_isc=True):
        super().__init__()
        if num_SA_heads is None:
            num_SA_heads = [1,2,4,8]
            
        cnn = eval(f'{cnn_arch}')()
        # self.cnn_stem = cnn.stem
        # self.cnn_body = cnn.trunk_output
        stem = [cnn.features[i] for i in range(0, 3)]
        self.cnn_stem = nn.ModuleList(stem)
        stage1 = [cnn.features[i] for i in range(3, 5)]
        self.cnn_body_stage1 = nn.ModuleList(stage1)
        stage2 = [cnn.features[i] for i in range(5, 8)]
        self.cnn_body_stage2 = nn.ModuleList(stage2)

        stage3 = [cnn.features[i] for i in range(8, 14)]
        self.cnn_body_stage3 = nn.ModuleList(stage3)
        stage4 = [cnn.features[i] for i in range(14, 17)]
        self.cnn_body_stage4 = nn.ModuleList(stage4)
        # num_blocks_per_stage = [len(self.cnn_body.block1)*depth_factor[0], len(self.cnn_body.block2)*depth_factor[1], 
        #                         len(self.cnn_body.block3)*depth_factor[2], len(self.cnn_body.block4)*depth_factor[3]]
        num_blocks_per_stage = [3*depth_factor[0], 3*depth_factor[1], 9*depth_factor[2], 3*depth_factor[3]]
        self.avgpool = cnn.avgpool
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
        x_channels_stage1in = self.cnn_body_stage1[0].c_in
        x_channels_stage2in = self.cnn_body_stage2[0].c_in
        x_channels_stage3in = self.cnn_body_stage3[0].c_in
        x_channels_stage4in = self.cnn_body_stage4[0].c_in
        x_channels_stage4out = cnn.lastconv_output_channels
        z_channels = [x_channels_stage1in, x_channels_stage2in, x_channels_stage3in, x_channels_stage4in]
        # print(z_channels)
        # assert(0==1)
        if num_latent_channels is None:
            num_latent_channels = x_channels_stage1in
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        # there are 2 cross attentions from x to z and from z to x.
        self.with_x2z = with_x2z # pass info from x (feature branch) to the classification branch (z)
        self.with_z2x = with_z2x # the other way
        self.with_dwc = with_dwc # depth wise convolution to enhance local feature extraction.
        
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
                # Feed the input of stage 2 convolution
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

        if with_z2x: # cross attention at stage 2
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
                # feed the the convolutions
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
            print(x_channels_stage4in)
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
        print(x_channels_stage4out, x_channels_stage4in)
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

        # CLASSIFIERS
        self.early_classifier3 = nn.Linear(x_channels_stage3in, num_classes) # early classifier at branch 3
        # takes feature from third stage.
        self.with_isc = with_isc
        
        if not with_isc:    
            self.classifier_att = nn.Linear(x_channels_stage4in, num_classes) # at branch 4, after attention
            
            self.classifier_cnn = nn.Sequential(
                                    nn.Linear(cnn.lastconv_output_channels, cnn.last_channel),
                                    nn.Hardswish(inplace=True),
                                    nn.Dropout(p=0.2, inplace=True),
                                    nn.Linear(cnn.last_channel, num_classes),
                                )
            self.classifier_cnn_flops = cnn.lastconv_output_channels*cnn.last_channel + cnn.last_channel*num_classes
            
            
            self.classifier_merge = nn.Sequential(nn.BatchNorm1d(cnn.lastconv_output_channels+x_channels_stage4in),
                                            nn.Linear(cnn.lastconv_output_channels+x_channels_stage4in, cnn.last_channel),
                                            nn.Hardswish(inplace=True),
                                            nn.Dropout(p=0.2, inplace=True),
                                            nn.Linear(cnn.last_channel, num_classes),
            )
            self.classifier_merge_flops = (cnn.lastconv_output_channels+x_channels_stage4in) * cnn.last_channel + cnn.last_channel * num_classes
        else: # with_isc
            
            self.isc3 = nn.Sequential(nn.Linear(num_classes, x_channels_stage4in),
                                    nn.BatchNorm1d(x_channels_stage4in),
                                    nn.ReLU(inplace=True)
                                    )

            self.classifier_att = nn.Linear(2*x_channels_stage4in, num_classes)
            self.isc4 = nn.Sequential(nn.Linear(num_classes, x_channels_stage4in),
                                        nn.BatchNorm1d(x_channels_stage4in),
                                        nn.ReLU(inplace=True)
                                        )
            cnn_channels = cnn.lastconv_output_channels
            self.classifier_cnn = nn.Sequential(
                                    nn.Linear(cnn.lastconv_output_channels, cnn.last_channel),
                                    nn.Hardswish(inplace=True),
                                    nn.Dropout(p=0.2, inplace=True),
                                    nn.Linear(cnn.last_channel, num_classes),
                                )
            self.classifier_cnn_flops = cnn.lastconv_output_channels*cnn.last_channel + cnn.last_channel*num_classes
            self.classifier_merge = nn.Sequential(nn.BatchNorm1d(cnn_channels+2*x_channels_stage4in),
                                        nn.Linear(cnn_channels+2*x_channels_stage4in, cnn.last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(cnn.last_channel, num_classes),
            )
            self.classifier_merge_flops = (cnn_channels+2*x_channels_stage4in) * cnn.last_channel + cnn.last_channel * num_classes

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

            # linear_layer = nn.Linear(c_in, 2, bias=True)
            # linear_layer.bias.data[0] = 0.0
            # linear_layer.bias.data[1] = 0.0
            # token_mixer.append(nn.Sequential(
            #     nn.LayerNorm(c_in),
            #     linear_layer
            # ))
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
        x = torch.rand(2,3,224,224)
        self.forward_calc_flops(x)
        

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None):
        #TODO
        b, c_in, _, _ = x.shape
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        x = self.cnn_stem[0](x)
        # cnn_flops = c_in * x.shape[1] * x.shape[2] * x.shape[3] * 9

        for j in range(1, len(self.cnn_stem)):
            x = self.cnn_stem[j](x)
        # before stage1
        # conv to transformer
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

        for j in range(len(self.cnn_body_stage1)):
            x = self.cnn_body_stage1[j](x)

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

        for j in range(len(self.cnn_body_stage2)):
            x = self.cnn_body_stage2[j](x)

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
        y_early3 = self.early_classifier3(y_early3) # EXIT 3

        for j in range(len(self.cnn_body_stage3)):
            x = self.cnn_body_stage3[j](x)

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
            y_att = self.classifier_att(y_att) # EXIT ATT
        else:
            y_att = self.classifier_att(x_latent_mean)

        for j in range(len(self.cnn_body_stage4)):
            x = self.cnn_body_stage4[j](x)
        

        x_mean = self.avgpool(x)
        x_mean = x_mean.flatten(start_dim=1)
        y_cnn = self.classifier_cnn(x_mean) # EXIT CNN


        # cross attention from z to x
        if self.last_cross_att_z2x is not None:
            _,_,h,w = x.shape
            x = rearrange(x, "b c ... -> b (...) c")
            x = self.last_cross_att_z2x(x, x_latent, pad_mask)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            x_mean = self.avgpool(x)
            x_mean = x_mean.flatten(start_dim=1)

        if self.with_isc:
            y4_ = self.isc4(y_att)
            x_merge = torch.cat((x_mean, x_latent_mean, y4_), dim=1)
            y_merge = self.classifier_merge(x_merge)
        else:
            x_merge = torch.cat((x_mean, x_latent_mean), dim=1)
            y_merge = self.classifier_merge(x_merge)

        return y_early3, y_att, y_cnn, y_merge


    def forward_calc_flops(self, x, pad_mask=None):
        #TODO
        b, c_in, _, _ = x.shape
        x_latent = repeat(self.latent, "... -> b ...", b=b)
        
        # cnn_flops = 0
        att_flops = 0

        x = self.cnn_stem[0](x)
        cnn_flops = c_in * x.shape[1] * x.shape[2] * x.shape[3] * 9

        for j in range(1, len(self.cnn_stem)):
            x, cnn_flops = self.cnn_stem[j].forward_calc_flops((x, cnn_flops))
        # before stage1
        # conv to transformer
        if self.with_dwc:
            x_kv = self.dwc1_x2z(x) + x
            att_flops += x_kv.shape[1] * x_kv.shape[2] * x_kv.shape[3] * 49
            x_kv = self.ca_pooling(x_kv)
        else:
            x_kv = self.ca_pooling(x)
        att_flops += x.shape[1] * x.shape[2] * x.shape[3] # pooling flops
        
        x_kv = rearrange(x_kv, "b c ... -> b (...) c")
        x_latent, CA1x2z_flops = self.cross_att1_x2z.forward_calc_flops(x_latent, x_kv, pad_mask)
        att_flops += CA1x2z_flops

        # stage1, conv and self attention
        x_latent, SA1_flops = self.self_att1.forward_calc_flops(x_latent)
        att_flops += SA1_flops

        # c_in = x_latent.shape[-1]
        # y_early1 = torch.mean(x_latent, dim=1).squeeze(1)
        # y_early1 = self.early_classifier1(y_early1)
        # flops_early1 = cnn_flops + att_flops + c_in*y_early1.shape[-1]

        for j in range(len(self.cnn_body_stage1)):
            x, cnn_flops = self.cnn_body_stage1[j].forward_calc_flops((x, cnn_flops))
        # assert(0==1)

        # between stage1 and stage2
        _, n_tokens, c_in = x_latent.shape
        x_latent = x_latent.permute(0,2,1)
        x_latent = self.token_mixer[0](x_latent)
        x_latent = x_latent.permute(0,2,1)
        att_flops += c_in * n_tokens * x_latent.shape[1]

        x_latent = self.token_expander[0](x_latent)
        att_flops += n_tokens * c_in * x_latent.shape[-1]
        # transformer to conv
        
        CA2z2x_flops = 0
        if self.with_z2x:
            _,_,h,w = x.shape
            x = rearrange(x, "b c ... -> b (...) c")
            x, CA2z2x_flops = self.cross_att2_z2x.forward_calc_flops(x, x_latent, pad_mask)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            att_flops += CA2z2x_flops
        
        
        # conv to transformer
        CA2_x2z_flops = 0
        if self.with_x2z:
            if self.with_dwc:
                x_kv = self.dwc2_x2z(x) + x
                att_flops += x_kv.shape[1] * x_kv.shape[2] * x_kv.shape[3] * 49
                x_kv = self.ca_pooling(x_kv)
            else:
                x_kv = self.ca_pooling(x)
            
            att_flops += x.shape[1] * x.shape[2] * x.shape[3]
            x_kv = rearrange(x_kv, "b c ... -> b (...) c")
            x_latent, CA2_x2z_flops = self.cross_att2_x2z.forward_calc_flops(x_latent, x_kv, pad_mask)
            att_flops += CA2_x2z_flops
        
        # stage2
        x_latent, SA2_flops = self.self_att2.forward_calc_flops(x_latent)
        att_flops += SA2_flops
        c_in = x_latent.shape[-1]
        # y_early2 = torch.mean(x_latent, dim=1).squeeze(1)
        # y_early2 = self.early_classifier2(y_early2)
        # flops_early2 = cnn_flops + att_flops + c_in*y_early2.shape[-1]

        for j in range(len(self.cnn_body_stage2)):
            x, cnn_flops = self.cnn_body_stage2[j].forward_calc_flops((x, cnn_flops))

        # between stage2 and stage3
        _, n_tokens, c_in = x_latent.shape
        x_latent = x_latent.permute(0,2,1)
        x_latent = self.token_mixer[1](x_latent)
        x_latent = x_latent.permute(0,2,1)
        att_flops += c_in * n_tokens * x_latent.shape[1]
        x_latent = self.token_expander[1](x_latent)
        att_flops += n_tokens * c_in * x_latent.shape[-1]
        
        # transformer to conv
        CA3z2x_flops = 0
        if self.with_z2x:
            _,_,h,w = x.shape
            x = rearrange(x, "b c ... -> b (...) c")
            x, CA3z2x_flops = self.cross_att3_z2x.forward_calc_flops(x, x_latent, pad_mask)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            att_flops += CA3z2x_flops
        
        
        # conv to transformer
        CA3x2z_flops = 0
        if self.with_x2z:
            if self.with_dwc:
                x_kv = self.dwc3_x2z(x) + x
                att_flops += x_kv.shape[1] * x_kv.shape[2] * x_kv.shape[3] * 49
                x_kv = self.ca_pooling(x_kv)
            else:
                x_kv = self.ca_pooling(x)
            
            att_flops += x.shape[1] * x.shape[2] * x.shape[3]
            x_kv = rearrange(x_kv, "b c ... -> b (...) c")
            # print(x_latent.shape, x_kv.shape)
            x_latent, CA3x2z_flops = self.cross_att3_x2z.forward_calc_flops(x_latent, x_kv, pad_mask)
            att_flops += CA3x2z_flops

        # stage3
        x_latent, SA3_flops = self.self_att3.forward_calc_flops(x_latent)
        att_flops += SA3_flops
        c_in = x_latent.shape[-1]
        y_early3 = torch.mean(x_latent, dim=1).squeeze(1)
        y_early3 = self.early_classifier3(y_early3)
        flops_early3 = cnn_flops + att_flops + c_in*y_early3.shape[-1]

        for j in range(len(self.cnn_body_stage3)):
            x, cnn_flops = self.cnn_body_stage3[j].forward_calc_flops((x, cnn_flops))

        # between stage3 and stage4
        _, n_tokens, c_in = x_latent.shape
        x_latent = x_latent.permute(0,2,1)
        x_latent = self.token_mixer[2](x_latent)
        x_latent = x_latent.permute(0,2,1)
        att_flops += c_in * n_tokens * x_latent.shape[1]
        x_latent = self.token_expander[2](x_latent)
        att_flops += n_tokens * c_in * x_latent.shape[-1]
        
        
        # transformer to conv
        CA4z2x_flops = 0
        if self.with_z2x:
            _,_,h,w = x.shape
            x = rearrange(x, "b c ... -> b (...) c")
            x, CA4z2x_flops = self.cross_att4_z2x.forward_calc_flops(x, x_latent, pad_mask)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            att_flops += CA4z2x_flops

        # conv to transformer 
        CA4x2z_flops = 0       
        if self.with_x2z:
            if self.with_dwc:
                x_kv = self.dwc4_x2z(x) + x
                att_flops += x_kv.shape[1] * x_kv.shape[2] * x_kv.shape[3] * 49
                x_kv = self.ca_pooling(x_kv)
            else:
                x_kv = self.ca_pooling(x)
            att_flops += x.shape[1] * x.shape[2] * x.shape[3]
            x_kv = rearrange(x_kv, "b c ... -> b (...) c")
            # print(x_latent.shape, x_kv.shape)
            x_latent, CA4x2z_flops = self.cross_att4_x2z.forward_calc_flops(x_latent, x_kv, pad_mask)
            att_flops += CA4x2z_flops

        # stage4
        x_latent, SA4_flops = self.self_att4.forward_calc_flops(x_latent)
        att_flops += SA4_flops

        att_flops += x_latent.shape[1] * x_latent.shape[2]
        x_latent_mean = torch.mean(x_latent, dim=1).squeeze(1)
        if self.with_isc:
            c_in_ = y_early3.shape[-1]
            y3_ = self.isc3(y_early3)
            c_out_ = y3_.shape[-1]
            y_att = torch.cat((x_latent_mean, y3_), dim=1)
            c_in_att = y_att.shape[1]
            y_att = self.classifier_att(y_att)
            flops_early4 = cnn_flops + att_flops + c_in_att*y_att.shape[1] + c_in_ * c_out_
        else:
            c_in_att = x_latent_mean.shape[1]
            y_att = self.classifier_att(x_latent_mean)
            flops_early4 = cnn_flops + att_flops + c_in_att*y_att.shape[1]


        for j in range(len(self.cnn_body_stage4)-1):
            x, cnn_flops = self.cnn_body_stage4[j].forward_calc_flops((x, cnn_flops))
        
        c_in = x.shape[1]
        x = self.cnn_body_stage4[-1](x)
        cnn_flops += c_in * x.shape[1] * x.shape[2] * x.shape[3]

        cnn_flops += x.shape[1]*x.shape[2]*x.shape[3]
        x_mean = self.avgpool(x)
        x_mean = x_mean.flatten(start_dim=1)
        c_in_cnn = x_mean.shape[1]
        y_cnn = self.classifier_cnn(x_mean)
        cnn_flops += self.classifier_cnn_flops
        flops_early5 = cnn_flops + att_flops
        
        # lastCA_flops = 0
        if self.last_cross_att_z2x is not None:
            # cross attention from z to x
            _,_,h,w = x.shape
            x = rearrange(x, "b c ... -> b (...) c")
            
            # print(x_latent.shape, x.shape)
            x, lastCA_flops = self.last_cross_att_z2x.forward_calc_flops(x, x_latent, pad_mask)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            # x = x + x_q
            att_flops += lastCA_flops

            cnn_flops += x.shape[1]*x.shape[2]*x.shape[3]
            x_mean = self.avgpool(x)
            x_mean = x_mean.flatten(start_dim=1)
        
        if self.with_isc:
            c_in_ = y_att.shape[-1]
            y4_ = self.isc4(y_att)
            c_out_ = y4_.shape[-1]
            x_merge = torch.cat((x_mean, x_latent_mean, y4_), dim=1)
        else:
            x_merge = torch.cat((x_mean, x_latent_mean), dim=1)
        c_in = x_merge.shape[-1]
        y_merge = self.classifier_merge(x_merge)
        flops = att_flops + cnn_flops + self.classifier_merge_flops

        # print(f'total flops: {flops/1e9}\n\
        #     att_flops: {att_flops/1e9}\n\
        #     cnn_flops: {cnn_flops/1e9}\n\n\
        #     CA1x2z_flops: {CA1x2z_flops/1e8}\n\
        #     SA1_flops: {SA1_flops/1e8}\n\
        #     CA2z2x_flops: {CA2z2x_flops/1e8}\n\
        #     CA2_x2z_flops: {CA2_x2z_flops/1e8}\n\
        #     SA2_flops: {SA2_flops/1e8}\n\
        #     CA3z2x_flops: {CA3z2x_flops/1e8}\n\
        #     CA3x2z_flops: {CA3x2z_flops/1e8}\n\
        #     SA3_flops: {SA3_flops/1e8}\n\
        #     EE3_flops: {flops_early3/1e9}\n\n\
        #     CA4z2x_flops: {CA4z2x_flops/1e8}\n\
        #     CA4x2z_flops: {CA4x2z_flops/1e8}\n\
        #     SA4_flops: {SA4_flops/1e8}\n\
        #     EE4_flops: {flops_early4/1e9}\n\
        #     lastCA_flops: {lastCA_flops/1e8}')
        
        all_flops = [flops_early3/1e9, flops_early4/1e9, flops_early5/1e9, flops/1e9]
        print(all_flops)
        mul_adds = (torch.tensor(all_flops) / 2).tolist()
        print('MUL ADDS FOR DYN PERCEIVER MOBILENET')
        print(mul_adds)
        print('------------------------------------')
        np.savetxt(f'{self.output_dir}/flops.txt', all_flops)
        np.savetxt(f'{self.output_dir}/muladds.txt', mul_adds)
        return y_early3, y_att, y_cnn, y_merge


@register_model
def mobilenetV3_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='mobilenet_v3_large', **kwargs)
    return model


@register_model
def mobilenetV3_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='mobilenet_v3_large', **kwargs)
    return model


@register_model
def mobilenetV3_perceiver_t512(**kwargs):
    model = DynPerceiver(num_latents=512, cnn_arch='mobilenet_v3_large', **kwargs)
    return model


@register_model
def mobilenetV3_0x5_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='mobilenet_v3_large_0x5', **kwargs)
    return model


@register_model
def mobilenetV3_0x5_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='mobilenet_v3_large_0x5', **kwargs)
    return model


@register_model
def mobilenetV3_0x5_perceiver_t512(**kwargs):
    model = DynPerceiver(num_latents=512, cnn_arch='mobilenet_v3_large_0x5', **kwargs)
    return model


@register_model
def mobilenetV3_0x75_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='mobilenet_v3_large_0x75', **kwargs)
    return model


@register_model
def mobilenetV3_0x75_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='mobilenet_v3_large_0x75', **kwargs)
    return model


@register_model
def mobilenetV3_0x75_perceiver_t512(**kwargs):
    model = DynPerceiver(num_latents=512, cnn_arch='mobilenet_v3_large_0x75', **kwargs)
    return model


@register_model
def mobilenetV3_1x25_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='mobilenet_v3_large_1x25', **kwargs)
    return model


@register_model
def mobilenetV3_1x25_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='mobilenet_v3_large_1x25', **kwargs)
    return model


@register_model
def mobilenetV3_1x25_perceiver_t512(**kwargs):
    model = DynPerceiver(num_latents=512, cnn_arch='mobilenet_v3_large_1x25', **kwargs)
    return model


@register_model
def mobilenetV3_1x5_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='mobilenet_v3_large_1x5', **kwargs)
    return model


@register_model
def mobilenetV3_1x5_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='mobilenet_v3_large_1x5', **kwargs)
    return model


@register_model
def mobilenetV3_1x5_perceiver_t512(**kwargs):
    model = DynPerceiver(num_latents=512, cnn_arch='mobilenet_v3_large_1x5', **kwargs)
    return model


@register_model
def mobilenetV3_2x0_perceiver_t128(**kwargs):
    model = DynPerceiver(num_latents=128, cnn_arch='mobilenet_v3_large_2x0', **kwargs)
    return model


@register_model
def mobilenetV3_2x0_perceiver_t256(**kwargs):
    model = DynPerceiver(num_latents=256, cnn_arch='mobilenet_v3_large_2x0', **kwargs)
    return model


@register_model
def mobilenetV3_2x0_perceiver_t512(**kwargs):
    model = DynPerceiver(num_latents=512, cnn_arch='mobilenet_v3_large_2x0', **kwargs)
    return model


if __name__ == '__main__':

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

    model = mobilenetV3_0x75_perceiver_t128(depth_factor=[1,1,1,3], SA_widening_factor=4, spatial_reduction=True, with_last_CA=True,
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