
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from diffusers.models.attention_processor import Attention

from .ConsiStoryAttnProcessor import ConsiStoryAttnProcessor

class DreamStoryAttnProcessor(ConsiStoryAttnProcessor):
    def __init__(self, start_step=4, start_layer=54, self_attn_layer_idx_list=None, step_idx_list=None, total_steps=50, 
                thres=0, ref_token_idx=[1], cur_token_idx=[1], model_type="SDXL",
                cross_attn_gather_down_layer=2, latents_shape=(768//8, 1280//8),
                dropout=0, is_output_mask=False, sam_mask=None, sam_step=40, 
                is_spatial_self_attn=True, is_mutual_cross_attn=True,
                is_dropout_ca=False, mutual_cross_attention_lambda=0.9, 
                is_DIFT=False,
                **kwargs):
        super().__init__(start_step=start_step, start_layer=start_layer, self_attn_layer_idx_list=self_attn_layer_idx_list, 
                        step_idx_list=step_idx_list, total_steps=total_steps, is_output_mask=is_output_mask,
                        thres=thres, ref_token_idx=ref_token_idx, cur_token_idx=cur_token_idx,
                        model_type=model_type, cross_attn_gather_down_layer=cross_attn_gather_down_layer, latents_shape=latents_shape,
                        dropout=dropout, is_DIFT=is_DIFT,)
        
        self.exponent_N = 4
        
        self.is_mutual_cross_attn = is_mutual_cross_attn
        self.is_spatial_self_attn = is_spatial_self_attn

        self.sam_mask = sam_mask
        self.sam_step = sam_step

        self.is_dropout_ca = is_dropout_ca
        self.mutual_cross_attention_lambda = mutual_cross_attention_lambda

        self.self_attns_mask_dict = {}
        self.spatial_mask_dict = {}

        if len(kwargs) > 0:
            print(f"Warning: get unexpected parameters: {kwargs}")
        
        # if ref_token_idx and cur_token_idx is the list of list, it would be multi-subject generation
        if isinstance(ref_token_idx[0], list):
            self.subj_num = len(ref_token_idx)
            assert len(ref_token_idx) == len(cur_token_idx), f"The length of ref_token_idx and cur_token_idx should be the same. \
                ref_token_idx={ref_token_idx}, cur_token_idx = {cur_token_idx}"
        else:
            self.subj_num = 1

        self.self_attns = []
        self.self_attns_1 = []
        self.cross_attns = []
        self.cross_attns_1 = []
        self.self_cross_attns = []
        self.self_cross_attns_1 = []
        self.last_self_cross_attns = []

    def after_step(self):
        self.cur_att_layer += 1
        if self.cur_att_layer == self.total_attn_layers:
            self.cur_att_layer = 0  
            self.cur_step += 1
            self.last_self_attns = self.self_attns
            self.last_self_attns_1 = self.self_attns_1
            self.last_cross_attns = self.cross_attns
            self.last_cross_attns_1 = self.cross_attns_1

            # firtly, sum the sa and ca, then sa @ ca
            self.last_self_cross_attns = self.self_cross_attns
            self.last_self_cross_attns_1 = self.self_cross_attns_1
                
            # reset
            self.self_attns = []
            self.self_attns_1 = []
            self.cross_attns = []
            self.cross_attns_1 = []
            self.self_cross_attns = []
            self.self_cross_attns_1 = []

    def get_mask_by_2attention(self, target_mask_h, target_mask_w): # get target mask by self-attention map and cross-attention
        mask_target_list = [] # mask on target image, len=self.subj_num

        if self.last_self_cross_attns[0].shape[1] == target_mask_h*target_mask_w:
            mask_h = target_mask_h
            mask_w = target_mask_w
        else: # first attention layer
            mask_h = target_mask_h // 2
            mask_w = target_mask_w // 2

        # First sa@ca, then sum up.
        self_cross_attns = torch.stack(self.last_self_cross_attns, dim=0) # (L, 1, H*W, 77) : [32, 1, 960, 77]
        self_cross_attns = self_cross_attns.reshape((self_cross_attns.shape[0], mask_h, mask_w, self_cross_attns.shape[-1])) # (L, H, W, 77) : [32, 24, 40, 77]
        # rearrange to L, 77, H, W
        self_cross_attns = self_cross_attns.permute(0, 3, 1, 2) # (L, 77, H, W) : [32, 77, 24, 40]
        self_cross_attns = F.interpolate(self_cross_attns.to(torch.float), (target_mask_h, target_mask_w), mode='bilinear') # (L1, 77, H, W) : [32, 77, 24, 40]
        # rearrange to L, 1, H*W, 77
        self_cross_attns = self_cross_attns.permute(0, 2, 3, 1) # (L, H, W, 77) : [32, 24, 40, 77]
        self_cross_attns = self_cross_attns.reshape((self_cross_attns.shape[0], -1, self_cross_attns.shape[-1]) ).unsqueeze(1) # (L, 1, H*W, 77) : [32, 1, 960, 77]

        self_cross_attns_1 = torch.stack(self.last_self_cross_attns_1, dim=0) # (L, 1, 2H*2W, 77) : [32, 1, 3840, 77]
        self_cross_attns_1 = self_cross_attns_1.reshape((self_cross_attns_1.shape[0], mask_h*2, mask_w*2, self_cross_attns_1.shape[-1])) # (L, 2H, 2W, 77) : [32, 48, 80, 77]
        self_cross_attns_1 = self_cross_attns_1.permute(0, 3, 1, 2) # (L, 77, 2H, 2W) : [32, 77, 48, 80]
        self_cross_attns_1 = F.interpolate(self_cross_attns_1.to(torch.float), (target_mask_h, target_mask_w), mode='bilinear') # (L2, 77, H, W) : [32, 77, 24, 40]
        self_cross_attns_1 = self_cross_attns_1.permute(0, 2, 3, 1) # (L, H, W, 77) : [32, 24, 40, 77]
        self_cross_attns_1 = self_cross_attns_1.reshape((self_cross_attns_1.shape[0], -1, self_cross_attns_1.shape[-1]) ).unsqueeze(1) # (L, 1, H*W, 77) : [32, 1, 960, 77]

        # average
        self_cross_attns = torch.cat([self_cross_attns, self_cross_attns_1], dim=0) # (L, 1, H*W, 77) : [L, 1, 960, 77]

        sot_attn_mask = self_cross_attns[:,:,:, 0:1] # (L, 1, H*W, 1) : [32, 1, 960, 1]
        sot_attn_mask = sot_attn_mask.mean(dim=0, keepdim=False).reshape((target_mask_h, target_mask_w)) # (H, W) : [24, 40]
        for subj_idx in range(self.subj_num):
            target_attn_mask_i = self_cross_attns[:,:,:, self.cur_token_idx[subj_idx]] # (L, 1, H*W, token_idx) : [32, 1, 960, 1]
            target_attn_mask_i = target_attn_mask_i.mean(dim=0, keepdim=False).reshape((target_mask_h, target_mask_w, target_attn_mask_i.shape[-1])) # (H, W, idx_list) : ([24, 40, 1])
            if isinstance(self.cur_token_idx[subj_idx], list) or len(target_attn_mask_i.shape) == 4:
                target_attn_mask_i = target_attn_mask_i.max(-1)[0]

            target_attn_mask_i = self.normalize(target_attn_mask_i.unsqueeze(0)).squeeze(0)
            mask_target_list.append(target_attn_mask_i)

        return mask_target_list + [sot_attn_mask] # (H, W), (24, 40)

    # Treat self_attention (sa) and cross_attention (ca) as connected graphs
    # The accumulated sum is calculated as: sa^n * ca
    # self_attention.shape: B,H,W,H*W, cross_attention.shape: B,H,W,dim
    def self_cross_attn_sum_cg(self, self_attention, cross_attention):
        exponent_N = self.exponent_N
        B, H, W, N = cross_attention.shape
        cross_attention = cross_attention.reshape(B, H*W, N)
        self_attention = self_attention.reshape(B, H*W, H*W)
        total = self_attention
        sa_power = self_attention
        for _ in range(exponent_N - 1):
            sa_power = torch.bmm(sa_power, self_attention)
            total += sa_power
        total = torch.bmm(total, cross_attention) / exponent_N # [B, H*W, token_idx]: [1, 960, 77]

        return total

    def class_seg(self, mask_target_list):
        ret_mask_target_list = []

        attn_map = torch.stack(mask_target_list, dim=0) # (subj_num+1, H, W) : [3, 24, 40]
        # spatial_norm
        attn_map = attn_map / (attn_map.mean(dim=(1, 2), keepdim=True) + 1e-8) # (subj_num+1, H, W) : [3, 24, 40]
        max_attn_map, max_idx = attn_map.max(dim=0, keepdim=False) # (H, W) : [24, 40]
        for subj_idx in range(self.subj_num):
            mask_target_i = (attn_map[subj_idx] == max_attn_map).to(torch.float32)
            ret_mask_target_list.append(mask_target_i)

        return ret_mask_target_list

    def gen_binary_mask(self, mask_h, mask_w, is_cross_attention, device, dtype):
        if self.cur_att_layer == 0:
            self.self_attns_mask_dict[f"original_{mask_h}x{mask_w}"], self.spatial_mask_dict[f"original_{mask_h}x{mask_w}"] = \
                        self._gen_binary_mask(mask_h, mask_w, device, dtype)
            self.self_attns_mask_dict[f"original_{mask_h//2}x{mask_w//2}"], self.spatial_mask_dict[f"original_{mask_h//2}x{mask_w//2}"] = \
                        self._gen_binary_mask(mask_h//2, mask_w//2, device, dtype)
            
            self.self_attns_mask_dict[f"{mask_h}x{mask_w}"], self.spatial_mask_dict[f"{mask_h}x{mask_w}"] = \
                        self._dropout_mask(self.self_attns_mask_dict[f"original_{mask_h}x{mask_w}"], self.spatial_mask_dict[f"original_{mask_h}x{mask_w}"], 
                                            mask_h, mask_w, device, dtype)
            self.self_attns_mask_dict[f"{mask_h//2}x{mask_w//2}"], self.spatial_mask_dict[f"{mask_h//2}x{mask_w//2}"] = \
                        self._dropout_mask(self.self_attns_mask_dict[f"original_{mask_h//2}x{mask_w//2}"], self.spatial_mask_dict[f"original_{mask_h//2}x{mask_w//2}"],
                                            mask_h//2, mask_w//2, device, dtype)

        assert f"{mask_h}x{mask_w}" in self.self_attns_mask_dict, f"self_attns_mask_dict does not have {mask_h}x{mask_w}"
        
        return  self.self_attns_mask_dict[f"{mask_h}x{mask_w}"], self.spatial_mask_dict[f"{mask_h}x{mask_w}"], \
                self.self_attns_mask_dict[f"original_{mask_h}x{mask_w}"], self.spatial_mask_dict[f"original_{mask_h}x{mask_w}"]

    def _dropout_mask(self, self_attns_mask_list, spatial_mask_list, mask_h, mask_w, device, dtype):
        ret_self_attns_mask_list = []
        ret_spatial_mask_list = []
        for subj_idx in range(self.subj_num):
            self_attns_mask_i = self_attns_mask_list[subj_idx]
            spatial_mask_i = spatial_mask_list[subj_idx]
            # random dropout some tokens to enhance the diversity of the generated images
            if self.dropout > 0:
                dropout_mask = torch.bernoulli(torch.ones_like(self_attns_mask_i) * (1 - self.dropout))
                self_attns_mask_i = self_attns_mask_i * dropout_mask # TODO: self-attention mask dropout？
                spatial_mask_i = spatial_mask_i * dropout_mask.unsqueeze(-1)

                # guarantee that the mask is not all zero
                if spatial_mask_i.sum() == 0:
                    spatial_mask_i = spatial_mask_list[subj_idx]
                if self_attns_mask_i.sum() == 0:
                    self_attns_mask_i = self_attns_mask_list[subj_idx]

            ret_self_attns_mask_list.append(self_attns_mask_i)
            ret_spatial_mask_list.append(spatial_mask_i)
        return ret_self_attns_mask_list, ret_spatial_mask_list

    def _gen_binary_mask(self, mask_h, mask_w, device, dtype):
        mask_source_list = []
        mask_target_list = []
        # If sam_mask contains "source_mask" key, use its value as the source mask
        if self.sam_mask is not None and self.sam_mask["source_mask"] is not None and len(self.sam_mask["source_mask"]) > 0:
            for subj_idx in range(self.subj_num): # assume all source masks are from sam_mask
                mask_ref_subj_i = self.sam_mask["source_mask"][subj_idx]
                mask_ref_subj_i = F.interpolate(mask_ref_subj_i.unsqueeze(0).to(torch.float), (mask_h, mask_w), mode='bilinear').squeeze(0)
                mask_ref_subj_i = self.normalize(mask_ref_subj_i).squeeze(0)
                mask_source_list.append(mask_ref_subj_i)
        else:
            for subj_idx in range(self.subj_num):
                if self.cur_att_layer > 4:
                    mask_ref_subj_i = self.aggregate_cross_attn_map(idx=self.ref_token_idx[subj_idx], is_use_external=True)  # (2, H, W)?SD-XL: (torch.Size([4, 24, 40]))
                else:
                    mask_ref_subj_i = torch.ones((mask_h, mask_w), device=device, dtype=dtype)
                mask_source_list.append(mask_ref_subj_i)
        
        # Target mask selection: 1. Prefer sam_mask["target_mask"] if exists; 2. Fallback to traditional sa@ca generation otherwise
        if self.sam_mask is None or self.sam_mask["target_mask"] is None or len(self.sam_mask["target_mask"]) < self.subj_num: # can not use sam_mask
            if self.cur_step == 0: # No mask available for use
                for subj_idx in range(self.subj_num): # all 1 mask
                    mask_cur_subj_i = torch.ones((mask_h, mask_w), device=device, dtype=dtype)
                    mask_target_list.append(mask_cur_subj_i)
            else:
                # get mask by self-attention @ cross-attention
                mask_target_list = self.get_mask_by_2attention(mask_h, mask_w)
        elif self.cur_step < max(self.sam_step, 1): # use sam_mask
            for subj_idx in range(self.subj_num): # all use sam_mask
                mask_cur_subj_i = self.sam_mask["target_mask"][subj_idx] # shape: [1, original_H, original_W]
                if mask_cur_subj_i is not None:
                    mask_cur_subj_i = F.interpolate(mask_cur_subj_i.unsqueeze(0).to(torch.float), (mask_h, mask_w), mode='bilinear').squeeze(0)
                    mask_cur_subj_i = self.normalize(mask_cur_subj_i).squeeze(0)
                else:# all 0
                    mask_cur_subj_i = torch.zeros((mask_h, mask_w), device=device, dtype=dtype)
                mask_target_list.append(mask_cur_subj_i.to(device))
        else: # sam_mask + sa@ca
            if self.cur_step == 0:
                for subj_idx in range(self.subj_num): # No mask available for use
                    mask_cur_subj_i = torch.ones((mask_h, mask_w), device=device, dtype=dtype)
                    mask_target_list.append(mask_cur_subj_i)
            else:
                # get mask by self-attention @ cross-attention
                mask_target_list = self.get_mask_by_2attention(mask_h, mask_w)

            # normalize
            for subj_idx in range(self.subj_num):
                mask_cur_subj_i = mask_target_list[subj_idx]
                mask_cur_subj_i = self.normalize(mask_cur_subj_i.unsqueeze(0)).squeeze(0)
                mask_target_list[subj_idx] = (mask_cur_subj_i.to(device))

            if self.sam_mask is not None:
                for subj_idx in range(self.subj_num):
                    mask_cur_subj_i = self.sam_mask["target_mask"][subj_idx] # shape: [1, original_H, original_W]
                    if mask_cur_subj_i is not None:
                        mask_cur_subj_i = F.interpolate(mask_cur_subj_i.unsqueeze(0).to(torch.float), (mask_h, mask_w), mode='bilinear').squeeze(0)
                        mask_cur_subj_i = self.normalize(mask_cur_subj_i).squeeze(0).to(device).to(dtype)
                    else:# all 0
                        mask_cur_subj_i = torch.zeros((mask_h, mask_w), device=device, dtype=dtype)
                    mask_target_list[subj_idx] = torch.max(mask_target_list[subj_idx].to(device).to(dtype), mask_cur_subj_i)

            mask_target_list = mask_target_list[:self.subj_num]
        
        self_attns_mask_list = [] # mask for source image
        spatial_mask_list = [] # mask for target image
        for subj_idx in range(self.subj_num):
            self_attns_mask_i = F.interpolate(mask_source_list[subj_idx].unsqueeze(0).unsqueeze(0).to(torch.float), (mask_h, mask_w), mode='bilinear').flatten()
            spatial_mask_i = F.interpolate(mask_target_list[subj_idx].unsqueeze(0).unsqueeze(0).to(torch.float), (mask_h, mask_w), mode='bilinear').reshape(-1, 1)

            # binarize the mask
            self_attns_mask_i, spatial_mask_i = self.binarize_mask(self_attns_mask_i, spatial_mask_i)

            self_attns_mask_i = self_attns_mask_i.to(device).to(dtype)
            spatial_mask_i = spatial_mask_i.to(device).to(dtype)

            self_attns_mask_list.append(self_attns_mask_i)
            spatial_mask_list.append(spatial_mask_i)

        return self_attns_mask_list, spatial_mask_list

    def self_attention(self,
        attn: Attention,
        residual: torch.Tensor,
        hidden_states: torch.FloatTensor,
        attention_probs: Optional[torch.FloatTensor] = None,
        query = None, key = None, value = None,
        self_attns_mask_list = None,
        spatial_mask_list = None,
        scale: float = 1.0,
        DIFT_sim = None,
    ) -> torch.Tensor:
        # args = () if USE_PEFT_BACKEND else (scale,)
        args = ()

        if (self.cur_step in self.step_idx_list and self.cur_att_layer // 2 in self.self_attn_layer_idx_list ): # and self.is_spatial_self_attn
            query_unconditional, query_conditional = query.chunk(2, dim=0)
            key_unconditional, key_conditional = key.chunk(2, dim=0)
            value_unconditional, value_conditional = value.chunk(2, dim=0)
            attention_probs_unconditional, attention_probs_conditional = attention_probs.chunk(2, dim=0)

            query_unconditional_list = query_unconditional.chunk(self.subj_num+1, dim=0)
            key_unconditional_list = key_unconditional.chunk(self.subj_num+1, dim=0)
            value_unconditional_list = value_unconditional.chunk(self.subj_num+1, dim=0)
            attention_probs_unconditional_list = attention_probs_unconditional.chunk(self.subj_num+1, dim=0)

            query_conditional_list = query_conditional.chunk(self.subj_num+1, dim=0)
            key_conditional_list = key_conditional.chunk(self.subj_num+1, dim=0)
            value_conditional_list = value_conditional.chunk(self.subj_num+1, dim=0)
            attention_probs_conditional_list = attention_probs_conditional.chunk(self.subj_num+1, dim=0)

            hidden_states_unconditional_list = []
            hidden_states_conditional_list = []
            # source image
            for subj_idx in range(self.subj_num):
                hidden_states_unconditional_i = torch.bmm(attention_probs_unconditional_list[subj_idx], value_unconditional_list[subj_idx])
                hidden_states_unconditional_i = attn.batch_to_head_dim(hidden_states_unconditional_i)
                hidden_states_conditional_i = torch.bmm(attention_probs_conditional_list[subj_idx], value_conditional_list[subj_idx])
                hidden_states_conditional_i = attn.batch_to_head_dim(hidden_states_conditional_i)
                hidden_states_unconditional_list.append(hidden_states_unconditional_i)
                hidden_states_conditional_list.append(hidden_states_conditional_i)

            # vanilla target image
            vanilla_hidden_states_unconditional_target = torch.bmm(attention_probs_unconditional_list[-1], value_unconditional_list[-1])
            vanilla_hidden_states_unconditional_target = attn.batch_to_head_dim(vanilla_hidden_states_unconditional_target)
            vanilla_hidden_states_conditional_target = torch.bmm(attention_probs_conditional_list[-1], value_conditional_list[-1])
            vanilla_hidden_states_conditional_target = attn.batch_to_head_dim(vanilla_hidden_states_conditional_target)

            if not self.is_spatial_self_attn: # spatial_mask_list set to all one
                spatial_mask_all = None
            else:
                spatial_mask_all = torch.cat(spatial_mask_list + [torch.ones_like(spatial_mask_list[-1])], dim=0)
            self_attns_mask_all = torch.cat(self_attns_mask_list + [torch.ones_like(self_attns_mask_list[-1])], dim=0)
            hidden_states_unconditional_target = self.mask_attention_multi_subj(query_unconditional_list[-1], 
                                                        torch.cat(key_unconditional_list, dim=0), torch.cat(value_unconditional_list, dim=0),
                                                        attn, self_attns_mask_all,  spatial_mask_all)
            hidden_states_conditional_target = self.mask_attention_multi_subj(query_conditional_list[-1],
                                                        torch.cat(key_conditional_list, dim=0), torch.cat(value_conditional_list, dim=0),
                                                        attn, self_attns_mask_all,  spatial_mask_all)
            hidden_states_unconditional_list.append(hidden_states_unconditional_target)
            hidden_states_conditional_list.append(hidden_states_conditional_target)
            hidden_states = torch.cat(hidden_states_unconditional_list + hidden_states_conditional_list, dim=0) # shape: [2*(subj_num+1), 960, 1280]
        else:
            # standard calculation
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)

        # cur_step belongs to [5, 16) with 50 total steps, equal to  [680, 900] in DDIM with 1000 taotal steps
        if self.cur_step in self.step_idx_list and self.cur_att_layer // 2 in self.self_attn_layer_idx_list and \
                self.cur_step > 5 and self.cur_step < 16 and self.is_DIFT and DIFT_sim is not None: # DIFT feature injection
            hidden_states_unconditional, hidden_states_conditional = hidden_states.chunk(2, dim=0)
            hidden_states_unconditional, hidden_states_conditional = \
                        self.DIFT_injection_multisuj(DIFT_sim, self_attns_mask_list, spatial_mask_list,
                                            hidden_states_unconditional, hidden_states_conditional,
                                            dtype=query.dtype)
            hidden_states = torch.cat([hidden_states_unconditional, hidden_states_conditional], dim=0)
        
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def cross_attention(self,
        attn: Attention,
        residual: torch.Tensor,
        hidden_states: torch.FloatTensor,
        attention_probs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        query = None, key = None, value = None,
        spatial_mask_list = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        # args = () if USE_PEFT_BACKEND else (scale,)
        args = ()

        # hidden_states.shape: [6, 3840, 640], query.shape: [60, 3840, 64], key.shape: [60, 77, 64], value.shape: [60, 77, 64], attention_probs.shape: [60, 3840, 77]

        # standard calculation
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states) # shape = [2*(subj_num+1, H*W, dim)] : ([6, 3840, 640])
        
        if self.is_mutual_cross_attn : # and self.cur_step > 5
            query_unconditional, query_conditional = query.chunk(2, dim=0)
            query_unconditional_list = query_unconditional.chunk(self.subj_num+1, dim=0)
            query_conditional_list = query_conditional.chunk(self.subj_num+1, dim=0)

            key_unconditional, key_conditional = key.chunk(2, dim=0)
            key_unconditional_list = key_unconditional.chunk(self.subj_num+1, dim=0)
            key_conditional_list = key_conditional.chunk(self.subj_num+1, dim=0)

            value_unconditional, value_conditional = value.chunk(2, dim=0)
            value_unconditional_list = value_unconditional.chunk(self.subj_num+1, dim=0)
            value_conditional_list = value_conditional.chunk(self.subj_num+1, dim=0)

            mutual_hidden_states_unconditional_target_list = []
            mutual_hidden_states_conditional_target_list = []
            for subj_idx in range(self.subj_num):
                mutual_hidden_states_unconditional_target_i = self.mask_attention_fg(query_unconditional_list[-1], key_unconditional_list[subj_idx], value_unconditional_list[subj_idx], 
                                                                        attn, None, spatial_mask_list[subj_idx])
                mutual_hidden_states_conditional_target_i = self.mask_attention_fg(query_conditional_list[-1], key_conditional_list[subj_idx], value_conditional_list[subj_idx],
                                                                        attn, None, spatial_mask_list[subj_idx])
                mutual_hidden_states_unconditional_target_list.append(mutual_hidden_states_unconditional_target_i)
                mutual_hidden_states_conditional_target_list.append(mutual_hidden_states_conditional_target_i)

            # 拼成mutual_hidden_states
            mutual_hidden_states_unconditional_target = torch.cat(mutual_hidden_states_unconditional_target_list, dim=0)
            mutual_hidden_states_conditional_target = torch.cat(mutual_hidden_states_conditional_target_list, dim=0)
            mutual_hidden_states = torch.cat([mutual_hidden_states_unconditional_target, mutual_hidden_states_conditional_target], dim=0)
            # mutual_hidden_states.shape = [subj_num*2, H*W, dim]: ([4, 3840, 640])
                
            # linear proj
            mutual_hidden_states = attn.to_out[0](mutual_hidden_states, *args)
            # dropout
            mutual_hidden_states = attn.to_out[1](mutual_hidden_states)

            # fuse mutual_hidden_states to hidden_states before residual connection
            hidden_states_unconditional, hidden_states_conditional = hidden_states.chunk(2, dim=0)
            mutual_hidden_states_unconditional, mutual_hidden_states_conditional = mutual_hidden_states.chunk(2, dim=0)
            
            hidden_states_unconditional_list = list(hidden_states_unconditional.chunk(self.subj_num+1, dim=0))
            hidden_states_conditional_list = list(hidden_states_conditional.chunk(self.subj_num+1, dim=0))
            mutual_hidden_states_unconditional_list = list(mutual_hidden_states_unconditional.chunk(self.subj_num+1, dim=0))
            mutual_hidden_states_conditional_list = list(mutual_hidden_states_conditional.chunk(self.subj_num+1, dim=0))

            vanilla_hidden_states_unconditional_target = hidden_states_unconditional_list[-1]
            vanilla_hidden_states_conditional_target = hidden_states_conditional_list[-1]
            
            # average maked area in the hidden_states_conditional_target
            masked_area = torch.zeros_like(spatial_mask_list[-1]) # shape = [H*W, 1] : [3840, 1]
            hidden_states_unconditional_target = torch.zeros_like(vanilla_hidden_states_unconditional_target)
            hidden_states_conditional_target = torch.zeros_like(vanilla_hidden_states_conditional_target)
            
            scale_ratio = (self.latents_shape[0] * self.latents_shape[1] // query.shape[1]) ** 0.5
            mask_h, mask_w = int(self.latents_shape[0] // scale_ratio), int(self.latents_shape[1] // scale_ratio)
            mutual_ca_spatial_mask_list = spatial_mask_list if self.is_dropout_ca else self.spatial_mask_dict[f"original_{mask_h}x{mask_w}"]
            for subj_idx in range(self.subj_num): # TODO: accelerate
                masked_area += mutual_ca_spatial_mask_list[subj_idx] # get masked hidden_states and its erea
                hidden_states_unconditional_target += mutual_hidden_states_unconditional_list[subj_idx] * mutual_ca_spatial_mask_list[subj_idx]
                hidden_states_conditional_target += mutual_hidden_states_conditional_list[subj_idx] * mutual_ca_spatial_mask_list[subj_idx]
            
            hidden_states_unconditional_target = hidden_states_unconditional_target / (masked_area + 1e-8) # shape = [1, H*W, dim] : [1, 960, 1280]
            hidden_states_conditional_target = hidden_states_conditional_target / (masked_area + 1e-8)
            
            unmasked_area = (masked_area == 0).float().to(query.device).to(query.dtype) # shape = [H*W, 1] : [3840, 1]
            masked_area = 1-unmasked_area

            mutual_cross_attention_lambda = self.mutual_cross_attention_lambda
            hidden_states_unconditional_target = vanilla_hidden_states_unconditional_target * unmasked_area + \
                                                vanilla_hidden_states_unconditional_target * masked_area * (1-mutual_cross_attention_lambda) + \
                                                hidden_states_unconditional_target * masked_area * mutual_cross_attention_lambda
            hidden_states_conditional_target = vanilla_hidden_states_conditional_target * unmasked_area + \
                                                vanilla_hidden_states_conditional_target * masked_area * (1-mutual_cross_attention_lambda) + \
                                                hidden_states_conditional_target * masked_area * mutual_cross_attention_lambda

            hidden_states_unconditional_list[-1] = hidden_states_unconditional_target
            hidden_states_conditional_list[-1] = hidden_states_conditional_target
            hidden_states = torch.cat(hidden_states_unconditional_list + hidden_states_conditional_list, dim=0)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        DIFT_sim = None,
    ) -> torch.Tensor:
        residual = hidden_states

        # args = () if USE_PEFT_BACKEND else (scale,)
        args = ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # It's different from here
        query_dim = query.shape[1]
        key_dim = key.shape[1]
        is_cross_attention = (key_dim != query_dim)

        scale_ratio = (self.latents_shape[0] * self.latents_shape[1] // query.shape[1]) ** 0.5
        mask_h, mask_w = int(self.latents_shape[0] // scale_ratio), int(self.latents_shape[1] // scale_ratio) # assume the shape of mask_target is the same as the mask_source
        
        dropout_self_attns_mask_list, dropout_spatial_mask_list, self_attns_mask_list, spatial_mask_list = \
            self.gen_binary_mask(mask_h, mask_w, is_cross_attention, query.device, query.dtype)

        is_target_sam_mask = not (self.sam_mask is None or self.sam_mask["target_mask"] is None or len(self.sam_mask["target_mask"]) < self.subj_num)

        # 收集attention
        if is_cross_attention:
            if attention_probs.shape[1] == self.aggregation_attn_size:
                num_heads = attention_probs.shape[0] // batch_size
                self.cross_attns.append(attention_probs.reshape(-1, num_heads, *attention_probs.shape[-2:]).mean(1))
                if self.cur_step > self.sam_step - 2 or not is_target_sam_mask: # first sa@ca，than accumulate
                    _self_attention = self.self_attns[-1].reshape((self.self_attns[-1].shape[0], mask_h, mask_w, mask_h*mask_w)) # (subj_num*2+2, H, W, H*W): torch.Size([6, 24, 40, 960])
                    _cross_attention = self.cross_attns[-1].reshape((self.cross_attns[-1].shape[0], mask_h, mask_w, self.cross_attns[-1].shape[-1])) # (subj_num*2+2, H, W, 77): torch.Size([6, 24, 40, 77])
                    self.self_cross_attns.append(self.self_cross_attn_sum_cg(_self_attention[-1:,:,:,:], _cross_attention[-1:,:,:,:])) # (1, H*W, 77)
            elif attention_probs.shape[1] == self.aggregation_attn_size_1:
                num_heads = attention_probs.shape[0] // batch_size
                self.cross_attns_1.append(attention_probs.reshape(-1, num_heads, *attention_probs.shape[-2:]).mean(1))
                if self.cur_step > self.sam_step - 2 or not is_target_sam_mask: # first sa@ca，than accumulate
                    _self_attention_1 = self.self_attns_1[-1].reshape((self.self_attns_1[-1].shape[0], mask_h, mask_w, mask_h*mask_w)) # (subj_num*2+2, H, W, H*W)： [6, 3840, 3840]
                    _cross_attention_1 = self.cross_attns_1[-1].reshape((self.cross_attns_1[-1].shape[0], mask_h, mask_w, self.cross_attns_1[-1].shape[-1])) # (subj_num*2+2, H, W, 77): [6, 3840, 77]
                    self.self_cross_attns_1.append(self.self_cross_attn_sum_cg(_self_attention_1[-1:,:,:,:], _cross_attention_1[-1:,:,:,:])) # (1, H*W, 77): [1, 3840, 77]
        else: # gather self-attention map
            if attention_probs.shape[1] == self.aggregation_attn_size:
                num_heads = attention_probs.shape[0] // batch_size
                self.self_attns.append(attention_probs.reshape(-1, num_heads, *attention_probs.shape[-2:]).mean(1))
            elif attention_probs.shape[1] == self.aggregation_attn_size_1:
                num_heads = attention_probs.shape[0] // batch_size
                self.self_attns_1.append(attention_probs.reshape(-1, num_heads, *attention_probs.shape[-2:]).mean(1))

        if not is_cross_attention:
            hidden_states = self.self_attention(attn=attn, residual=residual, hidden_states=hidden_states,
                                                attention_probs=attention_probs, query=query, key=key, value=value,
                                                self_attns_mask_list=dropout_self_attns_mask_list, spatial_mask_list=dropout_spatial_mask_list, 
                                                scale=scale, DIFT_sim=DIFT_sim)
        else:
            hidden_states = self.cross_attention(attn=attn, residual=residual, hidden_states=hidden_states, attention_mask=attention_mask,
                                                attention_probs=attention_probs, query=query, key=key, value=value, 
                                                spatial_mask_list=dropout_spatial_mask_list, scale=scale)
            
        self.after_step() # update the step and layer index after every attention layer

        return hidden_states
    
    def get_mask_image(self, mask, mask_h=None, mask_w=None):
        if mask_h is not None or mask_w is not None:
            mask = mask.reshape((mask_h, mask_w))
        assert mask.shape[0] > 1 and mask.shape[1] > 1, f"mask.shape: {mask.shape} is not valid."
        assert mask.shape == (24, 40) or mask.shape == (48, 80), f"mask.shape: {mask.shape} is not valid!!!!!!!!!!!!!!!!!1." 
        ret_img = F.interpolate(mask.unsqueeze(0).unsqueeze(0).to(torch.float), self.latents_shape, mode='nearest')
        ret_img = (ret_img * 255).cpu().to(torch.float32).squeeze(0).squeeze(0).numpy().astype(np.uint8)
        ret_img = Image.fromarray(ret_img).resize((self.latents_shape[1] * 8, self.latents_shape[0] * 8), resample=Image.NEAREST)
        return ret_img

    def mask_attention_fg(self, query, key, value, 
                    attn, self_attns_mask, spatial_mask=None, attention_mask=None):
        """
        Implements foreground mask attention for cross-attention:
        Forces target image's query to attend ONLY to source image's text features
        """
        
        dtype = query.dtype
        if attn.upcast_attention:
            query = query.float()
            key = key.float()

        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
        
        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=attn.scale,
        )
        del baddbmm_input

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        # query.shape: torch.Size([20, 960, 64]), key.shape = value.shape =: torch.Size([40, 960, 64]), attention_scores.shape: torch.Size([20, 960, 1920])
        # self_attns_mask.shape: torch.Size([1920]), spatial_mask.shape: torch.Size([960, 1])
        if spatial_mask is not None:
            attention_scores = attention_scores + torch.where(spatial_mask == 0, torch.finfo(attention_scores.dtype).min, torch.tensor(0, dtype=attention_scores.dtype).to(attention_scores.device))

        Batch_size_x_head_dim, seq_len, attention_len = attention_scores.shape
        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        # attention_probs.shape: torch.Size([10, 960, 77*2]),value.shape: torch.Size([20, 77, 64])
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # hidden_states.shape = torch.Size([1, 960, 1280])

        return hidden_states
    
    def mask_attention_multi_subj(self, query, key, value, 
                        attn, self_attns_mask, spatial_mask=None,
                        attention_mask=None):
        """
        Performing mask attention for a batch of queries, keys, and values
        """
        dtype = query.dtype
        if attn.upcast_attention:
            query = query.float()
            key = key.float()

        _tmp_query = torch.cat([query]*(self.subj_num+1), dim=0)
        baddbmm_input = torch.empty(
            _tmp_query.shape[0], _tmp_query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
        
        attention_scores = torch.baddbmm(
            baddbmm_input,
            _tmp_query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=attn.scale,
        )
        del baddbmm_input
        attention_scores = torch.cat(attention_scores.chunk(self.subj_num+1, dim=0), dim=-1) # attention_scores[B, N, (subj_num+1)*dim]

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        # query.shape: torch.Size([20, 960, 64]), key.shape = value.shape : torch.Size([20*(subj_num+1), 960, 64]), 
        # attention_scores.shape: torch.Size([20, 960, 960*(subj_num+1)])
        # self_attns_mask.shape: torch.Size([960*(subj_num+1)]), spatial_mask.shape: torch.Size([960*(subj_num+1), 1])
        if self_attns_mask is not None:
            self_attns_mask = self_attns_mask.unsqueeze(0).unsqueeze(0) # to shape [1, 1, 960*(subj_num+1)]
            attention_scores = attention_scores + torch.where(self_attns_mask == 0, torch.finfo(attention_scores.dtype).min, torch.tensor(0, dtype=attention_scores.dtype).to(attention_scores.device))
        if spatial_mask is not None:
            Batch_size_x_head_dim, Dim1, Dim2 = attention_scores.shape
            single_Dim2 = Dim2 // (self.subj_num+1)
            for subj_idx in range(self.subj_num):
                spatial_mask_i = spatial_mask[subj_idx*single_Dim2:(subj_idx+1)*single_Dim2].unsqueeze(0) # to shape [1, 960, 1]
                attention_scores[:,:,subj_idx*single_Dim2:(subj_idx+1)*single_Dim2] += torch.where(spatial_mask_i == 0, torch.finfo(attention_scores.dtype).min, torch.tensor(0, dtype=attention_scores.dtype).to(attention_scores.device))

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        # attention_probs.shape: torch.Size([20, 960, 960*(subj_num+1)]), value.shape: torch.Size([20*(subj_num+1), 960, 64])
        value = torch.cat(value.chunk(self.subj_num+1, dim=0), dim=1)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # hidden_states.shape = torch.Size([1, 960, 1280])

        return hidden_states

    def DIFT_injection_multisuj(self, DIFT_sim, self_attns_mask_list, spatial_mask_list,
                        hidden_states_unconditional, hidden_states_conditional,
                        DIFT_injection_lambda=0.8, dtype=torch.float32):
        _, HW, _ = hidden_states_unconditional.shape
        DIFT_sim = DIFT_sim[f"DIFT_sim_{HW}"]
        for subj_idx in range(self.subj_num):
            DIFT_sim_i = DIFT_sim[subj_idx:subj_idx+1] # [subj_num, H*W, H*W] -> [1, H*W, H*W] = [1, 960, 960]
            self_attns_mask_i = self_attns_mask_list[subj_idx] # -> [960]
            spatial_mask_i = spatial_mask_list[subj_idx].unsqueeze(0) # [960, 1] -> [1, 960, 1]

            max_response, max_response_idx = DIFT_sim_i.max(dim=2) # [1, H*W] = [1, 960]
            ostu_thres = self.get_otsu_threshold(max_response).to(dtype)
            mask_dift_i = max_response > ostu_thres # shape: [1, H*W] = [1, 960]
            mask_dift_i = (mask_dift_i * DIFT_injection_lambda * self_attns_mask_i).unsqueeze(-1).to(dtype) # [1, H*W] -> [1, H*W, 1]: [1, 960] -> [1, 960, 1]
            mask_dift_i = (mask_dift_i * spatial_mask_i).to(dtype) # [1, H*W, 1] * [1, H*W, 1] -> [1, H*W, 1]: [1, 960, 1] * [1, 960, 1] -> [1, 960, 1]
            max_response_idx = max_response_idx[0,:]# [1, H*W] -> [H*W]: [1, 960] -> [960]
            
            hidden_states_unconditional[-1:] = hidden_states_unconditional[-1:] * (1 - mask_dift_i) + \
                                                hidden_states_unconditional[subj_idx:subj_idx+1][:, max_response_idx, :] * mask_dift_i
            hidden_states_conditional[-1:] = hidden_states_conditional[-1:] * (1 - mask_dift_i) + \
                                                hidden_states_conditional[subj_idx:subj_idx+1][:, max_response_idx, :] * mask_dift_i

        return hidden_states_unconditional, hidden_states_conditional