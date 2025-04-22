
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND

from .MasaCtrlAttnProcessor import MasaCtrlAttnProcessor

class ConsiStoryAttnProcessor(MasaCtrlAttnProcessor):

    '''
    Implementation of ConsiStory
    '''

    def __init__(self, start_step=4, start_layer=54, self_attn_layer_idx_list=None, step_idx_list=None, total_steps=50, 
                thres=0, ref_token_idx=[1], cur_token_idx=[1], model_type="SDXL",
                cross_attn_gather_down_layer=2, latents_shape=(768//8, 1280//8), is_all_cross_attns=False,
                dropout=0, is_output_mask=False, is_DIFT=False, **kwargs):
        super().__init__(start_step=start_step, start_layer=start_layer, self_attn_layer_idx_list=self_attn_layer_idx_list, 
                        step_idx_list=step_idx_list, total_steps=total_steps,
                        thres=thres, ref_token_idx=ref_token_idx, cur_token_idx=cur_token_idx,
                        model_type=model_type, cross_attn_gather_down_layer=cross_attn_gather_down_layer, latents_shape=latents_shape,
                        dropout=dropout, is_output_mask=is_output_mask,
                        is_all_cross_attns=is_all_cross_attns,)
        self.is_DIFT = is_DIFT
        if len(kwargs) > 0:
            print(f"Warning: get unexpected parameters: {kwargs}")

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
        residual = hidden_states # shape = (B, channel, seq_len)

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

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

        if is_cross_attention:
            if attention_probs.shape[1] == self.aggregation_attn_size: # for SD-XL is 32 * 32 and for SD is 16 * 16
                num_heads = attention_probs.shape[0] // batch_size
                self.cross_attns.append(attention_probs.reshape(-1, num_heads, *attention_probs.shape[-2:]).mean(1))
            
        # apply ConsiStory
        if not is_cross_attention and self.cur_step in self.step_idx_list and self.cur_att_layer // 2 in self.self_attn_layer_idx_list:
            query_unconditional, query_conditional = query.chunk(2, dim=0)
            key_unconditional, key_conditional = key.chunk(2, dim=0)
            value_unconditional, value_conditional = value.chunk(2, dim=0)
            attention_probs_unconditional, attention_probs_conditional = attention_probs.chunk(2, dim=0)

            query_unconditional_source, query_unconditional_target = query_unconditional.chunk(2, dim=0)
            key_unconditional_source, key_unconditional_target = key_unconditional.chunk(2, dim=0)
            value_unconditional_source, value_unconditional_target = value_unconditional.chunk(2, dim=0)
            attention_probs_unconditional_source, attention_probs_unconditional_target = attention_probs_unconditional.chunk(2, dim=0)

            query_conditional_source, query_conditional_target = query_conditional.chunk(2, dim=0)
            key_conditional_source, key_conditional_target = key_conditional.chunk(2, dim=0)
            value_conditional_source, value_conditional_target = value_conditional.chunk(2, dim=0)
            attention_probs_conditional_source, attention_probs_conditional_target = attention_probs_conditional.chunk(2, dim=0)

            # source image
            hidden_states_unconditional_source = torch.bmm(attention_probs_unconditional_source, value_unconditional_source)
            hidden_states_unconditional_source = attn.batch_to_head_dim(hidden_states_unconditional_source)
            hidden_states_conditional_source = torch.bmm(attention_probs_conditional_source, value_conditional_source)
            hidden_states_conditional_source = attn.batch_to_head_dim(hidden_states_conditional_source)

            # get mask
            mask = self.aggregate_cross_attn_map(idx=self.ref_token_idx)  # (2, H, W)? SD-XL: (torch.Size([4, 24, 40]))
            mask_source = mask[-2]  # (H, W) source image mask
            mask = self.aggregate_cross_attn_map(idx=self.cur_token_idx)  # (2, H, W)? SD-XL: (torch.Size([4, 24, 40]))
            mask_target = mask[-1]  # (H, W) target image mask
            
            scale_ratio = (self.latents_shape[0] * self.latents_shape[1] // query.shape[1]) ** 0.5
            mask_h, mask_w = int(self.latents_shape[0] // scale_ratio), int(self.latents_shape[1] // scale_ratio) # assume the shape of mask_target is the same as the mask_source
            # F.interpolate do not support bfloat16, transfer to float
            self_attns_mask = F.interpolate(mask_source.unsqueeze(0).unsqueeze(0).to(torch.float), (mask_h, mask_w)).flatten()
            spatial_mask = F.interpolate(mask_target.unsqueeze(0).unsqueeze(0).to(torch.float), (mask_h, mask_w)).reshape(-1, 1).to(query.device)

            # binarize the mask
            self_attns_mask, spatial_mask = self.binarize_mask(self_attns_mask, spatial_mask)            

            if self.is_remove_outliers: # remove the outliers
                self_attns_mask, spatial_mask = self.remove_outliers(self_attns_mask, spatial_mask, query.device, mask_h, mask_w)

            self_attns_mask = self_attns_mask.to(query.device).to(query.dtype)
            spatial_mask = spatial_mask.to(query.device).to(query.dtype)

            # random dropout some tokens to enhance the diversity of the generated images
            if self.dropout > 0:
                dropout_mask = torch.bernoulli(torch.ones_like(self_attns_mask) * (1 - self.dropout))
                self_attns_mask = self_attns_mask * dropout_mask
                spatial_mask = spatial_mask * dropout_mask.unsqueeze(-1)

            if self.cur_step < 5: # Using Vanilla Query Features
                lambda_q = 0.9
                query_unconditional_target = (1-lambda_q) * query_unconditional_source + lambda_q * query_unconditional_target
                query_conditional_target = (1-lambda_q) * query_conditional_source + lambda_q * query_conditional_target

            self_attns_mask = torch.cat([self_attns_mask, torch.ones_like(self_attns_mask)], dim=0)
            hidden_states_unconditional_target = self.mask_attention(query_unconditional_target, key_unconditional, value_unconditional, attn, self_attns_mask)  # torch.Size([1, 960, 1280])
            hidden_states_conditional_target = self.mask_attention(query_conditional_target, key_conditional, value_conditional, attn, self_attns_mask)  # torch.Size([1, 960, 1280])

            hidden_states = torch.cat([hidden_states_unconditional_source, hidden_states_unconditional_target, 
                                        hidden_states_conditional_source, hidden_states_conditional_target], dim=0)

            del hidden_states_unconditional_source, hidden_states_unconditional_target, hidden_states_conditional_source, hidden_states_conditional_target

        else: # standard calculation
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)

        # According to the paper, DIFT feature injection should be applied here.
        # cur_step belongs to [5, 16) with 50 total steps, equal to  [680, 900] in DDIM with 1000 taotal steps
        if not is_cross_attention and self.cur_step in self.step_idx_list and self.cur_att_layer // 2 in self.self_attn_layer_idx_list and \
            self.cur_step > 5 and self.cur_step < 16 and self.is_DIFT and DIFT_sim is not None: # DIFT feature injection
            hidden_states_unconditional, hidden_states_conditional = hidden_states.chunk(2, dim=0)
            hidden_states_unconditional_source, hidden_states_unconditional_target = hidden_states_unconditional.chunk(2, dim=0)
            hidden_states_conditional_source, hidden_states_conditional_target = hidden_states_conditional.chunk(2, dim=0)
            hidden_states_unconditional_target, hidden_states_conditional_target = \
                        self.DIFT_injection(DIFT_sim, self_attns_mask,
                                            hidden_states_unconditional_source, hidden_states_conditional_source, 
                                            hidden_states_unconditional_target, hidden_states_conditional_target,
                                            dtype=query.dtype)
            hidden_states = torch.cat([hidden_states_unconditional_source, hidden_states_unconditional_target,
                                        hidden_states_conditional_source, hidden_states_conditional_target], dim=0)
        
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        self.after_step() # update the step and layer index after every attention layer

        return hidden_states

    def mask_attention(self, query, key, value, 
                        attn, self_attns_mask, spatial_mask=None, attention_mask=None):
        """
        Performing mask attention for a batch of queries, keys, and values
        """
        dtype = query.dtype
        if attn.upcast_attention:
            query = query.float()
            key = key.float()

        _tmp_query = torch.cat([query]*2, dim=0)
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
        attention_scores = torch.cat(attention_scores.chunk(2, dim=0), dim=-1) # attention_scores[B, N, 2*dim]

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        # query.shape: torch.Size([20, 960, 64]), key.shape = value.shape =: torch.Size([40, 960, 64]), attention_scores.shape: torch.Size([20, 960, 1920])
        # self_attns_mask.shape: torch.Size([1920]), spatial_mask.shape: torch.Size([960, 1])
        if self_attns_mask is not None:
            self_attns_mask = self_attns_mask.unsqueeze(0).unsqueeze(0) # to shape [1, 1, 1920]
            attention_scores = attention_scores + self_attns_mask.masked_fill(self_attns_mask == 0, torch.finfo(attention_scores.dtype).min)
        if spatial_mask is not None:
            Batch_size_x_head_dim, Dim1, Dim2 = attention_scores.shape
            attention_scores[:,:,:Dim2//2] = attention_scores[:,:,:Dim2//2] + spatial_mask.masked_fill(spatial_mask == 0, torch.finfo(attention_scores.dtype).min)

        Batch_size_x_head_dim, seq_len, attention_len = attention_scores.shape
        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        # attention_probs.shape: torch.Size([20, 960, 960*2]), value.shape: torch.Size([40, 960, 64])
        value = torch.cat(value.chunk(2, dim=0), dim=1)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # hidden_states.shape = torch.Size([1, 960, 1280])

        return hidden_states

    def DIFT_injection(self, DIFT_sim, self_attns_mask,
                        hidden_states_unconditional_source, hidden_states_conditional_source,
                        hidden_states_unconditional_target, hidden_states_conditional_target,
                        DIFT_injection_lambda=0.8, dtype=torch.float32):
        _, HW, _ = hidden_states_unconditional_target.shape
        DIFT_sim = DIFT_sim[f"DIFT_sim_{HW}"]
        self_attns_mask = self_attns_mask.chunk(2, dim=0)[0] # [1920] -> [960]

        # DIFT_sim.shape = [1, H*W, H*W]
        # dift_sim[i] = torch.matmul(DIFT_sims[-1], DIFT_sims[i].transpose(0, 1))
        # DIFT_sim is a matrix, representing the response value of the source image at each position on the target image

        max_response, max_response_idx = DIFT_sim.max(dim=2) # [1, H*W] = [1, 960]
        ostu_thres = self.get_otsu_threshold(max_response).to(hidden_states_conditional_source.dtype)
        mask_dift = max_response > ostu_thres # shape: [1, H*W] = [1, 960]
        mask_dift = (mask_dift * DIFT_injection_lambda * self_attns_mask).unsqueeze(-1) # [1, H*W] -> [1, H*W, 1]: [1, 960] -> [1, 960, 1]
        max_response_idx = max_response_idx[0,:] # [1, H*W] -> [H*W]: [1, 960] -> [960]
        
        hidden_states_unconditional_target = hidden_states_unconditional_target * (1 - mask_dift) + \
                                             hidden_states_unconditional_source[:, max_response_idx, :] * mask_dift
        hidden_states_conditional_target = hidden_states_conditional_target * (1 - mask_dift) + \
                                             hidden_states_conditional_source[:, max_response_idx, :] * mask_dift
        
        return hidden_states_unconditional_target.to(dtype), hidden_states_conditional_target.to(dtype)