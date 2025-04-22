from typing import Optional
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.nn import functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND

class MasaCtrlAttnProcessor:
    MODEL_SELF_ATTN_LAYERS = {
        "SD": 16,
        "SDXL": 70,
    }

    def __init__(self, start_step=4, start_layer=54, self_attn_layer_idx_list=None, step_idx_list=None, total_steps=50, 
                thres=0, ref_token_idx=[1], cur_token_idx=[1], model_type="SDXL",
                cross_attn_gather_down_layer=2, latents_shape=(768//8, 1280//8), is_all_cross_attns=False,
                dropout=0, is_output_mask=False,
                is_remove_outliers=False, outlier_neighbor_size=1, is_filter_attn=False, filter_kernel=3,
                external_cross_attns=None, **kwargs):
        """
        Modified from MasaCtrl (https://github.com/TencentARC/MasaCtrl)
        MasaCtrl with mask auto generation from cross-attention map
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            self_attn_layer_idx_list: list of the layers to apply mutual self-attention control
            step_idx_list: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation
            cur_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image

            dropout: (from ConsiStory) random dropout some tokens in the mask of self-attention to enhance the diversity of the generated images
        """
        # A self-attention layer is always followed by a cross-attention layer
        # - Number of cross-attention layers = total_layers // 2 (with mod 2 == 1)
        # - Number of self-attention layers = total_layers // 2 (with mod 2 == 0)
        self.cur_step = 0
        self.cur_att_layer = 0

        self.thres = thres
        self.ref_token_idx = ref_token_idx
        self.cur_token_idx = cur_token_idx

        self.self_attns = []
        self.cross_attns = []

        self.cross_attns_mask = None
        self.self_attns_mask = None

        self.total_steps = total_steps
        self.total_attn_layers = self.MODEL_SELF_ATTN_LAYERS.get(model_type, 70) * 2 # total layers, 140 for SD-XL
        self.start_step = start_step
        self.start_layer = start_layer
        self.self_attn_layer_idx_list = self_attn_layer_idx_list if self_attn_layer_idx_list is not None else list(range(start_layer, self.total_attn_layers//2))
        self.step_idx_list = step_idx_list if step_idx_list is not None else list(range(start_step, total_steps))

        self.cross_attn_gather_down_layer = cross_attn_gather_down_layer
        self.latents_shape = latents_shape
        self.aggregation_attn_size_1 = self.latents_shape[-2]//2 * self.latents_shape[-1]//2
        self.aggregation_attn_size = self.latents_shape[-2]//(2 ** self.cross_attn_gather_down_layer) * self.latents_shape[-1]//(2 ** self.cross_attn_gather_down_layer)
        self.is_all_cross_attns = is_all_cross_attns
        
        self.external_cross_attns = external_cross_attns # for get mask from external attention map

        self.dropout = dropout # random dropout some tokens in the mask of self-attention to enhance the diversity of the generated images
        self.is_output_mask = is_output_mask

        self.mask_dict = {}

        self.is_remove_outliers = is_remove_outliers
        self.outliers_neighbor = outlier_neighbor_size
        self.is_filter_attn = is_filter_attn
        self.filter_kernel = filter_kernel
        self.gaussian_filter_kernel = torchvision.transforms.GaussianBlur(kernel_size=self.filter_kernel, sigma=0.5)
        # self.mean_filter_kernel = torch.ones((1, 1, self.filter_kernel, self.filter_kernel)) / (self.filter_kernel ** 2)

        if len(kwargs) > 0:
            print(f"Warning: get unexpected parameters: {kwargs}")

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
    
    def reset_mask_dict(self):
        self.mask_dict = {}
    
    def after_step(self): # update after each attention layer
        self.cur_att_layer += 1
        if self.cur_att_layer == self.total_attn_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            if not self.is_all_cross_attns:
                self.self_attns = []
                self.cross_attns = []

    def normalize(self, x):
        # x.shape = (B, H, W)
        x_min = x.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + 1e-8)

    def get_otsu_threshold(self, x:torch.tensor, bins:int =256):
        x = x.to(torch.float)
        # Calculate histogram
        hist = torch.histc(x, bins=bins)

        # Calculate cumulative sum
        cumsum = torch.cumsum(hist, dim=0)

        # Calculate cumulative mean
        cummean = torch.cumsum(hist * torch.arange(bins).to(hist.device), dim=0) / cumsum

        # Calculate global mean
        mean = cummean[-1]

        # Calculate between class variance
        variance = cumsum * (cumsum - bins) * (cummean - mean) ** 2

        # Find the threshold that maximizes the between class variance
        idx = torch.argmax(variance)

        # Normalize the threshold to the range [0, 1]
        threshold = idx / (bins - 1)

        return threshold
    
    def aggregate_cross_attn_map(self, idx, is_use_external=False):
        if self.external_cross_attns is not None and is_use_external:
            cross_attns = self.external_cross_attns # TODO!!!!!!!!!!!!!!!
        else:
            cross_attns = self.cross_attns

        assert len(cross_attns) > 0, "No cross attention map is found."
        attn_map = torch.stack(cross_attns, dim=1).mean(1)  # (B, N, dim)
        attn_map = attn_map.chunk(2)[-1] # attention map in conditional attn
        return self._aggregate_cross_attn_map(attn_map, idx)
        
    def _aggregate_cross_attn_map(self, attn_map, idx):
        mask_h, mask_w = self.latents_shape[-2], self.latents_shape[-1]
        ratio = (mask_h * mask_w / attn_map.shape[1]) ** 0.5
        mask_h = int(mask_h // ratio)
        mask_w = int(mask_w // ratio)

        attn_map = attn_map.reshape(-1, mask_h, mask_w, attn_map.shape[-1]) # attn_map.shape: torch.Size([2, 24, 40, 77]) (B*2, H, W, token_num)

        assert not torch.isnan(attn_map).any(), f"attn_map has nan, self.cur_step: {self.cur_step}, \
            self.cur_att_layer: {self.cur_att_layer}"
        image = attn_map[..., idx] # 0 is <sot> toke
        if isinstance(idx, list) or len(image.shape) == 4:
            image = image.mean(-1)
        if self.is_filter_attn:
            filter = self.gaussian_filter_kernel.to(image.device).to(image.dtype)
            image = filter(image.unsqueeze(0)).squeeze(0)

        image = self.normalize(image)
        return image
    
    def binarize_mask(self, self_attns_mask, spatial_mask):
        if self.thres is None or self.thres <= 0:
            threshold_source = self.get_otsu_threshold(self_attns_mask)

            threshold_target = self.get_otsu_threshold(spatial_mask)
        else:
            threshold_source = self.thres
            threshold_target = self.thres

        threshold_source = threshold_source if threshold_source < 1 else 1-1e-6
        threshold_source = threshold_source if threshold_source > 0 else 1e-6
        threshold_target = threshold_target if threshold_target < 1 else 1-1e-6
        threshold_target = threshold_target if threshold_target > 0 else 1e-6

        self_attns_mask[self_attns_mask > threshold_source] = 1
        self_attns_mask[self_attns_mask <= threshold_source] = 0
        spatial_mask[spatial_mask > threshold_target] = 1
        spatial_mask[spatial_mask <= threshold_target] = 0
        return self_attns_mask, spatial_mask
    
    # Remove outliers. A point is considered an outlier if no value=1 region exists within self.outliers_neighbor range in the binary mask.
    def remove_outliers(self, self_attns_mask, spatial_mask, device, mask_h, mask_w):
        self_attns_mask = self_attns_mask.reshape(mask_h, mask_w)
        spatial_mask = spatial_mask.reshape(mask_h, mask_w)
        # 1 means outlier, 0 means inlier
        conv_weight = torch.ones(1, 1, self.outliers_neighbor * 2 + 1, self.outliers_neighbor * 2 + 1).to(device)
        tmp_self_attns_mask = F.conv2d(self_attns_mask.unsqueeze(0).unsqueeze(0), conv_weight, padding=self.outliers_neighbor).squeeze(0).squeeze(0)
        tmp_spatial_mask = F.conv2d(spatial_mask.unsqueeze(0).unsqueeze(0), conv_weight, padding=self.outliers_neighbor).squeeze(0).squeeze(0)
        self_attns_mask = (tmp_self_attns_mask-1).clamp(0,1) * self_attns_mask
        spatial_mask = (tmp_spatial_mask-1).clamp(0,1) * spatial_mask
        self_attns_mask = self_attns_mask.flatten()
        spatial_mask = spatial_mask.reshape(-1, 1)

        return self_attns_mask, spatial_mask

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
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
        # In MasaCtrl, source and target image generation use different KV:
        # - Source generation: Q, K, V remain original (unchanged)
        # - Target generation: Q keeps original, while K and V use reference image's KV
        if is_cross_attention:
            if attention_probs.shape[1] == self.aggregation_attn_size: # for SD-XL is 32 * 32 and for SD is 16 * 16
                num_heads = attention_probs.shape[0] // batch_size
                self.cross_attns.append(attention_probs.reshape(-1, num_heads, *attention_probs.shape[-2:]).mean(1))
            
        # apply MasaCtrl
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

            hidden_states_unconditional_target = self.mask_attention(query_unconditional_target, key_unconditional_source, value_unconditional_source, 
                                                                    attn, self_attns_mask)
            hidden_states_conditional_target = self.mask_attention(query_conditional_target, key_conditional_source, value_conditional_source, 
                                                                    attn, self_attns_mask)

            hidden_states_unconditional_target_foreground, hidden_states_unconditional_target_background = hidden_states_unconditional_target.chunk(2)
            hidden_states_conditional_target_foreground, hidden_states_conditional_target_background = hidden_states_conditional_target.chunk(2)

            hidden_states_unconditional_target = hidden_states_unconditional_target_foreground * spatial_mask + hidden_states_unconditional_target_background * (1 - spatial_mask)
            hidden_states_conditional_target = hidden_states_conditional_target_foreground * spatial_mask + hidden_states_conditional_target_background * (1 - spatial_mask)

            hidden_states = torch.cat([hidden_states_unconditional_source, hidden_states_unconditional_target, 
                                        hidden_states_conditional_source, hidden_states_conditional_target], dim=0)

            del hidden_states_unconditional_source, hidden_states_unconditional_target, hidden_states_conditional_source, hidden_states_conditional_target
            del hidden_states_unconditional_target_foreground, hidden_states_unconditional_target_background, hidden_states_conditional_target_foreground, hidden_states_conditional_target_background

        else: # standard calculation
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        # original code
        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
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
                        attn, self_attns_mask, attention_mask=None):
        """
        Performing mask attention for a batch of queries, keys, and values
        """
        dtype = query.dtype
        if attn.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

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

        if self_attns_mask is not None:
            mask = self_attns_mask
            # mask.shape: torch.Size([960]), attention_scores.shape: torch.Size([20, 960, 960])
            attention_scores_foreground = attention_scores + mask.masked_fill(mask == 0, torch.finfo(attention_scores.dtype).min)
            attention_scores_background = attention_scores + mask.masked_fill(mask == 1, torch.finfo(attention_scores.dtype).min)
            attention_scores = torch.cat([attention_scores_foreground, attention_scores_background])

        Batch_size_x_head_dim, seq_len, attention_len = attention_scores.shape
        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        # attention_probs.shape: torch.Size([40, 960, 960]), value.shape: torch.Size([20, 960, 64])
        if attention_probs.shape[0] == 2 * value.shape[0]:
            value = torch.cat([value, value], dim=0).to(dtype)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # hidden_states.shape = [2, 960, 1280]

        return hidden_states
    
    @staticmethod
    def _get_attention_layers_num(net, count):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                return count + 1
            elif hasattr(net, 'children'):
                count = MasaCtrlAttnProcessor._get_attention_layers_num(subnet, count)
        return count

    def get_attention_layers_num(unet):
        attn_count = 0
        for net_name, net in unet.named_children():
            if "down" in net_name:
                attn_count += MasaCtrlAttnProcessor._get_attention_layers_num(net, 0)
            elif "mid" in net_name:
                attn_count += MasaCtrlAttnProcessor._get_attention_layers_num(net, 0)
            elif "up" in net_name:
                attn_count += MasaCtrlAttnProcessor._get_attention_layers_num(net, 0)
        print(f"attn_count: {attn_count}")
        return attn_count