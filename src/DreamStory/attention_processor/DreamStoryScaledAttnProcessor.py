
from typing import Optional
import torch
from torch.nn import functional as F
from diffusers.models.attention_processor import Attention

from .DreamStoryAttnProcessor import DreamStoryAttnProcessor

class DreamStoryScaledAttnProcessor(DreamStoryAttnProcessor):
    def __init__(self, start_step=4, start_layer=54, self_attn_layer_idx_list=None, step_idx_list=None, total_steps=50, 
                thres=0, ref_token_idx=[1], cur_token_idx=[1], model_type="SDXL",
                cross_attn_gather_down_layer=2, latents_shape=(768//8, 1280//8),
                dropout=0, is_output_mask=False, sam_mask=None, sam_step=40, 
                is_spatial_self_attn=True, is_mutual_cross_attn=True, 
                is_dropout_ca=False, mutual_cross_attention_lambda=0.9, 
                is_rescale_self_attn=True, is_isolate_sa=True, ref_sa_lambda=0, 
                is_DIFT=False,
                 **kwargs):
        super().__init__(start_step=start_step, start_layer=start_layer, self_attn_layer_idx_list=self_attn_layer_idx_list, 
                        step_idx_list=step_idx_list, total_steps=total_steps,
                        thres=thres, ref_token_idx=ref_token_idx, cur_token_idx=cur_token_idx,
                        model_type=model_type, cross_attn_gather_down_layer=cross_attn_gather_down_layer, latents_shape=latents_shape,
                        dropout=dropout, is_output_mask=is_output_mask, sam_mask=sam_mask, sam_step=sam_step,
                        is_spatial_self_attn=is_spatial_self_attn, is_mutual_cross_attn=is_mutual_cross_attn, 
                        is_dropout_ca=is_dropout_ca, mutual_cross_attention_lambda=mutual_cross_attention_lambda,
                        is_DIFT=is_DIFT,)
        self.ref_sa_lambda = ref_sa_lambda
        self.is_rescale_self_attn = is_rescale_self_attn
        self.is_isolate_sa = is_isolate_sa

    '''
    把reference的hidden_states，根据self_attns_mask和spatial_mask，resize到target image的大小。
    它是将reference hidden_states的特征，resize后粘贴到target hidden_states上,。
    这个新的hidden_states由mutual self attention计算
    '''
    def rescale_hidden_states_sa(self, hidden_states, self_attns_mask_list, spatial_mask_list, inter_mode="nearest"):
        # hidden_states.shape: torch.Size([6, 3840, 640])
        scale_ratio = (self.latents_shape[0] * self.latents_shape[1] // hidden_states.shape[1]) ** 0.5
        mask_h, mask_w = int(self.latents_shape[0] // scale_ratio), int(self.latents_shape[1] // scale_ratio)

        hidden_states_unconditional, hidden_states_conditional = hidden_states.chunk(2, dim=0)
        hidden_states_unconditional_list = hidden_states_unconditional.chunk(self.subj_num+1, dim=0)
        hidden_states_conditional_list = hidden_states_conditional.chunk(self.subj_num+1, dim=0)

        ret_hidden_states_unconditional_list = []
        ret_hidden_states_conditional_list = []
        for subj_idx in range(self.subj_num):
            self_attns_mask_i = self_attns_mask_list[subj_idx].reshape(mask_h, mask_w)
            spatial_mask_i = spatial_mask_list[subj_idx].reshape(mask_h, mask_w)
            if spatial_mask_i.sum() == 0:
                ret_hidden_states_unconditional_list.append(hidden_states_unconditional_list[subj_idx])
                ret_hidden_states_conditional_list.append(hidden_states_conditional_list[subj_idx])
                continue

            # get bbox of the target image
            target_nonzero_idx = torch.nonzero(spatial_mask_i, as_tuple=True)
            target_left, target_right = target_nonzero_idx[1].min(), target_nonzero_idx[1].max()
            target_top, target_bottom = target_nonzero_idx[0].min(), target_nonzero_idx[0].max()
            assert target_left <= target_right and target_top <= target_bottom, \
                    f"target_left: {target_left}, target_right: {target_right}, target_top: {target_top}, target_bottom: {target_bottom}"
            ref_nonzero_idx = torch.nonzero(self_attns_mask_i, as_tuple=True)
            ref_left, ref_right = ref_nonzero_idx[1].min(), ref_nonzero_idx[1].max()
            ref_top, ref_bottom = ref_nonzero_idx[0].min(), ref_nonzero_idx[0].max()
            assert ref_left < ref_right and ref_top < ref_bottom, \
                    f"ref_left: {ref_left}, ref_right: {ref_right}, ref_top: {ref_top}, ref_bottom: {ref_bottom}"
            B, HW, C = hidden_states_unconditional_list[-1].shape
            target_hidden_states_unconditional_i = hidden_states_unconditional_list[-1].transpose(1, 2).reshape(B, C, mask_h, mask_w)
            target_hidden_states_conditional_i = hidden_states_conditional_list[-1].transpose(1, 2).reshape(B, C, mask_h, mask_w)
            ref_hidden_states_unconditional_i = hidden_states_unconditional_list[subj_idx].transpose(1, 2).reshape(B, C, mask_h, mask_w) * self_attns_mask_i.unsqueeze(0)
            ref_hidden_states_conditional_i = hidden_states_conditional_list[subj_idx].transpose(1, 2).reshape(B, C, mask_h, mask_w) * self_attns_mask_i.unsqueeze(0)

            ret_hidden_states_unconditional_i = torch.zeros_like(target_hidden_states_unconditional_i)
            ret_hidden_states_conditional_i = torch.zeros_like(target_hidden_states_conditional_i)
            ret_hidden_states_unconditional_i[:,:, target_top:target_bottom+1, target_left:target_right+1] = F.interpolate(ref_hidden_states_unconditional_i[:, :, ref_top:ref_bottom+1, ref_left:ref_right+1].to(torch.float32), 
                                                                                                        (target_bottom-target_top+1, target_right-target_left+1), mode=inter_mode).to(target_hidden_states_unconditional_i.dtype)
            ret_hidden_states_conditional_i[:,:, target_top:target_bottom+1, target_left:target_right+1] = F.interpolate(ref_hidden_states_conditional_i[:, :, ref_top:ref_bottom+1, ref_left:ref_right+1].to(torch.float32),
                                                                                                        (target_bottom-target_top+1, target_right-target_left+1), mode=inter_mode).to(target_hidden_states_conditional_i.dtype)

            ret_hidden_states_unconditional_i = ret_hidden_states_unconditional_i.reshape(B, C, -1).transpose(1, 2)
            ret_hidden_states_conditional_i = ret_hidden_states_conditional_i.reshape(B, C, -1).transpose(1, 2)

            ret_hidden_states_unconditional_list.append(ret_hidden_states_unconditional_i)
            ret_hidden_states_conditional_list.append(ret_hidden_states_conditional_i)

        ret_hidden_states_unconditional_list.append(hidden_states_unconditional_list[-1])
        ret_hidden_states_conditional_list.append(hidden_states_conditional_list[-1])

        hidden_states_unconditional = torch.cat(ret_hidden_states_unconditional_list, dim=0)
        hidden_states_conditional = torch.cat(ret_hidden_states_conditional_list, dim=0)
        hidden_states = torch.cat([hidden_states_unconditional, hidden_states_conditional], dim=0)
        return hidden_states

    '''
    把target的hidden_states，根据self_attns_mask和spatial_mask，resize到reference image的大小。
    它是将target hidden_states的特征，resize后粘贴到reference hidden_states上,。
    然后这个新的hidden_states经过正常的cross attention计算后，再把对应位置的特征resize回target image中角色的大小。
    最后进行加权求和。
    '''
    def rescale_hidden_states_ca(self, hidden_states, self_attns_mask_list, spatial_mask_list, inter_mode="nearest"):
        # hidden_states.shape: torch.Size([6, 3840, 640])
        scale_ratio = (self.latents_shape[0] * self.latents_shape[1] // hidden_states.shape[1]) ** 0.5
        mask_h, mask_w = int(self.latents_shape[0] // scale_ratio), int(self.latents_shape[1] // scale_ratio)

        hidden_states_unconditional, hidden_states_conditional = hidden_states.chunk(2, dim=0)
        hidden_states_unconditional_list = hidden_states_unconditional.chunk(self.subj_num+1, dim=0)
        hidden_states_conditional_list = hidden_states_conditional.chunk(self.subj_num+1, dim=0)

        ret_hidden_states_unconditional_list = []
        ret_hidden_states_conditional_list = []
        for subj_idx in range(self.subj_num):
            self_attns_mask_i = self_attns_mask_list[subj_idx].reshape(mask_h, mask_w)
            spatial_mask_i = spatial_mask_list[subj_idx].reshape(mask_h, mask_w)

            if spatial_mask_i.sum() == 0:
                ret_hidden_states_unconditional_list.append(hidden_states_unconditional_list[subj_idx])
                ret_hidden_states_conditional_list.append(hidden_states_conditional_list[subj_idx])
                continue

            # the bbox of the target image
            target_nonzero_idx = torch.nonzero(spatial_mask_i, as_tuple=True)
            target_left, target_right = target_nonzero_idx[1].min(), target_nonzero_idx[1].max()
            target_top, target_bottom = target_nonzero_idx[0].min(), target_nonzero_idx[0].max()
            assert target_left <= target_right and target_top <= target_bottom, \
                    f"target_left: {target_left}, target_right: {target_right}, target_top: {target_top}, target_bottom: {target_bottom}"
            ref_nonzero_idx = torch.nonzero(self_attns_mask_i, as_tuple=True)
            ref_left, ref_right = ref_nonzero_idx[1].min(), ref_nonzero_idx[1].max()
            ref_top, ref_bottom = ref_nonzero_idx[0].min(), ref_nonzero_idx[0].max()
            assert ref_left < ref_right and ref_top < ref_bottom, \
                    f"ref_left: {ref_left}, ref_right: {ref_right}, ref_top: {ref_top}, ref_bottom: {ref_bottom}"

            B, HW, C = hidden_states_unconditional_list[-1].shape
            hidden_states_unconditional_i = hidden_states_unconditional_list[-1].transpose(1, 2).reshape(B, C, mask_h, mask_w) * spatial_mask_i.unsqueeze(0).unsqueeze(0)
            hidden_states_conditional_i = hidden_states_conditional_list[-1].transpose(1, 2).reshape(B, C, mask_h, mask_w) * spatial_mask_i.unsqueeze(0).unsqueeze(0)
            ref_hidden_states_unconditional_i = hidden_states_unconditional_list[subj_idx].transpose(1, 2).reshape(B, C, mask_h, mask_w)
            ref_hidden_states_conditional_i = hidden_states_conditional_list[subj_idx].transpose(1, 2).reshape(B, C, mask_h, mask_w)

            ret_hidden_states_unconditional_i = torch.zeros_like(ref_hidden_states_unconditional_i)
            ret_hidden_states_conditional_i = torch.zeros_like(ref_hidden_states_conditional_i)
            ret_hidden_states_unconditional_i[:,:, ref_top:ref_bottom+1, ref_left:ref_right+1] = F.interpolate(hidden_states_unconditional_i[:, :, target_top:target_bottom+1, target_left:target_right+1].to(torch.float32), 
                                                                                                        (ref_bottom-ref_top+1, ref_right-ref_left+1), mode=inter_mode).to(hidden_states_unconditional_i.dtype)
            ret_hidden_states_conditional_i[:,:, ref_top:ref_bottom+1, ref_left:ref_right+1] = F.interpolate(hidden_states_conditional_i[:, :, target_top:target_bottom+1, target_left:target_right+1].to(torch.float32),
                                                                                                        (ref_bottom-ref_top+1, ref_right-ref_left+1), mode=inter_mode).to(hidden_states_conditional_i.dtype)

            ret_hidden_states_unconditional_i = ret_hidden_states_unconditional_i.reshape(B, C, -1).transpose(1, 2)
            ret_hidden_states_conditional_i = ret_hidden_states_conditional_i.reshape(B, C, -1).transpose(1, 2)

            ret_hidden_states_unconditional_list.append(ret_hidden_states_unconditional_i)
            ret_hidden_states_conditional_list.append(ret_hidden_states_conditional_i)

        ret_hidden_states_unconditional_list.append(hidden_states_unconditional_list[-1])
        ret_hidden_states_conditional_list.append(hidden_states_conditional_list[-1])

        hidden_states_unconditional = torch.cat(ret_hidden_states_unconditional_list, dim=0)
        hidden_states_conditional = torch.cat(ret_hidden_states_conditional_list, dim=0)
        hidden_states = torch.cat([hidden_states_unconditional, hidden_states_conditional], dim=0)
        return hidden_states
    
    def rescale_hidden_states_ca_reverse(self, hidden_states, self_attns_mask, spatial_mask, inter_mode="nearest"):
        # hidden_states.shape: torch.Size([1, 3840, 640]) 
        # spatial_mask.shape: torch.Size([3840, 1]), self_attns_mask.shape: torch.Size([3840]) 
        scale_ratio = (self.latents_shape[0] * self.latents_shape[1] // hidden_states.shape[1]) ** 0.5
        mask_h, mask_w = int(self.latents_shape[0] // scale_ratio), int(self.latents_shape[1] // scale_ratio)

        # get bbox
        spatial_mask = spatial_mask.reshape(mask_h, mask_w)
        ref_mask = self_attns_mask.reshape(mask_h, mask_w)

        if spatial_mask.sum() == 0:
            return hidden_states
        target_nonzero_idx = torch.nonzero(spatial_mask, as_tuple=True)
        target_left, target_right = target_nonzero_idx[1].min(), target_nonzero_idx[1].max()
        target_top, target_bottom = target_nonzero_idx[0].min(), target_nonzero_idx[0].max()
        assert target_left <= target_right and target_top <= target_bottom, \
                f"target_left: {target_left}, target_right: {target_right}, target_top: {target_top}, target_bottom: {target_bottom}"
        ref_nonzero_idx = torch.nonzero(ref_mask, as_tuple=True)
        ref_left, ref_right = ref_nonzero_idx[1].min(), ref_nonzero_idx[1].max()
        ref_top, ref_bottom = ref_nonzero_idx[0].min(), ref_nonzero_idx[0].max()
        assert ref_left < ref_right and ref_top < ref_bottom, \
                f"ref_left: {ref_left}, ref_right: {ref_right}, ref_top: {ref_top}, ref_bottom: {ref_bottom}"
        # resize hidden_states to (mask_h, mask_w) according to the bbox
        B, HW, C = hidden_states.shape
        # transfer to shape of [B，C，HW]，than reshape to shape of [B，C，H，W]
        hidden_states = hidden_states.transpose(1, 2).reshape(B, C, mask_h, mask_w)
        ret_hidden_states = torch.zeros_like(hidden_states)
        ret_hidden_states[:,:, target_top:target_bottom+1, target_left:target_right+1] = F.interpolate(hidden_states[:, :, ref_top:ref_bottom+1, ref_left:ref_right+1].to(torch.float32),
                                                                                                    (target_bottom-target_top+1, target_right-target_left+1), mode=inter_mode).to(hidden_states.dtype)

        ret_hidden_states = ret_hidden_states.reshape(B, C, -1).transpose(1, 2)
        return ret_hidden_states

    # perform normal SA while ensuring that the subjects do not interfere with each other.
    def isolate_subj_sa(self, attn, query, key):
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=attn.scale,
        )
        del baddbmm_input

        attention_probs = self._isolate_subj_sa(query, attention_scores)

        attention_probs = attention_scores.softmax(dim=-1)
        return attention_probs

    def _isolate_subj_sa(self, query, attention_scores):
        scale_ratio = (self.latents_shape[0] * self.latents_shape[1] // query.shape[1]) ** 0.5
        mask_h, mask_w = int(self.latents_shape[0] // scale_ratio), int(self.latents_shape[1] // scale_ratio)
        spatial_mask_list = self.spatial_mask_dict[f"original_{mask_h}x{mask_w}"]

        # attention_scores.shape: torch.Size([60, 3840, 3840]), query.shape: torch.Size([60, 3840, 64]), key.shape: torch.Size([60, 3840, 64]) 
        B, dim1, dim2 = attention_scores.shape
        min_value = torch.finfo(attention_scores.dtype).min
        if dim1 == dim2: # For U-Net encoder: no mutual attention, only standard self attention.
            attention_scores_list = list(attention_scores.chunk((self.subj_num+1)*2, dim=0))
            attention_scores_unconditional_target = attention_scores_list[self.subj_num]
            attention_scores_conditional_target = attention_scores_list[-1]
            for subj_i in range(self.subj_num):
                spatial_mask_i = spatial_mask_list[subj_i] # shape: torch.Size([3840, 1])

                for subj_j in range(self.subj_num):
                    if subj_i != subj_j:
                        spatial_mask_j = spatial_mask_list[subj_j]
                        # spatial_mask_i.shape: torch.Size([3840, 1]), spatial_mask_j.shape: torch.Size([3840, 1])print(f"spatial_mask_i.shape: {spatial_mask_i.shape}, spatial_mask_j.shape: {spatial_mask_j.shape}")
                        mask =  spatial_mask_i @ spatial_mask_j.squeeze(-1).unsqueeze(0) # mask.shape: torch.Size([3840, 3840])
                        mask = mask.masked_fill(mask != 0, min_value).unsqueeze(0)
                        attention_scores_unconditional_target += mask
                        attention_scores_conditional_target += mask

            attention_scores_list[self.subj_num] = attention_scores_unconditional_target
            attention_scores_list[-1] = attention_scores_conditional_target
            attention_scores = torch.cat(attention_scores_list, dim=0)
        else:
            for subj_i in range(self.subj_num):
                spatial_mask_i = spatial_mask_list[subj_i]

                for subj_j in range(self.subj_num):
                    if subj_i != subj_j:
                        spatial_mask_j = spatial_mask_list[subj_j]
                        mask = spatial_mask_i @ spatial_mask_j.squeeze(-1).unsqueeze(0) # mask.shape: torch.Size([3840, 3840])
                        mask = mask.masked_fill(mask != 0, min_value).unsqueeze(0)
                        attention_scores[:, :, self.subj_num*dim1:(self.subj_num+1)*dim1] += mask

        return attention_scores

    def relocate_mask(self, src_mask, target_mask, is_reshape=False, mask_h=None, mask_w=None, inter_mode="nearest"):
        # src_mask.shape: torch.Size([3840, 1]), target_mask.shape: torch.Size([3840])
        if src_mask.sum() == 0 or target_mask.sum() == 0:
            return src_mask
        if is_reshape:
            original_shape = src_mask.shape
            assert mask_h is not None and mask_w is not None, "mask_h and mask_w should be provided"
            src_mask = src_mask.reshape(mask_h, mask_w)
            target_mask = target_mask.reshape(mask_h, mask_w)

        src_nonzero_idx = torch.nonzero(src_mask, as_tuple=True)
        src_left, src_right = src_nonzero_idx[1].min(), src_nonzero_idx[1].max()
        src_top, src_bottom = src_nonzero_idx[0].min(), src_nonzero_idx[0].max()
        target_nonzero_idx = torch.nonzero(target_mask, as_tuple=True)
        target_left, target_right = target_nonzero_idx[1].min(), target_nonzero_idx[1].max()
        target_top, target_bottom = target_nonzero_idx[0].min(), target_nonzero_idx[0].max()

        ret_mask = torch.zeros_like(target_mask)
        ret_mask[target_top:target_bottom+1, target_left:target_right+1] = F.interpolate(src_mask[src_top:src_bottom+1, src_left:src_right+1].to(torch.float32).unsqueeze(0).unsqueeze(0),
                                                                                        (target_bottom-target_top+1, target_right-target_left+1), mode=inter_mode).to(target_mask.dtype).squeeze(0).squeeze(0)

        ret_mask = (ret_mask > 0).to(target_mask.dtype)
        if is_reshape:
            ret_mask = ret_mask.reshape(original_shape)

        return ret_mask

    def self_attention(self,
        attn: Attention,
        residual: torch.Tensor,
        hidden_states: torch.FloatTensor,
        attention_probs: Optional[torch.FloatTensor] = None,
        query = None, key = None, value = None,
        self_attns_mask_list = None,
        spatial_mask_list = None,
        dropout_self_attns_mask_list = None,
        dropout_spatial_mask_list = None,
        scale: float = 1.0,
        DIFT_sim = None,
    ) -> torch.Tensor:
        # args = () if USE_PEFT_BACKEND else (scale,)
        args = ()

        if (self.cur_step in self.step_idx_list and self.cur_att_layer // 2 in self.self_attn_layer_idx_list and self.is_spatial_self_attn):
            if self.is_rescale_self_attn:
                rescale_hidden_states = self.rescale_hidden_states_sa(hidden_states, self_attns_mask_list, spatial_mask_list)
                rescale_key = attn.to_k(rescale_hidden_states, *args)
                rescale_value = attn.to_v(rescale_hidden_states, *args)
                rescale_key = attn.head_to_batch_dim(rescale_key)
                rescale_value = attn.head_to_batch_dim(rescale_value)
            else:
                rescale_key = key
                rescale_value = value

            query_unconditional, query_conditional = query.chunk(2, dim=0)
            key_unconditional, key_conditional = key.chunk(2, dim=0)
            value_unconditional, value_conditional = value.chunk(2, dim=0)
            rescale_key_unconditional, rescale_key_conditional = rescale_key.chunk(2, dim=0)
            rescale_value_unconditional, rescale_value_conditional = rescale_value.chunk(2, dim=0)
            attention_probs_unconditional, attention_probs_conditional = attention_probs.chunk(2, dim=0)

            query_unconditional_list = query_unconditional.chunk(self.subj_num+1, dim=0)
            key_unconditional_list = key_unconditional.chunk(self.subj_num+1, dim=0)
            value_unconditional_list = value_unconditional.chunk(self.subj_num+1, dim=0)
            rescale_key_unconditional_list = rescale_key_unconditional.chunk(self.subj_num+1, dim=0)
            rescale_value_unconditional_list = rescale_value_unconditional.chunk(self.subj_num+1, dim=0)
            attention_probs_unconditional_list = attention_probs_unconditional.chunk(self.subj_num+1, dim=0)

            query_conditional_list = query_conditional.chunk(self.subj_num+1, dim=0)
            key_conditional_list = key_conditional.chunk(self.subj_num+1, dim=0)
            value_conditional_list = value_conditional.chunk(self.subj_num+1, dim=0)
            rescale_key_conditional_list = rescale_key_conditional.chunk(self.subj_num+1, dim=0)
            rescale_value_conditional_list = rescale_value_conditional.chunk(self.subj_num+1, dim=0)
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
                spatial_mask_all = torch.cat(dropout_spatial_mask_list + [torch.ones_like(dropout_spatial_mask_list[-1])], dim=0)

            if self.is_rescale_self_attn:
                mask_attn_key_unconditional = torch.cat(rescale_key_unconditional_list, dim=0)
                mask_attn_key_conditional = torch.cat(rescale_key_conditional_list, dim=0)
                mask_attn_value_unconditional = torch.cat(rescale_value_unconditional_list, dim=0)
                mask_attn_value_conditional = torch.cat(rescale_value_conditional_list, dim=0)
                self_attns_mask_all = [] # TODO
                scale_ratio = (self.latents_shape[0] * self.latents_shape[1] // query.shape[1]) ** 0.5
                mask_h, mask_w = int(self.latents_shape[0] // scale_ratio), int(self.latents_shape[1] // scale_ratio)
                for mask_idx, self_attns_mask_i in enumerate(self_attns_mask_list):
                    spatial_mask_i = spatial_mask_list[mask_idx]
                    rescale_self_attns_mask_i = self.relocate_mask(self_attns_mask_i, spatial_mask_i.squeeze(-1), 
                            is_reshape=True, mask_h=mask_h, mask_w=mask_w)
                    self_attns_mask_all.append(rescale_self_attns_mask_i)
                self_attns_mask_all = torch.cat(self_attns_mask_all + [torch.ones_like(self_attns_mask_list[-1])], dim=0)

            else:
                mask_attn_key_unconditional = torch.cat(key_unconditional_list, dim=0)
                mask_attn_key_conditional = torch.cat(key_conditional_list, dim=0)
                mask_attn_value_unconditional = torch.cat(value_unconditional_list, dim=0)
                mask_attn_value_conditional = torch.cat(value_conditional_list, dim=0)
                self_attns_mask_all = torch.cat(dropout_self_attns_mask_list + [torch.ones_like(dropout_self_attns_mask_list[-1])], dim=0)

            hidden_states_unconditional_target = self.mask_attention_multi_subj(query_unconditional_list[-1], 
                                                    mask_attn_key_unconditional, mask_attn_value_unconditional,
                                                    attn, self_attns_mask_all,  spatial_mask_all)
            hidden_states_conditional_target = self.mask_attention_multi_subj(query_conditional_list[-1],
                                                    mask_attn_key_conditional, mask_attn_value_conditional,
                                                    attn, self_attns_mask_all,  spatial_mask_all)
            hidden_states_unconditional_list.append(hidden_states_unconditional_target)
            hidden_states_conditional_list.append(hidden_states_conditional_target)
            hidden_states = torch.cat(hidden_states_unconditional_list + hidden_states_conditional_list, dim=0) # shape: [2*(subj_num+1), 960, 1280]
        else:
            if self.is_isolate_sa:
                attention_probs = self.isolate_subj_sa(attn, query, key)
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
        self_attns_mask_list = None,
        spatial_mask_list = None,
        dropout_self_attns_mask_list = None,
        dropout_spatial_mask_list = None,
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
        
        if self.is_mutual_cross_attn:
            query_unconditional, query_conditional = query.chunk(2, dim=0)
            query_unconditional_list = query_unconditional.chunk(self.subj_num+1, dim=0)
            query_conditional_list = query_conditional.chunk(self.subj_num+1, dim=0)

            # mask attention的方式
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

            mutual_hidden_states_unconditional_target = torch.cat(mutual_hidden_states_unconditional_target_list, dim=0)
            mutual_hidden_states_conditional_target = torch.cat(mutual_hidden_states_conditional_target_list, dim=0)
            mutual_hidden_states = torch.cat([mutual_hidden_states_unconditional_target, mutual_hidden_states_conditional_target], dim=0)
            # mutual_hidden_states.shape = [subj_num*2, H*W, dim]: ([4, 3840, 640])
                
            # linear proj
            mutual_hidden_states = attn.to_out[0](mutual_hidden_states, *args)
            # dropout
            mutual_hidden_states = attn.to_out[1](mutual_hidden_states)

            # fuse mutual_hidden_states to hidden_states before residual connection
            mutual_hidden_states_unconditional, mutual_hidden_states_conditional = mutual_hidden_states.chunk(2, dim=0)
            hidden_states_unconditional, hidden_states_conditional = hidden_states.chunk(2, dim=0)
            
            hidden_states_unconditional_list = list(hidden_states_unconditional.chunk(self.subj_num+1, dim=0))
            hidden_states_conditional_list = list(hidden_states_conditional.chunk(self.subj_num+1, dim=0))
            mutual_hidden_states_unconditional_list = list(mutual_hidden_states_unconditional.chunk(self.subj_num, dim=0))
            mutual_hidden_states_conditional_list = list(mutual_hidden_states_conditional.chunk(self.subj_num, dim=0))

            vanilla_hidden_states_unconditional_target = hidden_states_unconditional_list[-1]
            vanilla_hidden_states_conditional_target = hidden_states_conditional_list[-1]
            
            # solution 1: average maked area in the hidden_states_unconditional_target
            masked_area = torch.zeros_like(dropout_spatial_mask_list[-1]) # shape = [H*W, 1] : [3840, 1]
            hidden_states_unconditional_target = torch.zeros_like(vanilla_hidden_states_conditional_target)
            hidden_states_conditional_target = torch.zeros_like(vanilla_hidden_states_conditional_target)
            mutual_ca_spatial_mask_list = dropout_spatial_mask_list if self.is_dropout_ca else spatial_mask_list
            for subj_idx in range(self.subj_num): # TODO: accelerate
                masked_area += mutual_ca_spatial_mask_list[subj_idx] # get masked hidden_states and its erea
                hidden_states_unconditional_target += mutual_hidden_states_unconditional_list[subj_idx] * mutual_ca_spatial_mask_list[subj_idx]
                hidden_states_conditional_target += mutual_hidden_states_conditional_list[subj_idx] * mutual_ca_spatial_mask_list[subj_idx]

            hidden_states_unconditional_target = hidden_states_unconditional_target / (masked_area + 1e-8)
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
    
    def isolate_subj_ca(self, query, key, value, attn, spatial_mask_list):

        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=attn.scale,
        )
        del baddbmm_input

        # attention_probs.shape: torch.Size([60, 3840, 77]), query.shape: torch.Size([60, 3840, 64]), key.shape: torch.Size([60, 77, 64]), value.shape: torch.Size([60, 77, 64])

        attention_scores_list = list(attention_scores.chunk((self.subj_num+1)*2, dim=0))
        attention_scores_unconditional_target = attention_scores_list[self.subj_num]
        attention_scores_conditional_target = attention_scores_list[-1]
        for subj_i in range(self.subj_num):
            spatial_mask_i = spatial_mask_list[subj_i]
            for cur_token_idx_i in self.cur_token_idx[subj_i]:
                attention_scores_unconditional_target[:, :, cur_token_idx_i:cur_token_idx_i+1] += torch.where(spatial_mask_i == 0, torch.finfo(attention_scores.dtype).min, 0).unsqueeze(0).to(attention_scores.dtype).to(attention_scores.device)
                attention_scores_conditional_target[:, :, cur_token_idx_i:cur_token_idx_i+1] += torch.where(spatial_mask_i == 0, torch.finfo(attention_scores.dtype).min, 0).unsqueeze(0).to(attention_scores.dtype).to(attention_scores.device)
        attention_scores_list[self.subj_num] = attention_scores_unconditional_target
        attention_scores_list[-1] = attention_scores_conditional_target
        attention_scores = torch.cat(attention_scores_list, dim=0)
        attention_probs = attention_scores.softmax(dim=-1)
        return attention_probs
    
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

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            assert 1==3, f"didn't implement this branch yet"
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

        query_dim = query.shape[1]
        key_dim = key.shape[1]
        is_cross_attention = (key_dim != query_dim)
        
        scale_ratio = (self.latents_shape[0] * self.latents_shape[1] // query.shape[1]) ** 0.5
        mask_h, mask_w = int(self.latents_shape[0] // scale_ratio), int(self.latents_shape[1] // scale_ratio) # assume the shape of mask_target is the same as the mask_source
        dropout_self_attns_mask_list, dropout_spatial_mask_list, self_attns_mask_list, spatial_mask_list = \
            self.gen_binary_mask(mask_h, mask_w, is_cross_attention, query.device, query.dtype)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        is_target_sam_mask = not (self.sam_mask is None or self.sam_mask["target_mask"] is None or len(self.sam_mask["target_mask"]) < self.subj_num)

        # gather attention
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
                                                self_attns_mask_list=self_attns_mask_list, spatial_mask_list=spatial_mask_list, 
                                                dropout_self_attns_mask_list=dropout_self_attns_mask_list, dropout_spatial_mask_list=dropout_spatial_mask_list,
                                                scale=scale, DIFT_sim=DIFT_sim)
        else:
            hidden_states = self.cross_attention(attn=attn, residual=residual, hidden_states=hidden_states, attention_mask=attention_mask,
                                                attention_probs=attention_probs, query=query, key=key, value=value, 
                                                self_attns_mask_list=self_attns_mask_list, spatial_mask_list=spatial_mask_list, 
                                                dropout_self_attns_mask_list=dropout_self_attns_mask_list, dropout_spatial_mask_list=dropout_spatial_mask_list,
                                                scale=scale)
            
        self.after_step() # update the step and layer index after every attention layer

        return hidden_states

    def mask_attention_fg(self, query, key, value, 
                    attn, self_attns_mask, spatial_mask=None, attention_mask=None):
        """
        这里实现cross attention的foreground mask attention。其中query是target image的query, key和value是source image的。
        目的是让target image的query只关注source image的text。
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
        min_value = torch.finfo(attention_scores.dtype).min
        zero_tensor = torch.tensor(0, dtype=attention_scores.dtype).to(attention_scores.device)
        if spatial_mask is not None:
            attention_scores = attention_scores + torch.where(spatial_mask == 0, min_value, zero_tensor).to(attention_scores.device)

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
                    attn, self_attns_mask, spatial_mask=None, attention_mask=None):
        """
        这里实现mask attention。其中query是target image的query, key和value是source image与target image拼接起来的。
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
        attention_scores = torch.cat(attention_scores.chunk(self.subj_num+1, dim=0), dim=-1) # attention_scores[B, dim, (subj_num+1)*dim] torch.Size([20, 960, 2880])
        B, N, _ = attention_scores.shape
        attention_scores[:,:,:N*self.subj_num] += self.ref_sa_lambda

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

        if self.is_isolate_sa:
            attention_probs = self._isolate_subj_sa(query=query, attention_scores=attention_scores)
        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        # attention_probs.shape: torch.Size([20, 960, 960*(subj_num+1)]), value.shape: torch.Size([20*(subj_num+1), 960, 64])
        value = torch.cat(value.chunk(self.subj_num+1, dim=0), dim=1)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # hidden_states.shape = torch.Size([1, 960, 1280])

        return hidden_states