import json
import os
import re
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

def get_exp_name_from_attn_processor(attn_processor_name):
    if attn_processor_name == 'DreamStoryAttnProcessor':
        return "DreamStory"
    elif attn_processor_name == 'ConsiStoryAttnProcessor':
        return "ConsiStory"
    elif attn_processor_name == 'MasaCtrlAttnProcessor':
        return "MasaCtrl"
    else:
        raise NotImplementedError(f"attn_processor_name {attn_processor_name} is not implemented yet.")
    return None

def get_word_token_idx(pipe, prompt):
    token_idx_to_word = {idx: pipe.tokenizer.decode(t, clean_up_tokenization_spaces=True)
                        for idx, t in enumerate(pipe.tokenizer(prompt)['input_ids'])
                        if 0 < idx < len(pipe.tokenizer(prompt)['input_ids']) - 1}
    return token_idx_to_word

def get_word_by_token_idx(pipe, prompt, token_idx):
    token_idx_to_word = {idx: pipe.tokenizer.decode(t)
                        for idx, t in enumerate(pipe.tokenizer(prompt)['input_ids'])
                        if 0 < idx < len(pipe.tokenizer(prompt)['input_ids']) - 1}
    assert token_idx > 0 and token_idx < len(token_idx_to_word), f"Error: token_idx should be in (0, {len(token_idx_to_word)}), but got {token_idx}"
    return token_idx_to_word[token_idx]

# Read the JSON file, obtain the prompt and token indices.
def read_test_prompts(file_path, pipe):
    assert os.path.exists(file_path), f"File {file_path} does not exist."
    with open(file_path, 'r') as f:
        ret_dict = json.load(f)
    ret_dict["prompt_list"] = []
    
    for idx, subj_dict in enumerate(ret_dict["subjects"]):
        prompt = subj_dict["prompt"]
        ret_dict["prompt_list"].append(prompt)
    ret_dict["prompt_list"].append(ret_dict["scene"]["prompt"])
    
    # get word token idx for each subjects
    for idx, subj_dict in enumerate(ret_dict["subjects"]):
        prompt = subj_dict["prompt"]
        word_token_idx = get_word_token_idx(pipe, prompt)
        
        if "word_token_idx" not in subj_dict:
            subj_dict["word_token_idx"] = {}
        subj_dict["word_token_idx"] = subj_dict["word_token_idx"]
        
        if "ref_token_idx" not in subj_dict:
            subj_dict["ref_token_idx"] = []
        for subj_word in subj_dict["subject_word"]:
            for token_idx, word in word_token_idx.items():
                if word.lower() == subj_word.lower():
                    subj_dict["ref_token_idx"].append(token_idx)
    
    # get word token idx for scene
    scene_prompt = ret_dict["scene"]["prompt"]
    word_token_idx = get_word_token_idx(pipe, scene_prompt)
    if "word_token_idx" not in ret_dict["scene"]:
        ret_dict["scene"]["word_token_idx"] = {}
    ret_dict["scene"]["word_token_idx"] = word_token_idx

    if "ref_token_idx" not in ret_dict["scene"]:
        ret_dict["scene"]["ref_token_idx"] = []
    if len(ret_dict["scene"]["ref_token_idx"]) == 0: # if not set, set it
        for tmp_subj_word_list in ret_dict["scene"]["subject_word"]:
            append_word_list = []
            for tmp_subj_word in tmp_subj_word_list:
                for token_idx, word in word_token_idx.items():
                    if word.lower() == tmp_subj_word.lower():
                        append_word_list.append(token_idx)
            ret_dict["scene"]["ref_token_idx"].append(append_word_list)

    scene_ref_token_idx_list = ret_dict["scene"]["ref_token_idx"]
    for ref_token_idx in scene_ref_token_idx_list: # Ensure all list elements have the same length.
        if len(ref_token_idx) == 0:
            print(f"Warning: ref_token_idx is empty. scene_ref_token_idx_list: {scene_ref_token_idx_list}")
            tmp_list = []
            for i in range(len(scene_ref_token_idx_list)):
                tmp_list.append([])
            ret_dict["scene"]["ref_token_idx"] = tmp_list
            break

    return ret_dict

# Since DiT(PixArt) doesn't have a set_attn_processors() function yet, we implement custom version for accommodate all methods.
def recursive_set_attn_processors(pipe, module, processor):
    if hasattr(module, "set_processor"):
        module.set_processor(processor)
    for sub_name, child in module.named_children():
        recursive_set_attn_processors(pipe, child, processor)

def set_attn_processors(pipe, attn_processor):
    if hasattr(pipe, "unet"):
        model = pipe.unet
    elif hasattr(pipe, "transformer"):
        model = pipe.transformer
    else:
        raise NotImplementedError(f"Model {pipe} is not implemented yet.")
    for name, module in model.named_children():
        recursive_set_attn_processors(pipe, module, attn_processor)

# for analysis
def save_mask(mask, image_path, output_path):
    image_np = Image.open(image_path).convert("RGBA")
    image_np = np.array(image_np)
    mask = mask.squeeze().cpu().numpy()
    np.where(mask == 0, 127, mask)
    image_np[:, :, 3] = mask
    image = Image.fromarray(image_np)
    image.save(output_path)