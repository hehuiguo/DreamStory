import os, sys, fire
import traceback
import random
import torch
from pytorch_lightning import seed_everything

# load diffusers API
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import DiffusionPipeline

# load model API
from DreamStory.attention_processor import get_attn_processor_by_name
from DreamStory.gen_mask.dino_sam_mask_generator import load_model, generate_sam_mask, post_process_sam_mask
from DreamStory.utils.tools import read_test_prompts, set_attn_processors, get_word_token_idx, save_mask

# solution from https://stackoverflow.com/questions/78827482/cant-suppress-warning-from-transformers-src-transformers-modeling-utils-py
import logging
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers: # suppress transformers logging
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)


def get_model(model, device):
    if isinstance(model, str):
        model = DiffusionPipeline.from_pretrained(model)
        model = model.to(device).to(torch.bfloat16)
    return model

def update_new_seed(seed, try_count, min=100, max=1000): # re-gen a new start code
    new_seed = seed + try_count * random.randint(100, 1000)
    return new_seed

def test(prompts_path="./results/examples/example_dog_boy.json", 
        model="playgroundai/playground-v2.5-1024px-aesthetic", output_root = "./result/example_debug/", seed=-1, device="cuda:0",
        width=1280, height=768, num_inference_steps=50, num_images_per_prompt=1, guidance_scale=7.0, 
        start_step=0, start_layer=36, dropout=0.5, mutual_cross_attention_lambda=0.9, ref_sa_lambda=0,
        is_spatial_self_attn=True, is_mutual_cross_attn=True, is_dropout_ca=False,
        is_rescale_self_attn=True, is_isolate_sa=True,
        is_DIFT=False, sam_step=50, style="",
        attn_processor_name="DreamStoryScaledAttnProcessor", 
        groundingdino_model=None, sam_predictor=None, sam_name="sam", 
        is_expand_small_mask=True, is_keep_mask_ratio=False, is_no_mask_overlap=True, is_morphology_mask=True,
        is_overwrite=False, retry_num=20, is_output_mask=False,
    ):
    pipe = get_model(model, device)  if isinstance(model, str) else model
    test_dict = read_test_prompts(prompts_path, pipe)

    os.makedirs(output_root, exist_ok=True)
    output_image_path = os.path.join(output_root, "output_image.png")
    if os.path.exists(output_image_path) and not is_overwrite:
        return

    test_prompts = test_dict["prompt_list"]
    if style is not None and len("style") > 0:
        test_prompts = [f"{p} {style}" for p in test_prompts]

    # save prompts and word token idx 
    with open(os.path.join(output_root, "prompts.txt"), "w", encoding='utf-8') as f:
        for p in test_prompts:
            f.write(f"{p}\n")
        for p_i, p in enumerate(test_prompts):
            word_token_idx = get_word_token_idx(pipe, p)
            f.write(f"word token idx of prompt {p_i} : {word_token_idx}\n")
    
    output_param_path = os.path.join(output_root, "params.txt")
    with open(output_param_path, "w", encoding='utf-8') as f: # save all param
        locals_param = locals()
        for k, v in locals_param.items():
            f.write(f"{k}: {v}\n")

    # noise start code
    if seed >= 0:
        seed_everything(seed)
    shape = (1, 4, height // 8, width // 8) # in_channel = 4
    noise_start_code = torch.randn(shape).to(pipe.device).to(torch.bfloat16)
    consis_start_code = noise_start_code.expand(len(test_prompts)-1, -1, -1, -1)

    if len(test_dict["subjects"]) == 0: # start to generate original images (without any control)
        for p_i, p in enumerate(test_prompts):
            original_image_output_path = os.path.join(output_root, f"original_generated_image_{p_i:02d}.png")
            if os.path.exists(original_image_output_path) and not is_overwrite:
                continue
            if seed >= 0:
                seed_everything(seed)
            set_attn_processors(pipe, AttnProcessor2_0())
            original_images = pipe(
                prompt=p, width=width, height=height, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                num_images_per_prompt=1, latents=consis_start_code[p_i:p_i+1, ...],
            ).images
            original_images[0].save(original_image_output_path)
        return 0 
    
    # begin to generate images with XXXCtrl
    if seed >= 0:
        seed_everything(seed)
    set_attn_processors(pipe, AttnProcessor2_0())

    # rehearsal generate original images for SAM mask
    original_images = pipe(
        prompt=test_prompts[:-1], width=width, height=height, guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps, num_images_per_prompt=1,
        latents=noise_start_code.expand(len(test_prompts)-1, -1, -1, -1),
    ).images

    for i, img in enumerate(original_images):
        original_image_path = os.path.join(output_root, f"original_generated_image_{i:02d}.png")
        if not os.path.exists(original_image_path):
            img.save(original_image_path)
    
    sam_mask = {}
    sam_mask["source_mask"], sam_mask["target_mask"] = [], []
    if groundingdino_model is None or sam_predictor is None:
        groundingdino_model, sam_predictor = load_model(sam_name=sam_name)
    
    # generate mask for reference subject portrait
    scene_detect_prompt = ""
    for i, subj_dict in enumerate(test_dict["subjects"]):
        original_image_path = os.path.join(output_root, f"original_generated_image_{i:02d}.png")
        detect_prompt = subj_dict["subject_detection_keywords"]
        detect_prompt = detect_prompt + "." if detect_prompt[-1] != "." else detect_prompt
        mask_dict = generate_sam_mask(image=original_image_path, text_prompt=detect_prompt, 
                                        groundingdino_model=groundingdino_model, sam_predictor=sam_predictor)
        mask = mask_dict[detect_prompt[:-1]] # discard '.' in the end
        sam_mask["source_mask"].append(mask[0,:,:,:]) # mask.shape: [N, 1, H, W]
        scene_detect_prompt += detect_prompt if len(scene_detect_prompt) == 0 else f" {detect_prompt}" # 拼接所有的detect_prompt, 用于检测场景

        # save sam mask
        original_image_path = os.path.join(output_root, f"original_generated_image_{i:02d}.png")
        reference_mask_image_path = os.path.join(output_root, f"source_mask_{i:02d}.png")
        save_mask(sam_mask["source_mask"][i], original_image_path, reference_mask_image_path)
    
    # try to get sam mask:
    repeat_count = 0
    new_seed = seed
    while repeat_count < num_images_per_prompt:
        if seed >= 0:
            seed_everything(new_seed)
        noise_start_code = torch.randn(shape).to(pipe.device).to(torch.bfloat16)
        original_scene_path = os.path.join(output_root, f"original_scene_image_{repeat_count:02d}.png")
        repeat_count += 1

        # prepare ref_token_idx and cur_token_idx for attn_processor
        ref_token_idx = []
        for subj_dict in test_dict["subjects"]:
            ref_token_idx.append(subj_dict["ref_token_idx"])
        cur_token_idx = test_dict["scene"]["ref_token_idx"]

        for idx_list in test_dict["scene"]["ref_token_idx"]:
            if len(idx_list) == 0:
                sam_step = num_inference_steps + 2
                break

        try_count = 0
        while try_count < retry_num:
            set_attn_processors(pipe, AttnProcessor2_0()) # reset attn_processor
            is_need_retry = False
            sam_mask["target_mask"] = []
            scene_images = pipe(
                prompt=test_prompts[-1], width=width, height=height, guidance_scale=guidance_scale, 
                num_inference_steps=num_inference_steps, num_images_per_prompt=1,
                latents=noise_start_code,
            ).images
            scene_images[-1].save(original_scene_path)
            try:
                # generate mask for target scene
                mask_dict = generate_sam_mask(image=original_scene_path, text_prompt=scene_detect_prompt,
                                                groundingdino_model=groundingdino_model, sam_predictor=sam_predictor, 
                                                is_morphology_mask=is_morphology_mask)
                class_name_list = scene_detect_prompt.split(".")
                class_name_list = [name.strip() for name in class_name_list if name.strip()]
                for class_name in class_name_list:
                    if class_name in mask_dict:
                        mask = mask_dict[class_name][0,:,:,:]
                        mask_np = mask.cpu().squeeze().numpy()
                        mask = torch.tensor(mask_np).unsqueeze(0).to(device)
                        sam_mask["target_mask"].append(mask)
                    else:
                        sam_mask["target_mask"].append(None)
                        if try_count != retry_num - 1:
                            is_need_retry = True
                            break
                        else:
                            continue

                new_seed = update_new_seed(new_seed, try_count+1)
                if is_need_retry:
                    try_count += 1
                    print(f"Warning: get sam mask error, retry {try_count} times")
                    with open(os.path.join(output_root, f"sam_mask_retry_{repeat_count}_{try_count}.log"), "w", encoding='utf-8') as f:
                        f.write(f"retry {try_count} times\n")

                    if seed >= 0:
                        seed_everything(new_seed)
                    noise_start_code = torch.randn(shape).to(pipe.device).to(torch.bfloat16)
                    continue
                else:
                    # save sam mask before post process
                    for i in range(len(sam_mask["target_mask"])): #  save sam mask with scene image
                        target_mask = sam_mask["target_mask"][i]
                        if target_mask is None:
                            continue # should not happen
                        target_mask_image_path = os.path.join(output_root, f"target_mask_bf_{i:02d}.png")
                        save_mask(target_mask, original_scene_path, target_mask_image_path)

                    # post process sam mask
                    sam_mask = post_process_sam_mask(sam_mask, is_expand_small_mask=is_expand_small_mask, is_keep_mask_ratio=is_keep_mask_ratio, 
                                                    is_no_mask_overlap=is_no_mask_overlap, device=device)

                    for i in range(len(sam_mask["target_mask"])): #  save sam mask with scene image
                        target_mask = sam_mask["target_mask"][i]
                        if target_mask is None:
                            continue # should not happen
                        target_mask_image_path = os.path.join(output_root, f"target_mask_{i:02d}.png")
                        save_mask(target_mask, original_scene_path, target_mask_image_path)
                    break

            except Exception as e: # only for debug
                try_count += 1
                print(f"Warning: get sam mask error: {e}")
                sam_mask["source_mask"] = [] if len(sam_mask["source_mask"]) < len(test_prompts) - 1 else sam_mask["source_mask"]
                sam_mask["target_mask"] = [] if len(sam_mask["target_mask"]) < len(test_prompts) - 1 else sam_mask["target_mask"]
                with open(os.path.join(output_root, "sam_mask_error.log"), "w", encoding='utf-8') as f:
                    traceback.print_exc(file=f) # save log to file for debug
            finally:
                pass

        for i in range(len(sam_mask["source_mask"])):
            source_mask = sam_mask["source_mask"][i]
            if source_mask is not None:
                source_mask_np = source_mask.cpu().squeeze().numpy()
                source_mask_np[source_mask_np.shape[0]//6*5:, :] = 0
                source_mask = torch.tensor(source_mask_np).unsqueeze(0).to(device)
                sam_mask["source_mask"][i] = source_mask
        
        set_attn_processors(pipe, 
            get_attn_processor_by_name(name=attn_processor_name,
                    start_step=start_step, start_layer=start_layer, dropout=dropout,
                    latents_shape=(noise_start_code.shape[-2], noise_start_code.shape[-1]),
                    ref_token_idx=ref_token_idx, cur_token_idx=cur_token_idx,
                    sam_mask=sam_mask, sam_step=sam_step, 
                    is_spatial_self_attn=is_spatial_self_attn, is_mutual_cross_attn=is_mutual_cross_attn, 
                    is_rescale_self_attn=is_rescale_self_attn, is_isolate_sa=is_isolate_sa,
                    mutual_cross_attention_lambda=mutual_cross_attention_lambda, 
                    is_dropout_ca=is_dropout_ca, ref_sa_lambda=ref_sa_lambda,
                    is_DIFT=is_DIFT, is_output_mask=is_output_mask,
                    )
            )

        final_consis_start_code = torch.concatenate((consis_start_code, noise_start_code), dim=0)
        Story_images = pipe(
            prompt=test_prompts, width=width, height=height, guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps, num_images_per_prompt=1,
            latents=final_consis_start_code, 
            # TODO: sam_mask=sam_mask, sam_step=sam_step, is_DIFT=is_DIFT,
        ).images

        output_image_path = os.path.join(output_root, f"output_image_{repeat_count-1:02d}.png")
        Story_images[-1].save(output_image_path) # save the scene image
        for i in range(len(sam_mask["target_mask"])): # save sam mask with scene image
            target_mask_image_path = os.path.join(output_root, f"output_mask_{i:02d}.png")
            target_mask = sam_mask["target_mask"][i]
            if target_mask is not None:
                save_mask(target_mask, output_image_path, target_mask_image_path)
        
        if is_output_mask:
            output_mask_root = os.path.join(output_root, f"step-{start_step}_layer-{start_layer}")
            os.makedirs(output_mask_root, exist_ok=True)
            attn_processor = list(pipe.unet.attn_processors.values())[0]
            mask_dict = attn_processor.mask_dict
            assert len(mask_dict) > 0, "mask_dict should not be empty."
            attn_processor.reset_mask_dict()

if __name__ == "__main__": # test the pipeline
    fire.Fire()

# Examples:
# CUDA_VISIBLE_DEVICES=0 python ./src/DreamStory/pipe/pipe_test.py test --prompts_path="./results/examples/example.json" --output_root="./results/example_debug/" 
