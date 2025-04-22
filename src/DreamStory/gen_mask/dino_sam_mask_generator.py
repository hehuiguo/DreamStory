import os, sys, fire
from typing import Tuple
import random
from PIL import Image

# Grounding DINO
import groundingdino
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict
# segment anything
from segment_anything import build_sam, SamPredictor 
import torch
import numpy as np
import cv2

from huggingface_hub import hf_hub_download

dino_src_path = os.path.dirname(os.path.dirname(os.path.dirname(groundingdino.__file__)))

def load_image_str(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def load_image_pil(image_source: Image) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu', weights_only=True)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

def load_dino_model(dino_ckpt_repo_id = "ShilongLiu/GroundingDINO", 
                dino_ckpt_filenmae = "groundingdino_swinb_cogcoor.pth", 
                dino_ckpt_config_filename = "GroundingDINO_SwinB.cfg.py", 
                device='cuda:0'):
    groundingdino_model = load_model_hf(dino_ckpt_repo_id, dino_ckpt_filenmae, dino_ckpt_config_filename, 
                            device=device)
    return groundingdino_model

def load_sam_model(sam_name="sam", device='cuda:0'):
    if sam_name.lower() == "sam":
        sam_checkpoint = os.path.join(dino_src_path, 'sam_vit_h_4b8939.pth')
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
    elif sam_name.lower() == "mobile_sam":
        MOBILE_SAM_CHECKPOINT_PATH = os.path.join(dino_src_path, "EfficientSAM/mobile_sam.pt")
        assert os.path.exists(MOBILE_SAM_CHECKPOINT_PATH), f"Please download the MobileSAM checkpoint, and place it in {MOBILE_SAM_CHECKPOINT_PATH}"
        checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
        mobile_sam = setup_mobile_sam()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=device)
        sam_predictor = SamPredictor(mobile_sam)
    elif sam_name.lower() == "hq_sam":
        HQSAM_CHECKPOINT_PATH = os.path.join(dino_src_path, "EfficientSAM/sam_hq_vit_tiny.pth")
        assert os.path.exists(HQSAM_CHECKPOINT_PATH), f"Please download the HQSAM checkpoint, and place it in {HQSAM_CHECKPOINT_PATH}"
        checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
        light_hqsam = setup_HQ_SAM()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=device)
        sam_predictor = SamPredictor(light_hqsam)
    # elif sam_name.lower() == "efficient_sam":
    #     # EFFICIENT_SAM_CHECHPOINT_PATH = "./EfficientSAM/efficient_sam_vitt.pt"
    #     # EFFICIENT_SAM_CHECHPOINT_PATH = "./EfficientSAM/efficient_sam_vits.pt"
    #     # download from https://huggingface.co/spaces/yunyangx/EfficientSAM/tree/main
    #     EFFICIENT_SAM_CHECHPOINT_PATH = os.path.join(dino_src_path, "EfficientSAM/efficientsam_s_gpu.jit")
    #     sam_predictor = torch.jit.load(EFFICIENT_SAM_CHECHPOINT_PATH)
    else:
        raise ValueError(f"Unknown sam_name: {sam_name}, only support ['sam', 'mobile_sam', 'efficient_sam']")
    return sam_predictor

def setup_mobile_sam():
    from EfficientSAM.MobileSAM.tiny_vit_sam import TinyViT
    from segment_anything.modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
    return mobile_sam


def setup_HQ_SAM():
    from EfficientSAM.LightHQSAM.tiny_vit_sam import TinyViT
    from segment_anything.modeling import MaskDecoderHQ, PromptEncoder, Sam, TwoWayTransformer
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoderHQ(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                vit_dim=160,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
    return mobile_sam

def load_model(dino_ckpt_repo_id = "ShilongLiu/GroundingDINO", 
                dino_ckpt_filenmae = "groundingdino_swinb_cogcoor.pth", 
                dino_ckpt_config_filename = "GroundingDINO_SwinB.cfg.py", 
                sam_name="sam", device='cuda:0'):
    groundingdino_model = load_dino_model(dino_ckpt_repo_id, dino_ckpt_filenmae, dino_ckpt_config_filename, device=device)

    sam_predictor = load_sam_model(sam_name, device=device)

    return groundingdino_model, sam_predictor

def get_key_from_prompts(text, phrase, count=0):
    assert phrase is not None and len(phrase) > 1, f"phrase should not be None or empty, but got {phrase}"
    class_name = text.split(".")
    class_name = [name.strip() for name in class_name if name.strip()]
    if count > 10:
        # Compare which one (phrase or class_name) is more similar and return the closer match
        best_i = -1
        best_sim = 0
        for name in class_name:
            cur_sim = 0
            for ch in name:
                if ch in phrase:
                    cur_sim += 1
            if cur_sim > best_sim:
                best_sim = cur_sim
                best_i = name
            elif cur_sim == best_sim:# Returns the shorter one
                if len(name) < best_i:
                    best_i = name
        if best_i != -1:
            return best_i
        return None

    def is_in_name(name, phrase):
        # Split name and phrase by ' '/ ',', then check if every word in phrase exists in name
        phrase_list = phrase.replace(", ", " ").split(" ")
        name_list = name.replace(", ", " ").split(" ")
        for p in phrase_list:
            if p not in name_list:
                return False
        return True
    
    for name in class_name:
        if name is None or len(name) < 1:
            continue # should not be happened
        if is_in_name(name, phrase):
            return name

    phrase_list = phrase.replace(", ", " ").split(" ")
    phrase_list = [p.strip() for p in phrase_list if p.strip()]
    # TODO: bug here, IDK why :(
    new_phrase = random.choice(phrase_list)
    return get_key_from_prompts(text, new_phrase, count+1)


def generate_sam_mask(image, text_prompt, 
                        groundingdino_model=None, sam_predictor=None, 
                        text_threshold=0.25, box_threshold=0.3, 
                        device='cuda', # TODO: the device is not set correctly due to the bug in GroundingSAM
                        is_morphology_mask=False,
                        output_mask_path=None,
                    ):
    if groundingdino_model is None or sam_predictor is None:
        groundingdino_model, sam_predictor = load_model(device=device)

    if isinstance(image, str):
        image_source_np, image = load_image_str(image) # np, tensor
    elif isinstance(image, Image):
        image_source_np, image = load_image_pil(image) # np, tensor
    else:
        raise ValueError(f"image should be a path or PIL image, but got {type(image)}")
    
    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=text_prompt, 
        box_threshold=text_threshold, 
        text_threshold=box_threshold,
        # device=device, # uncomment this line due to the bug in GroundingDINO
    )

    # set image
    sam_predictor.set_image(image_source_np)

    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source_np.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H]) # tensor: (N, 4)

    # Filter overlapping bboxes, keep only: 1. The one with highest probability; 2. The one with largest area
    new_phrases_idx_list = []
    for i, phrase_i in enumerate(phrases): # Filter bboxes that are entirely contained within other bboxes of the same class_name
        is_in_other_bbox = False
        if phrase_i is None or len(phrase_i) < 1:
            continue
        for j, phrase_j in enumerate(phrases):
            if i == j or phrase_j is None or len(phrase_j) < 1:
                continue
            # Removes bbox A if: A is entirely contained within bbox B; and both share the same class_name
            is_in_box = False
            area_size_i = (boxes_xyxy[i][2] - boxes_xyxy[i][0]) * (boxes_xyxy[i][3] - boxes_xyxy[i][1])
            area_size_insection = (min(boxes_xyxy[j][2], boxes_xyxy[i][2]) - max(boxes_xyxy[j][0], boxes_xyxy[i][0])) * (min(boxes_xyxy[j][3], boxes_xyxy[i][3]) - max(boxes_xyxy[j][1], boxes_xyxy[i][1]))

            if area_size_insection / area_size_i > 0.8:
                is_in_box = True
                iou = 1.0
            else:
                iou = box_ops.box_iou(boxes_xyxy[i:i+1], boxes_xyxy[j:j+1])[0]

            class_name_i = get_key_from_prompts(text_prompt, phrase_i)
            class_name_j = get_key_from_prompts(text_prompt, phrase_j)
            if is_in_box or (iou > 0.9 and class_name_i == class_name_j):
                # only keep the one with largest area
                area_size_j = (boxes_xyxy[j][2] - boxes_xyxy[j][0]) * (boxes_xyxy[j][3] - boxes_xyxy[j][1])
                if area_size_i < area_size_j:
                    is_in_other_bbox = True
                    break
        if not is_in_other_bbox:
            new_phrases_idx_list.append(i)

    # Calculates overlap count for each bbox
    repeat_scores = {}
    for i in new_phrases_idx_list:
        phrase_i = phrases[i]
        for j in new_phrases_idx_list:
            if i == j:
                continue
            phrase_j = phrases[j]
            iou = box_ops.box_iou(boxes_xyxy[i:i+1], boxes_xyxy[j:j+1])[0]
            if iou > 0.5:
                if i not in repeat_scores:
                    repeat_scores[i] = 1
                else:
                    repeat_scores[i] += 1

    # hen boxes of different class_name overlap significantly, retain only the highest-confidence or largest-area instance
    new_no_repeat_phrases_idx_list = []
    for i in new_phrases_idx_list:
        phrase_i = phrases[i]
        is_in_other_bbox = False
        for j in new_phrases_idx_list:
            if i == j:
                continue
            phrase_j = phrases[j]
            is_in_box = False
            area_size_i = (boxes_xyxy[i][2] - boxes_xyxy[i][0]) * (boxes_xyxy[i][3] - boxes_xyxy[i][1])
            area_size_insection = (min(boxes_xyxy[j][2], boxes_xyxy[i][2]) - max(boxes_xyxy[j][0], boxes_xyxy[i][0])) * (min(boxes_xyxy[j][3], boxes_xyxy[i][3]) - max(boxes_xyxy[j][1], boxes_xyxy[i][1]))

            if area_size_insection / area_size_i > 0.8:
                is_in_box = True
                iou = 1.0
            else:
                iou = box_ops.box_iou(boxes_xyxy[i:i+1], boxes_xyxy[j:j+1])[0]
            if iou > 0.5 or is_in_box:
                class_name_i = get_key_from_prompts(text_prompt, phrase_i)
                class_name_j = get_key_from_prompts(text_prompt, phrase_j)
                if class_name_i != class_name_j and logits[i] < logits[j]:
                    is_in_other_bbox = True
                    break
        if not is_in_other_bbox:
            new_no_repeat_phrases_idx_list.append(i)

    new_phrases_idx_list = new_no_repeat_phrases_idx_list

    # Get the highest-response bbox per class. This method is required, using prompt sentences directly as keys gives worse results
    # Note: Only two distinct subject types are currently supported. Calling get_key_from_prompts() with more types will raise an error
    best_bbox = {}
    bbox_idx = {}
    for i in new_phrases_idx_list:
        phrase = phrases[i]
        if phrase is None or len(phrase) < 1:
            continue
        class_name = get_key_from_prompts(text_prompt, phrase)
        if class_name not in best_bbox:
            best_bbox[class_name] = (logits[i], boxes_xyxy[i:i+1])
            bbox_idx[class_name] = i
        else:
            # Merge only if two bboxes have IoU > 0.5
            IOU = box_ops.box_iou(boxes_xyxy[i:i+1], best_bbox[class_name][1])[0]
            if abs(logits[i] - best_bbox[class_name][0]) < (0.2 * max(logits[i], best_bbox[class_name][0])):
                if IOU > 0.5:
                    best_x1 = min(best_bbox[class_name][1][0][0], boxes_xyxy[i][0])
                    best_y1 = min(best_bbox[class_name][1][0][1], boxes_xyxy[i][1])
                    best_x2 = max(best_bbox[class_name][1][0][2], boxes_xyxy[i][2])
                    best_y2 = max(best_bbox[class_name][1][0][3], boxes_xyxy[i][3])
                    best_logit = max(best_bbox[class_name][0], logits[i])
                    best_bbox[class_name] = (best_logit, torch.Tensor([best_x1, best_y1, best_x2, best_y2]).unsqueeze(0))
                    bbox_idx[class_name] = i
                else: # When logits are similar but not highly overlapping: 1. Keep the one with smaller repeat_score; 2. If equal, keep the larger logits 
                    if repeat_scores.get(i, 0) < repeat_scores.get(bbox_idx[class_name], 0):
                        best_bbox[class_name] = (logits[i], boxes_xyxy[i:i+1])
                        bbox_idx[class_name] = i
                    elif repeat_scores.get(i, 0) == repeat_scores.get(bbox_idx[class_name], 0):
                        if logits[i] > best_bbox[class_name][0]:
                            best_bbox[class_name] = (logits[i], boxes_xyxy[i:i+1])
                            bbox_idx[class_name] = i
            elif logits[i] > best_bbox[class_name][0]:
                best_bbox[class_name] = (logits[i], boxes_xyxy[i:i+1])
                bbox_idx[class_name] = i

    all_class_name_list = text_prompt.split(".")
    all_class_name_list = [name.strip() for name in all_class_name_list if name.strip()]

    # Check for undetected classes (may occur initially due to multi-role confusion), andassign unallocated bbox if needed
    for class_name in all_class_name_list:
        if class_name in best_bbox: # # Find the highest-probability bbox in phrase that doesn't overlap with ret_masks_dict
            continue
        else:
            best_idx = -1
            for i in range(len(phrases)):
                phrase = phrases[i]
                if phrase is None or len(phrase) < 1:
                    continue
                # 如果和best_bbox中的重叠，则继续
                area_size_i = (boxes_xyxy[i][2] - boxes_xyxy[i][0]) * (boxes_xyxy[i][3] - boxes_xyxy[i][1])
                is_in_best_bbox = False
                for _, (logit, box) in best_bbox.items():
                    area_size_insection = (min(box[0][2], boxes_xyxy[i][2]) - max(box[0][0], boxes_xyxy[i][0])) * (min(box[0][3], boxes_xyxy[i][3]) - max(box[0][1], boxes_xyxy[i][1]))
                    if area_size_insection / area_size_i > 0.8:
                        is_in_best_bbox = True
                        break
                    if box_ops.box_iou(boxes_xyxy[i:i+1], box)[0] > 0.5:
                        is_in_best_bbox = True
                        break
                if is_in_best_bbox:
                    continue
                    
                # non-overlapping bbox, add to best_bbox
                if best_idx == -1:
                    best_idx = i
                else: # only keep the one with largest area
                    bbox_best_idx = boxes_xyxy[best_idx]
                    area_size_best_idx = (bbox_best_idx[2] - bbox_best_idx[0]) * (bbox_best_idx[3] - bbox_best_idx[1])
                    if area_size_i > area_size_best_idx:
                        best_idx = i
            if best_idx != -1:
                best_bbox[class_name] = (logits[best_idx], boxes_xyxy[best_idx:best_idx+1])
                bbox_idx[class_name] = best_idx
            else:
                print(f"warning: can not find bbox for class_name: {class_name}")

    ret_masks_dict = {}
    for class_name, (logit, box) in best_bbox.items():

        with torch.no_grad():
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(box, image_source_np.shape[:2]).to(device)
            masks, _, _ = sam_predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes,
                        multimask_output = False,
                    )
        masks = masks.to("cpu").to(torch.float32) * 255 # tensor: [1, 1, 768, 1280]
        if masks.shape[0] > 1 or masks.shape[1] > 1:
            print(f"warning: masks shape should be [1, 1, H, W], but got {masks.shape}. Use the max area mask.")
            # Return the largest-area mask reshaped to [1, 1, H, W]
            mask_area = torch.sum(masks, dim=(2, 3)) # shape [n, m]
            max_area_idx = torch.argmax(mask_area)
            masks = masks[max_area_idx:max_area_idx+1]

        ret_masks_dict[class_name] = masks

    # Fill hollow noise in masks (may contain interior artifacts)
    if is_morphology_mask:
        for class_name, mask in ret_masks_dict.items():
            mask = mask.squeeze(1).squeeze(0)
            mask_np = mask.numpy().astype(np.uint8)
            # Constrain new mask within original boundary coordinates
            mask_nonzero = np.nonzero(mask_np)
            top, bottom = mask_nonzero[0].min(), mask_nonzero[0].max()
            left, right = mask_nonzero[1].min(), mask_nonzero[1].max()
            border_mask_np = np.zeros_like(mask_np)
            border_mask_np[top:bottom+1, left:right+1] = 1

            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, np.ones((128, 128), np.uint8))
            mask_np = mask_np * border_mask_np
            mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(torch.float32)
            ret_masks_dict[class_name] = mask
    if output_mask_path is not None:
        if not output_mask_path.endswith(".png") and not output_mask_path.endswith(".jpg"):
            os.makedirs(output_mask_path, exist_ok=True)
        else:
            output_mask_path = os.path.dirname(output_mask_path)
            os.makedirs(output_mask_path, exist_ok=True)
        for key in ret_masks_dict:
            mask = ret_masks_dict[key][0,:,:,:]

            mask = mask.squeeze().cpu().numpy()
            np.where(mask == 0, 127, mask)
            image = Image.fromarray(mask.astype(np.uint8))

            save_mask_path = os.path.join(output_mask_path, f"{key}.png")
            image.save(save_mask_path)

            # save the mask in the original image
            save_fused_path = os.path.join(output_mask_path, f"fuse_{key}.png")
            fused_image = image_source_np
            fused_image = cv2.cvtColor(fused_image, cv2.COLOR_BGR2BGRA)
            fused_image[:,:,3] = mask
            fused_image = Image.fromarray(fused_image)
            fused_image.save(save_fused_path)
            
        return
    else:
        return ret_masks_dict # N,1,H,W


# SAM mask post-processing steps:
# 1. Expand undersized masks to meet min_mask_pixel_size requirement
# 2. Adjust masks with deviating source_mask ratios (expand to approximate original)
# 3. Ensure non-overlapping mask outputs
def post_process_sam_mask(sam_mask, 
                            min_mask_pixel_size=256, device='cuda:0',
                            is_expand_small_mask=False, is_keep_mask_ratio=False, is_no_mask_overlap=False):
    target_mask_list = sam_mask["target_mask"]
    source_mask_list = sam_mask["source_mask"]

    source_mask_ratio_list = []
    for source_mask in source_mask_list:
        mask_np = source_mask[0,:,:].cpu().numpy()
        mask_nonzero = np.nonzero(mask_np)
        top, bottom = mask_nonzero[0].min(), mask_nonzero[0].max()
        left, right = mask_nonzero[1].min(), mask_nonzero[1].max()
        ratio = (bottom - top) / (right - left)
        source_mask_ratio_list.append(ratio)
    
    target_mask_ratio_list = []
    target_mask_sorted_idx_list = []
    for i, target_mask in enumerate(target_mask_list):
        mask_np = target_mask[0,:,:].cpu().numpy()
        mask_nonzero = np.nonzero(mask_np)
        top, bottom = mask_nonzero[0].min(), mask_nonzero[0].max()
        left, right = mask_nonzero[1].min(), mask_nonzero[1].max()
        ratio = (bottom - top) / (right - left)
        target_mask_ratio_list.append(ratio)

        pixel_sum = mask_np.sum() / 255.0
        target_mask_sorted_idx_list.append((i, pixel_sum))
    target_mask_sorted_idx_list.sort(key=lambda x: x[1]) # from small to large
    
    for (i, pixel_sum) in target_mask_sorted_idx_list:
        target_mask = target_mask_list[i]
        mask_np = target_mask[0,:,:].cpu().numpy()
        mask_nonzero = np.nonzero(mask_np)
        top, bottom = mask_nonzero[0].min(), mask_nonzero[0].max()
        left, right = mask_nonzero[1].min(), mask_nonzero[1].max()

        if is_expand_small_mask and (pixel_sum < min_mask_pixel_size * min_mask_pixel_size or \
            bottom - top < min_mask_pixel_size or right - left < min_mask_pixel_size): # expand small mask

            if bottom - top < min_mask_pixel_size or pixel_sum < min_mask_pixel_size * min_mask_pixel_size:
                center = (top + bottom) // 2
                top = max(0, center - min_mask_pixel_size // 2)
                bottom = min(mask_np.shape[0], center + min_mask_pixel_size // 2)

            if right - left < min_mask_pixel_size or pixel_sum < min_mask_pixel_size * min_mask_pixel_size:
                center = (left + right) // 2
                left = max(0, center - min_mask_pixel_size // 2)
                right = min(mask_np.shape[1], center + min_mask_pixel_size // 2)
                    
            mask_np[top:bottom+1, left:right+1] = 255

            for j in range(len(target_mask_list)): # Drop overlapping mask regions
                if i == j:
                    continue
                mask_j = target_mask_list[j][0,:,:].cpu().numpy()
                inter_mask = mask_np * mask_j
                if inter_mask.sum() > 0:
                    mask_np[inter_mask > 0] = 0

        if is_no_mask_overlap:
            for j in range(len(target_mask_list)):
                if i == j:
                    continue
                mask_j = target_mask_list[j][0,:,:].cpu().numpy()
                inter_mask = mask_np * mask_j
                if inter_mask.sum() > 0:
                    mask_np[inter_mask > 0] = 0
        
        mask = torch.tensor(mask_np).unsqueeze(0).to(device)
        sam_mask["target_mask"][i] = mask
    return sam_mask

if __name__ == "__main__":
    fire.Fire()

# examples:
# python ./src/DreamStory/gen_mask/dino_sam_mask_generator.py generate_sam_mask --image ./results/test.jpg --text_prompt "cat" --output_mask_path ./results/output_mask