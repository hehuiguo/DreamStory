"""
Evaluate the generated images of one DS-500 subset with the 4 metrics of the paper: AES / CLIP-T / DS / D&C-DS.

The scene masks used for evaluation are re-detected on the OUTPUT image with GroundingDINO + SAM, so they are
independent of the masks that drove the generation; the subject masks are the reference portrait masks of the run.

Extra dependencies:
  pip install dreamsim==0.2.1 torchmetrics jsonlines
  pip install git+https://github.com/openai/CLIP.git
Aesthetics predictor weights (sac+logos+ava1-l14-linearMSE.pth):
  https://github.com/christophschuhmann/improved-aesthetic-predictor

Usage:
  python ./src/DreamStory/eval/eval_metric.py evaluate --gen_root ./results/DS-500/IR-Diffusion/2-subj/ \
      --data_root ./results/DS-500/data/2-subj/
"""
from torchmetrics.multimodal.clip_score import CLIPScore
import torch
import torch.nn as nn
from torchvision import transforms
from dreamsim.model import dreamsim
import clip
from PIL import Image
import numpy as np
import tqdm
import jsonlines
import fire
import os
import json

from groundingdino.util.inference import predict
from groundingdino.util import box_ops
from DreamStory.gen_mask.dino_sam_mask_generator import load_model, load_image_str

IOU_MERGE = 0.5          # boxes of different keywords above this IoU are the same entity
BOX_THRESHOLD = 0.3      # same thresholds as generate_sam_mask()
TEXT_THRESHOLD = 0.25

def dreamsim_preprocess(path):
    img_size = 224
    t = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    pil_img = Image.open(path).convert('RGB')
    return t(pil_img).unsqueeze(0)

class AestheticsMLP(nn.Module):
    # The aesthetics predictor is an MLP on top of CLIP ViT-L/14 image embeddings
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def init_aesthetics_model(aesthetics_model_path="./checkpoints/sac+logos+ava1-l14-linearMSE.pth",
                          local_dir="./checkpoints/clip",
                          device="cuda:0", ):
    # CLIP image encoder
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device, download_root=local_dir)
    clip_model.eval()

    # MLP head of the aesthetics predictor
    aesthetics_model = AestheticsMLP(768)
    aesthetics_model.load_state_dict(torch.load(aesthetics_model_path))
    aesthetics_model.eval().to(device)
    print("aesthetics model loaded")
    return clip_model, clip_preprocess, aesthetics_model

def get_aesthetics_score(clip_model, clip_preprocess, aesthetics_model, images, device="cuda:0"):
    # normalized CLIP image features
    if isinstance(images, list):
        images = torch.stack([clip_preprocess(image) for image in images]).to(device)
    elif isinstance(images, torch.Tensor):
        images = images.to(device)
    else:
        images = clip_preprocess(images).to(device).unsqueeze(0)

    with torch.no_grad():
        features = clip_model.encode_image(images)
        features /= features.norm(dim=1, keepdim=True)
        features = features.to(aesthetics_model.layers[0].weight.dtype)
        aesthetics_score = aesthetics_model(features).item()
    return aesthetics_score

def get_bbox_from_mask(mask_path):
    mask_img = np.array(Image.open(mask_path))
    locs = np.where(mask_img==255)
    x0 = np.min(locs[1])
    x1 = np.max(locs[1])
    y0 = np.min(locs[0])
    y1 = np.max(locs[0])
    return (x0, x1, y0, y1)

def detect_subjects(image_path, keywords, groundingdino_model, sam_predictor):
    """Return (entity_count, {keyword: mask}) for the subjects found in image_path."""
    caption = " ".join(f"{k}." for k in keywords)
    image_source_np, image = load_image_str(image_path)
    boxes, logits, phrases = predict(model=groundingdino_model, image=image, caption=caption,
                                     box_threshold=TEXT_THRESHOLD, text_threshold=BOX_THRESHOLD)
    H, W, _ = image_source_np.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    # best box per keyword: the highest-scoring phrase that the keyword's words all appear in
    best = {}
    for i, phrase in enumerate(phrases):
        if not phrase:
            continue
        for keyword in keywords:
            if all(word in phrase.lower() for word in keyword.lower().split()):
                if keyword not in best or logits[i] > best[keyword][0]:
                    best[keyword] = (float(logits[i]), boxes_xyxy[i:i+1])

    # merge keywords whose boxes overlap: a fused subject is detected by several keywords
    found = list(best)
    parent = {k: k for k in found}
    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k
    for a_i, a in enumerate(found):
        for b in found[a_i+1:]:
            if box_ops.box_iou(best[a][1], best[b][1])[0].item() > IOU_MERGE:
                parent[root(a)] = root(b)
    entity_count = len({root(k) for k in found})

    masks = {}
    if found:
        sam_predictor.set_image(image_source_np)
        boxes_found = torch.cat([best[k][1] for k in found], dim=0)
        transformed = sam_predictor.transform.apply_boxes_torch(boxes_found, image_source_np.shape[:2]).to(sam_predictor.device)
        with torch.no_grad():
            sam_masks, _, _ = sam_predictor.predict_torch(point_coords=None, point_labels=None,
                                                          boxes=transformed, multimask_output=False)
        for i, keyword in enumerate(found):
            masks[keyword] = (sam_masks[i, 0].cpu().numpy() * 255).astype(np.uint8)
    return entity_count, masks

# Turn the generated cases under gen_root into the story_info.jsonl that cal_clip_text_img() expects.
# Masks are written as single-channel {0,255} PNGs, paths are relative to eval_root.
def gather_story_info(gen_root, data_root, eval_root, device="cuda:0"):
    groundingdino_model, sam_predictor = load_model(device=device)
    cases = sorted(d for d in os.listdir(gen_root) if os.path.exists(os.path.join(gen_root, d, "output_image_00.png")))
    assert len(cases) > 0, f"no generated cases under {gen_root}"
    print(f"{gen_root}: {len(cases)} generated cases")

    for sub_dir in ["subj_masks", "scene_masks"]:
        os.makedirs(os.path.join(eval_root, sub_dir), exist_ok=True)
    story_info_path = os.path.join(eval_root, "story_info.jsonl")
    with open(story_info_path, "w", encoding="utf-8") as f_out:
        for case in tqdm.tqdm(cases):
            case_dir = os.path.join(gen_root, case)
            with open(os.path.join(data_root, case, "final_meta", f"{case}.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            keywords = [s["subject_detection_keywords"] for s in meta["subjects"]]

            scene_image_path = os.path.join(case_dir, "output_image_00.png")
            info = {"story_name": case, "subj_prompts_list": [s["prompt"] for s in meta["subjects"]],
                    "scene_prompts_list": [meta["scene"]["prompt"]],
                    "scene_image_path_list": [os.path.relpath(scene_image_path, eval_root)],
                    "subj_image_path_list": [], "subj_mask_path_list": [], "scene_mask_path_list": []}

            for i, _ in enumerate(keywords): # reference portrait and the mask used to generate it
                subj_image_path = os.path.join(case_dir, f"original_generated_image_{i:02d}.png")
                info["subj_image_path_list"].append(os.path.relpath(subj_image_path, eval_root))
                subj_mask_path = os.path.join(eval_root, "subj_masks", f"{case}_{i:02d}.png")
                alpha = np.asarray(Image.open(os.path.join(case_dir, f"source_mask_{i:02d}.png")).convert("RGBA"))[:, :, 3]
                Image.fromarray(alpha).save(subj_mask_path)
                info["subj_mask_path_list"].append(os.path.relpath(subj_mask_path, eval_root))

            _, masks = detect_subjects(scene_image_path, keywords, groundingdino_model, sam_predictor)
            for i, keyword in enumerate(keywords):
                if keyword not in masks:
                    continue # subject missing from the output image: no scene mask, D&C-DS falls back to 0
                os.makedirs(os.path.join(eval_root, "scene_masks", case), exist_ok=True)
                scene_mask_path = os.path.join(eval_root, "scene_masks", case, f"mask_00_{i:02d}.png")
                Image.fromarray(masks[keyword]).save(scene_mask_path)
                info["scene_mask_path_list"].append(os.path.relpath(scene_mask_path, eval_root))

            f_out.write(json.dumps(info) + "\n")

    del groundingdino_model, sam_predictor
    torch.cuda.empty_cache()
    return story_info_path

def cal_clip_text_img(input_file, is_masked=False, device="cuda:0",
                      aesthetics_model_path="./checkpoints/sac+logos+ava1-l14-linearMSE.pth",
                      clip_root="./checkpoints/clip", dreamsim_root="./checkpoints/dreamsim",
                      clip_t_model="openai/clip-vit-base-patch16"):
    metric_scene_text = CLIPScore(model_name_or_path=clip_t_model).to(device)
    aesthetic_score_list = []
    dreamsim_score_list = []
    total_subj_num = 0 # number of reference subjects, subjects missing from the scene images count as 0 in DS
    DC_DS_score_list = []

    clip_model, clip_preprocess, aesthetics_model = init_aesthetics_model(aesthetics_model_path, clip_root, device=device)
    dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device=device, cache_dir=dreamsim_root)
    with open(input_file, 'rb') as f:
        for data in tqdm.tqdm(jsonlines.Reader(f)): # for every story
            subj_image_path_list = data['subj_image_path_list']
            total_subj_num += len(subj_image_path_list)

            scene_prompts_list = data['scene_prompts_list']
            scene_image_path_list = data['scene_image_path_list']

            # paths in story_info.jsonl are relative to its directory
            base_dir = os.path.dirname(input_file)
            if not os.path.exists(subj_image_path_list[0]):
                subj_image_path_list = [os.path.join(base_dir, path) for path in subj_image_path_list]
                scene_image_path_list = [os.path.join(base_dir, path) for path in scene_image_path_list]

            # obtain aesthetic score and dreamsim score for each scene
            for scene_text, scene_img_path in zip(scene_prompts_list, scene_image_path_list):
                scene_img = np.array(Image.open(scene_img_path).convert("RGB"))
                scene_img = torch.tensor(scene_img.transpose(2, 0, 1)).to(device)

                metric_scene_text.update(scene_img, scene_text)

                # obtain aesthetic score
                aesthetics_score = get_aesthetics_score(clip_model, clip_preprocess, aesthetics_model,
                                                        Image.open(scene_img_path), device=device)
                aesthetic_score_list.append(aesthetics_score)

            # One scene per story: compare each subject detected in the scene with its reference subject image
            scene_mask_path_list = data['scene_mask_path_list']
            subj_mask_path_list = data['subj_mask_path_list']
            # probe with the subject list: scene_mask_path_list is empty when no subject is detected in the scene
            if subj_mask_path_list and not os.path.exists(subj_mask_path_list[0]):
                scene_mask_path_list = [os.path.join(base_dir, path) for path in scene_mask_path_list]
                subj_mask_path_list = [os.path.join(base_dir, path) for path in subj_mask_path_list]
            DS_scene_subj = np.zeros((len(subj_image_path_list), len(subj_image_path_list)))
            for indx, scene_mask in enumerate(scene_mask_path_list): # every detected subject of the scene
                mask_name = os.path.basename(scene_mask)
                scene_id = int(mask_name.split('_')[1]) # scene id
                subj_id = int(mask_name.split('.')[0].split('_')[-1]) # subject id of this mask

                scene_path = scene_image_path_list[scene_id]
                subj_path = subj_image_path_list[subj_id]

                subj_mask = subj_mask_path_list[subj_id]

                scene_image = np.array(Image.open(scene_path).convert("RGB"))  # h, w, c
                subj_image = np.array(Image.open(subj_path).convert("RGB"))    # h, w, c

                scene_box = get_bbox_from_mask(scene_mask)
                subj_box = get_bbox_from_mask(subj_mask)

                if is_masked:
                    mask_img = np.array(Image.open(scene_mask))
                    mask_img = np.expand_dims(mask_img, axis=-1)
                    scene_image = scene_image * (mask_img/255)
                    scene_image = scene_image.astype(np.uint8)

                    mask_img = np.array(Image.open(subj_mask))
                    mask_img = np.expand_dims(mask_img, axis=-1)
                    subj_image = subj_image * (mask_img/255)
                    subj_image = subj_image.astype(np.uint8)

                scene_image = scene_image[scene_box[2]:scene_box[3], scene_box[0]:scene_box[1], :]
                subj_image = subj_image[subj_box[2]:subj_box[3], subj_box[0]:subj_box[1], :]

                # DreamSim similarity (1 - distance), the higher the better
                dreamsim_scene_image = dreamsim_preprocess(Image.fromarray(scene_image)).to(device)
                dreamsim_subj_image = dreamsim_preprocess(Image.fromarray(subj_image)).to(device)

                dreamsim_distance = dreamsim_model(dreamsim_scene_image, dreamsim_subj_image)
                dreamsim_score_list.append(1-dreamsim_distance.item()) # save dreamsim similarity

                # Calculate the DS similarity between this subject and other reference subjects
                for subj_i, subj_img_path_i in enumerate(subj_image_path_list):
                    subj_img_i = np.array(Image.open(subj_img_path_i).convert("RGB"))
                    subj_mask_i = subj_mask_path_list[subj_i]
                    subj_bbox_i = get_bbox_from_mask(subj_mask_i)

                    if is_masked:
                        mask_img = np.array(Image.open(subj_mask_i))
                        mask_img = np.expand_dims(mask_img, axis=-1)
                        subj_img_i = subj_img_i * (mask_img/255)
                        subj_img_i = subj_img_i.astype(np.uint8)

                    subj_img_i = subj_img_i[subj_bbox_i[2]:subj_bbox_i[3], subj_bbox_i[0]:subj_bbox_i[1], :]
                    subj_img_i = dreamsim_preprocess(Image.fromarray(subj_img_i)).to(device)
                    dreamsim_distance = dreamsim_model(dreamsim_scene_image, subj_img_i)
                    DS_scene_subj[subj_id, subj_i] = 1 - dreamsim_distance.item()

            # (from MuDi) Compute D&C-DS similarity
            # Method: Calculate DS similarity between each pair of reference character images to get a similarity matrix.
            # Then, compute DS similarity between each scene character image and reference character images to get another similarity matrix.
            # Finally, subtract these two matrices and compute the spectral norm (ord=2). 1 minus it gives the final D&C-DS similarity.
            if len(subj_image_path_list) != len(scene_mask_path_list):
                # subject number mismatch: D&C-DS is set to 0, following MuDI
                DC_DS_score_list.append(0)
            else:
                DS_ref_subj = np.zeros((len(subj_image_path_list), len(subj_image_path_list)))
                for subj_i, subj_img_path_i in enumerate(subj_image_path_list):
                    for subj_j, subj_img_path_j in enumerate(subj_image_path_list):
                        if subj_i == subj_j:
                            DS_ref_subj[subj_i, subj_j] = 1
                        else:
                            subj_img_i = np.array(Image.open(subj_img_path_i).convert("RGB"))
                            subj_img_j = np.array(Image.open(subj_img_path_j).convert("RGB"))

                            # get subj_mask
                            subj_mask_i = subj_mask_path_list[subj_i]
                            subj_mask_j = subj_mask_path_list[subj_j]

                            # get bbox
                            subj_bbox_i = get_bbox_from_mask(subj_mask_i)
                            subj_bbox_j = get_bbox_from_mask(subj_mask_j)

                            if is_masked:
                                mask_img = np.array(Image.open(subj_mask_i))
                                mask_img = np.expand_dims(mask_img, axis=-1)
                                subj_img_i = subj_img_i * (mask_img/255)
                                subj_img_i = subj_img_i.astype(np.uint8)

                                mask_img = np.array(Image.open(subj_mask_j))
                                mask_img = np.expand_dims(mask_img, axis=-1)
                                subj_img_j = subj_img_j * (mask_img/255)
                                subj_img_j = subj_img_j.astype(np.uint8)

                            # crop img
                            subj_img_i = subj_img_i[subj_bbox_i[2]:subj_bbox_i[3], subj_bbox_i[0]:subj_bbox_i[1], :]
                            subj_img_j = subj_img_j[subj_bbox_j[2]:subj_bbox_j[3], subj_bbox_j[0]:subj_bbox_j[1], :]

                            subj_img_i = dreamsim_preprocess(Image.fromarray(subj_img_i)).to(device)
                            subj_img_j = dreamsim_preprocess(Image.fromarray(subj_img_j)).to(device)
                            dreamsim_distance = dreamsim_model(subj_img_i, subj_img_j)
                            DS_ref_subj[subj_i, subj_j] = 1 - dreamsim_distance.item()
                            DS_ref_subj[subj_j, subj_i] = DS_ref_subj[subj_i, subj_j] # symmetric matrix

                DC_DS_score = 1 - np.linalg.norm(DS_ref_subj - DS_scene_subj, ord=2)
                DC_DS_score_list.append(DC_DS_score)

    # the 4 metrics reported in the paper
    return {"AES": np.mean(aesthetic_score_list),
            "CLIP-T": metric_scene_text.score / metric_scene_text.n_samples,
            "DS": np.sum(dreamsim_score_list) / total_subj_num,
            "D&C-DS": np.mean(DC_DS_score_list)}

# Evaluate one generated subset: gen_root/<case>/output_image_00.png, data_root/<case>/final_meta/<case>.json.
# Results are written to {gen_root}/eval/metric.txt
def evaluate(gen_root="./results/DS-500/IR-Diffusion/2-subj/", data_root="./results/DS-500/data/2-subj/",
             is_masked=False, device="cuda:0", **model_paths):
    eval_root = os.path.join(gen_root, "eval")
    with torch.no_grad():
        story_info_path = gather_story_info(gen_root, data_root, eval_root, device=device)
        result_dict = cal_clip_text_img(story_info_path, is_masked=is_masked, device=device, **model_paths)

    output_file = os.path.join(eval_root, "metric.txt")
    with open(output_file, 'w') as f:
        for key, value in result_dict.items():
            value = float(value)
            print(f"{key} : {value:.4f}")
            f.write(f"{key} : {value:.4f}\n")
    print(f"metrics saved to {output_file}")

if __name__ == '__main__':
    fire.Fire()
