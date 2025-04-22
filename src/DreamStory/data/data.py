import os, sys, fire
from tqdm import tqdm
import fire
import json

from fuzzywuzzy import process, fuzz

from diffusers import DiffusionPipeline

from DreamStory.utils.tools import get_word_token_idx

def get_model(model_path):
    model = DiffusionPipeline.from_pretrained(model_path)
    model = model
    return model

def find_description(scene_prompt, character_short_prompt):
    words = scene_prompt.split()
    
    # generate all possible sub-sentences
    sub_sentences = []
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            sub_sentences.append(" ".join(words[i:j]))

    best_match = process.extractOne(character_short_prompt, sub_sentences, scorer=fuzz.token_sort_ratio)[0]
    return best_match

# Pre-process DS-500 benchmark to meta_file, which can be used for inferenece by DreamStory
def prepare_data(input_meta_file_path = "./DS-500/Synthetic-Cases-400/2-subject_meta.json",
                    output_meta_root = "./results/DS-500_2-subj_benchmark/"):
    os.makedirs(output_meta_root, exist_ok=True)
    pipe = get_model("playgroundai/playground-v2.5-1024px-aesthetic") # can be SDXL or other UNet-based models

    # Read the JSON file at input_meta_file_path and convert it into a Python object.
    assert os.path.exists(input_meta_file_path), f"input_meta_file_path: {input_meta_file_path} not exists"
    with open(input_meta_file_path, "r") as f:
        input_meta_dict = json.load(f)

    scene_prompts_list = input_meta_dict["scene_prompts_list"]
    scene_character_list = input_meta_dict["scene_character_list"]
    character_type_list = input_meta_dict["character_type_list"]
    character_short_prompts_list = input_meta_dict["character_short_prompts_list"]
    character_prompts_list = input_meta_dict["character_prompts_list"]

    for scene_idx, scene_prompt in enumerate(scene_prompts_list):
        scene_prompt = scene_prompt[0] # it should be only one scene prompt in this sample
        scene_character = scene_character_list[scene_idx]
        scene_character_type_list = []
        for character_idx in scene_character:
            scene_character_type_list.append(character_type_list[character_idx])

        # Each scene is saved as a JSON file named f"{scene_idx:02}_{char_type1}_{char_type2}.json".
        # The JSON contains three keys: test_case_name, subjects, and scene, where:
        # test_case_name matches the filename; subjects and scene are dictionaries storing subject and scene information.
        test_case_name = f"{scene_idx:02}_{'_'.join(scene_character_type_list)}"
        output_scene_root = os.path.join(output_meta_root, test_case_name)
        os.makedirs(output_scene_root, exist_ok=True)
        output_scene_meta_root = os.path.join(output_scene_root, "final_meta")
        os.makedirs(output_scene_meta_root, exist_ok=True)

        output_meta_file_path = os.path.join(output_scene_meta_root, f"{test_case_name}.json")
        output_meta_dict = { "test_case_name": test_case_name, "subjects": [], "scene": {} }

        match_str_list = []
        # The 'subjects' key contains a list, where each item in the list is a dictionary.
        for character_idx in scene_character:
            character_type = character_type_list[character_idx]
            character_short_prompt = character_short_prompts_list[character_idx]
            character_prompt = character_prompts_list[character_idx]
            word_token_idx = get_word_token_idx(pipe=pipe, prompt=character_prompt)

            output_meta_dict["subjects"].append({
                "subject_name": f"{character_idx:02}_{character_type}",
                "prompt": character_prompt,
                "subject_detection_keywords": character_type,
                "word_token_idx": word_token_idx,
                "subject_word": [character_type],
                "ref_token_idx": []
            })

            match_string = find_description(scene_prompt.lower(), character_short_prompt)
            match_str_list.append(match_string)

        # The 'scene' key contains a dictionary that stores scene information. Let me know if you need any refinements!
        scene_ref_token_idx = []
        word_token_idx = get_word_token_idx(pipe=pipe, prompt=scene_prompt)
        for match_string in match_str_list:
            match_token_idx = get_word_token_idx(pipe=pipe, prompt=match_string)
            tmp_ref_token_idx = []
            # Find the position of match_string in scene_prompt tokens by identifying its range using the first and last words. 
            # Since word_token_idx is a dictionary storing each token's position, the location of match_string can be directly determined.
            for token_idx in range(1, len(word_token_idx)+1):
                if word_token_idx[token_idx] == match_token_idx[1]:
                    start_token_idx = token_idx # start token index
                    is_match = True
                    for i in range(1, len(match_token_idx)):
                        if word_token_idx[token_idx + i] != match_token_idx[i+1]:
                            is_match = False
                            break
                    if is_match:
                        end_token_idx = token_idx + len(match_token_idx) - 1
                        for i in range(start_token_idx, end_token_idx + 1):
                            tmp_ref_token_idx.append(i)
                        break
            assert len(tmp_ref_token_idx) > 1, f"tmp_ref_token_idx is empty, match_string: {match_string}, scene_prompt: {scene_prompt}, short_prompts: {character_short_prompt}"
            scene_ref_token_idx.append(tmp_ref_token_idx)


        output_meta_dict["scene"] = {
            "prompt": scene_prompt,
            "word_token_idx": word_token_idx,
            "subject_word": [[character_type] for character_type in scene_character_type_list],
            "ref_token_idx": scene_ref_token_idx
        }
        
        with open(output_meta_file_path, "w") as f:
            json.dump(output_meta_dict, f, indent=4)

if __name__ == '__main__':
    fire.Fire()

# Example usage:

# python ./src/DreamStory/data/data.py prepare_data 
# python ./src/DreamStory/data/data.py prepare_data --input_meta_file_path ./DS-500/Synthetic-Cases-400/2-subject_meta.json --output_meta_root ./results/DS-500_2-subj_benchmark/
# python ./src/DreamStory/data/data.py prepare_data --input_meta_file_path ./DS-500/Synthetic-Cases-400/3-subject_meta.json --output_meta_root ./results/DS-500_3-subj_benchmark/