import re
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    MllamaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

from src.utils.clinical_guideline import (
    BIRADS_CLINICAL_GUIDELINE,
    BIRADS_CLINICAL_GUIDELINE_MG,
    BIRADS_DIAGNOSTIC_GUIDELINE_MG,
    BIRADS_DIAGNOSTIC_GUIDELINE_US,
    BIRDS_FIELD_GUIDE,
)
from src.utils.dataloaders import named_concept_bank


SUPPORTED_DATASETS = {"BREAST_US", "BUSBRA", "DDSM", "CUB"}
SUPPORTED_PROMPTS = {"guidelines", "concepts_only", "reasoning"}


def normalize_species_name(raw_name: str) -> str:
    name = re.sub(r"^\d+\.", "", raw_name)
    name = name.replace("_", " ")
    name = re.sub(
        r"\b([A-Z][a-z]+) (footed|winged|capped|crowned|necked|throated|tailed|backed)\b",
        r"\1-\2",
        name,
    )
    return " ".join([w.capitalize() if w.islower() else w for w in name.split()])


def _dataset_context(metadata: Dict) -> Tuple[str, str, str, str]:
    dataset = metadata["dataset"]
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Use one of {sorted(SUPPORTED_DATASETS)}")

    birads = metadata.get("birads")
    species_name = metadata.get("species_name")

    auxiliary = ""
    if dataset in ("BREAST_US", "BUSBRA") and birads is not None:
        birads_value = {0: 2, 1: 3, 2: 4, 3: 5}[int(birads)]
        auxiliary = f"Its BI-RADS category is {birads_value}."

    if dataset == "DDSM" and birads is not None:
        roi_id = metadata.get("roi_id", "")
        birads_value = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}[int(birads)]
        abnormality_type = "mass" if "mass" in roi_id.lower() else "calcification"
        which_breast = "left" if "left" in roi_id.lower() else "right"
        image_view = "CC" if "cc" in roi_id.lower() else "MLO"
        auxiliary = (
            f"The image shows a {abnormality_type} in the {which_breast} breast, in the {image_view} view. "
            f"Its BI-RADS category is {birads_value}."
        )

    if dataset == "CUB" and species_name:
        species_name = normalize_species_name(species_name)
        auxiliary = f"The image shows a bird of the species {species_name}."

    modality = {
        "BREAST_US": "breast ultrasound",
        "BUSBRA": "breast ultrasound",
        "DDSM": "mammography",
        "CUB": "natural image",
    }[dataset]

    guideline = {
        "BREAST_US": BIRADS_CLINICAL_GUIDELINE,
        "BUSBRA": BIRADS_CLINICAL_GUIDELINE,
        "DDSM": BIRADS_CLINICAL_GUIDELINE_MG,
        "CUB": BIRDS_FIELD_GUIDE[species_name] if species_name else '',
    }[dataset]

    diagnostic_guideline = {
        "BREAST_US": BIRADS_DIAGNOSTIC_GUIDELINE_US,
        "BUSBRA": BIRADS_DIAGNOSTIC_GUIDELINE_US,
        "DDSM": BIRADS_DIAGNOSTIC_GUIDELINE_MG,
        "CUB": "",
    }[dataset]

    return modality, auxiliary, guideline, diagnostic_guideline


def _concept_data(metadata: Dict) -> str:
    dataset = metadata["dataset"]
    concepts = metadata["concepts"]
    selected_concepts = metadata.get("selected_concepts", list(range(len(concepts))))
    concept_bank = named_concept_bank(dataset=dataset)

    lines = ["Finally, you are given the following 'concepts' that are present in the image."]
    if dataset in {"BREAST_US", "BUSBRA", "DDSM"}:
        lines.append("These concepts ultimately determine the BI-RADS category of the image.")

    for idx in selected_concepts:
        if idx < len(concepts) and int(concepts[idx]) == 1:
            concept_name = concept_bank[idx].capitalize().replace("_", " ")
            lines.append(f"{concept_name}: {concepts[idx]}")

    return "\n".join(lines)


def _build_prompts(metadata: Dict) -> Dict[str, str]:
    modality, auxiliary, guideline, diagnostic_guideline = _dataset_context(metadata)
    concept_data = _concept_data(metadata)

    prompt_with_guidelines = f"""You are given the following {modality} <image>. {auxiliary}
You are also given the following guideline:

{guideline}

{concept_data}

Write a professional and concise report based on the image, guideline, and concepts.
"""

    prompt_with_concepts = f"""You are given the following {modality} <image>.
You are also given the following 'concepts' that are present in the image.

{concept_data}

Write a professional and concise report based on the image and concepts.
"""

    reasoning_prompt = f"""You are given the following {modality} <image>.
You are tasked with detecting concepts present in the image.
Given the extracted concepts, explain implications for final diagnosis based on the guideline below.
Analyze concept alignment with prediction, infer likely category, and give follow-up recommendation.

{diagnostic_guideline}
"""

    return {
        "guidelines": prompt_with_guidelines,
        "concepts_only": prompt_with_concepts,
        "reasoning": reasoning_prompt,
    }


def _to_pil_rgb(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = np.stack([x] * 3, axis=-1)
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = np.transpose(x, (1, 2, 0))
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 1) if x.max() <= 1.0 else np.clip(x, 0, 255)
            x = (x * 255).astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)
        return Image.fromarray(x).convert("RGB")
    raise TypeError(type(x))


class Qwen:
    def __init__(self, size=3, min_px=256, max_px=1280, device="cuda"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/model-weights/Qwen2.5-VL-7B-Instruct",
            torch_dtype="auto",
            device_map="auto",
        ).to(device)
        self.device = device

        min_pixels = min_px * 28 * 28
        max_pixels = max_px * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    def run(self, image, metadata, prompt_mode="guidelines"):
        prompts = _build_prompts(metadata)
        prompt = prompts[prompt_mode if prompt_mode in SUPPORTED_PROMPTS else "guidelines"]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image, "name": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=768)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text


class Llama:
    def __init__(self, size=3, min_px=256, max_px=1280, device="cuda"):
        model_name = "/model-weights/Llama-3.2-11B-Vision-Instruct/"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device)
        self.device = device

        min_pixels = min_px * 28 * 28
        max_pixels = max_px * 28 * 28
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

    def run(self, image, metadata, prompt_mode="guidelines"):
        prompts = _build_prompts(metadata)
        prompt = prompts[prompt_mode if prompt_mode in SUPPORTED_PROMPTS else "guidelines"]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image, "name": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=768)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text


class MedGemma:
    def __init__(self, size=4, min_px=256, max_px=1280, device="cuda"):
        model_name = "/model-weights/medgemma-4b-it"
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
        )

    def run(self, image, metadata, prompt_mode="guidelines"):
        prompts = _build_prompts(metadata)
        prompt = prompts[prompt_mode if prompt_mode in SUPPORTED_PROMPTS else "guidelines"]

        image = _to_pil_rgb(image)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]
        generation = self.model.generate(
            **inputs,
            max_new_tokens=768,
            min_new_tokens=64,
            do_sample=False,
            max_length=input_len + 768,
        )

        gen_tokens = generation[:, input_len:]
        out = self.processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        return out


class LVLMGenerator(nn.Module):
    def __init__(self, lvlm_name="qwen", size=3, min_px=256, max_px=1280, prompt_mode="guidelines", device="cuda"):
        super(LVLMGenerator, self).__init__()
        if lvlm_name == "qwen":
            self.llama = Qwen(size=size, min_px=min_px, max_px=max_px, device=device)
        elif lvlm_name == "llama":
            self.llama = Llama(size=size, min_px=min_px, max_px=max_px, device=device)
        elif lvlm_name == "medgemma":
            self.llama = MedGemma(size=size, min_px=min_px, max_px=max_px, device=device)
        else:
            raise ValueError(f"Unknown LVLM model name: {lvlm_name}")

        self.prompt_mode = prompt_mode

    def generate_clinical_report(self, image, metadata):
        if isinstance(image, list):
            if len(image) == 1:
                return self.llama.run(image[0], metadata, self.prompt_mode)
            return [self.llama.run(img, metadata, self.prompt_mode) for img in image]
        return self.llama.run(image, metadata, self.prompt_mode)

    def forward(self, images, metadata_list):
        outputs = []
        for i in range(len(images)):
            outputs.append(self.generate_clinical_report(images[i], metadata_list[i]))
        return outputs
