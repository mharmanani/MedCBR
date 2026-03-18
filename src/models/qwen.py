import torch
import torch.nn as nn
import numpy as np

from src.utils.clinical_guideline import (
    BIRADS_CLINICAL_GUIDELINE, 
    BIRADS_CLINICAL_GUIDELINE_MG, 
    BIRADS_DIAGNOSTIC_GUIDELINE_US,
    BIRADS_DIAGNOSTIC_GUIDELINE_MG,
    DERMATOLOGY_CLINICAL_GUIDELINE_2011,
    BIRDS_FIELD_GUIDE
)

from src.utils.dataloaders import named_concept_bank

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoImageProcessor
from qwen_vl_utils import process_vision_info

import re

def normalize_species_name(raw_name: str) -> str:
    """
    Convert species names from classes.txt style (with leading numbers/underscores)
    into dictionary keys used in birds_guideline.
    
    Example:
        "001.Black_footed_Albatross" -> "Black-footed Albatross"
        "121.White_crowned_Sparrow"  -> "White-crowned Sparrow"
    """
    # Remove any leading numbers and dot (e.g., "001.")
    name = re.sub(r"^\d+\.", "", raw_name)
    
    # Replace underscores with spaces
    name = name.replace("_", " ")
    
    # Hyphenate common compound modifiers
    name = re.sub(
        r"\b([A-Z][a-z]+) (footed|winged|capped|crowned|necked|throated|tailed|backed)\b",
        r"\1-\2",
        name,
    )
    
    # Capitalize words if they appear lowercased
    name = " ".join([w.capitalize() if w.islower() else w for w in name.split()])
    
    return name


class Qwen:
    def __init__(self, size=3, min_px=256, max_px=1280, device='cuda'):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            f"/model-weights/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        ).to(device)

        self.device = device

        # default processer
        min_pixels = min_px*28*28
        max_pixels = max_px*28*28

        import json

        # Load and patch the config manually
        with open("/model-weights/Qwen2.5-VL-7B-Instruct/preprocessor_config.json") as f:
            preprocessor_config = json.load(f)

        # Manually delete the 'image_processor_type' key if it exists
        #if "image_processor_type" in preprocessor_config:
        #    del preprocessor_config["image_processor_type"]
        preprocessor_config["image_processor_type"] = "Qwen2VLImageProcessor"
        print("Preprocessor config:", preprocessor_config)

        # Now manually create the processor
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",min_pixels=min_pixels, max_pixels=max_pixels)

        #self.processor = AutoProcessor.from_pretrained("/model-weights/Qwen2.5-VL-7B-Instruct/", min_pixels=min_pixels, max_pixels=max_pixels)

    def run(self, image, metadata, prompt_mode="guidelines"):
        birads = metadata["birads"]
        dataset_name = metadata["dataset"]
        named_concepts = named_concept_bank(dataset=dataset_name)
        selected_concepts = metadata["selected_concepts"]
        species_name = metadata["species_name"]

        print(species_name)

        print(selected_concepts)
        
        # Determine the auxiliary data to include in the prompt
        auxiliary_data = ''
        #if dataset_name in ["BUSBRA", "BREAST_US", "BrEaST", "DDSM", "MVKL"]: # include BI-RADS
        if dataset_name in ["BUSBRA", "BREAST_US"]:
            birads_value = { 0: 2, 1: 3, 2: 4, 3: 5}[birads]
            auxiliary_data += f"Its BI-RADS category is {birads_value}"
        if dataset_name == "BrEaST":
            birads_value = {0: 2, 1: 3, 2: '4a', 3: '4b', 4: '4c', 5: 5}[birads]
            auxiliary_data += f"Its BI-RADS category is {birads_value}"
        if dataset_name == "DDSM":
            birads_value = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}[birads]
            roi_id = metadata["roi_id"]
            abnormality_type = "mass" if "mass" in roi_id.lower() else "calcification"
            which_breast = "left" if "left" in roi_id.lower() else "right"
            image_view = "CC" if "cc" in roi_id.lower() else "MLO"
            auxiliary_data += f"The image shows a {abnormality_type} in the {which_breast} breast, in the {image_view} view. "
            auxiliary_data += f"Its BI-RADS category is {birads_value}"
        if dataset_name == "CUB":
            species_name = normalize_species_name(species_name)
            auxiliary_data += f"The image shows a bird of the species {species_name}"
        if dataset_name == "Derm7pt":
            label = metadata["label"]
            elevation = metadata["elevation"]
            location = metadata["location"]
            auxiliary_data += f"The image shows a {elevation} skin lesion classified as {label}, located in the {location}"

        modality = {
            "BUSBRA": "breast ultrasound",
            "BREAST_US": "breast ultrasound",
            "BrEaST": "breast ultrasound",
            "DDSM": "mammography",
            "Derm7pt": "dermatology",
            "CUB": "natural"
        }[dataset_name]

        type_of_guideline = {
            "BUSBRA": "clinical",
            "BREAST_US": "clinical",
            "BrEaST": "clinical",
            "DDSM": "clinical",
            "Derm7pt": "dermatology",
            "CUB": "bird watching"
        }[dataset_name]

        GUIDELINE = {
            "BUSBRA": BIRADS_CLINICAL_GUIDELINE,
            "BrEaST": BIRADS_CLINICAL_GUIDELINE,
            "BREAST_US": BIRADS_CLINICAL_GUIDELINE,
            "DDSM": BIRADS_CLINICAL_GUIDELINE_MG,
            "Derm7pt": DERMATOLOGY_CLINICAL_GUIDELINE_2011,
            "CUB": BIRDS_FIELD_GUIDE[species_name] if species_name else ''
        }[dataset_name]

        GUIDELINE_DIAGNOSTIC = {
            "BUSBRA": BIRADS_DIAGNOSTIC_GUIDELINE_US,
            "BrEaST": BIRADS_DIAGNOSTIC_GUIDELINE_US,
            "BREAST_US": BIRADS_DIAGNOSTIC_GUIDELINE_US,
            "DDSM": BIRADS_DIAGNOSTIC_GUIDELINE_MG,
        }[dataset_name]

        # Prepare the concept data for the prompt
        concept_data = "Finally, you are given the following 'concepts' that are present in the image.\n"
        if dataset_name in ["BUSBRA", "BREAST_US", "BrEaST", "DDSM", "MVKL"]:
            concept_data += "These concepts ultimately determine the BI-RADS category of the image."

        for i in range(len(selected_concepts)):
            if metadata["concepts"][i] == 1:
                concept_data += f"{named_concepts[i].capitalize().replace('_', ' ')}: {metadata['concepts'][i]}\n"

        if type_of_guideline == "clinical":
            ending = """Write a clinical report based on the image, the clinical guideline provided, and the concepts present in the image.
            The report should be written in a professional and concise manner, suitable for a medical context."""
        else:
            ending = """Write a report based on the image, the guideline provided, and the concepts present in the image.
            The report should be written in a professional and concise manner, suitable for a professional context."""

        guideline_only_prompt = f"""You are given the following {modality} <image>. {auxiliary_data}.
        You are also given the following clinical guideline: \n\n

        {GUIDELINE}

        \n\n

        Write a clinical report based on the image, and the clinical guideline provided. 
        The report should be written in a professional and concise manner, suitable for a medical context.
        """

        prompt_with_guidelines = f"""You are given the following {modality} <image>. {auxiliary_data}.
        You are also given the following {type_of_guideline} guideline: \n\n

        {GUIDELINE}

        \n\n

        {concept_data}

        \n\n

        {ending}
        """

        prompt_with_concepts = f"""You are given the following breast ultrasound <image>. 
        You are also given the following 'concepts' that are present in the image.

        {concept_data}

        \n\n

        {ending}
        """

        reasoning_prompt = f"""You are given the following {modality} <image>. 
        You are tasked with detecting clinical concepts (BI-RADS descriptors) present in the image. 
        Given the extracted concepts, what are the implications of these concepts for the final diagnosis based on the BIRADS clinical guideline, provided below?
        Based on the guideline, explain the predicted concepts, analyze how those concepts align with the models' prediction, determine the most likely BI-RADS category and give follow-up recommendation.

        {GUIDELINE_DIAGNOSTIC}
        """

        prompt = '' # Initialize prompt as empty string
        if prompt_mode == "guidelines":
            prompt = prompt_with_guidelines

        elif prompt_mode == "concepts_only":
            prompt = prompt_with_concepts

        elif prompt_mode == "guideline_only":
            prompt = guideline_only_prompt

        elif prompt_mode == "reasoning":
            prompt = reasoning_prompt

        elif prompt_mode == "description":
            prompt = f"""You are given the following breast ultrasound <image>. 
            Write a clinical report based on the image. 
            The report should be written in a professional and concise manner, suitable for a medical context.
            """

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "name": "image"
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if prompt_mode == "hallucinate":
            # remove image from the messages
            import random
            rand_birads = random.choice(['1', '2', '3', '4A', '4B', '4C', '5'])
            auxiliary_data = f'its BI-RADS category is {rand_birads}.'
            prompt = f"""You are given the following clinical guidelineline. 
            Your job is to write a clinical report that sounds like it could be based on the guideline, but you do not have access to any images.
            You are free to hallucinate the content of the report, but it should be coherent and relevant to the guideline. Pretend that {auxiliary_data}.
            The report should be written in a professional and concise manner, suitable for a medical context.

            {GUIDELINE}
            """
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=768)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


class QwenGenerator(nn.Module):
    def __init__(self, size=3, min_px=256, max_px=1280, prompt_mode='guidelines', device='cuda'):
        super(QwenGenerator, self).__init__()
        self.llama = Qwen(size=size, min_px=min_px, max_px=max_px, device=device)
        self.device = device
        self.prompt_mode = prompt_mode

    def generate_clinical_report(self, image, metadata):
        print(image)
        if isinstance(image, list):
            print(len(image))
            if len(image) == 1:
                image = image[0]
                text = self.llama.run(image, metadata, self.prompt_mode)
                return text
            else:
                text = []
                for i in range(len(image)):
                    text.append(self.llama.run(image[i], metadata))
                return text
        else:
            text = self.llama.run(image, metadata, self.prompt_mode)
            return text

    def forward(self, images, birads_list):
        llama_output = []
        for i in range(len(images)):
            image = images[i]
            birads = birads_list[i]
            llama_output.append(
                self.generate_clinical_report(image, birads)
            )
        return llama_output
