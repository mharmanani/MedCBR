import torch
import torch.nn as nn
import numpy as np

from src.utils.clinical_guideline import (
    BIRADS_DIAGNOSTIC_GUIDELINE_US, 
    BIRADS_DIAGNOSTIC_GUIDELINE_MG, 
    DERMATOLOGY_CLINICAL_GUIDELINE_2011,
    BIRDS_FIELD_GUIDE
)

from src.utils.dataloaders import named_concept_bank
from src.utils.classes import CUB


from transformers import AutoModelForCausalLM, AutoTokenizer
import json

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


class LRM:
    def __init__(self, device='cuda', use_guideline=True, enable_thinking=True):
        model_name = f"/model-weights/Qwen3-8B"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        ).to(device)
        
        with open("/model-weights/Qwen3-8B/tokenizer_config.json") as f:
            tokenizer_config = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        self.device = device
        self.use_guideline = use_guideline
        self.enable_thinking = enable_thinking

    def run(self, metadata):
        y_pred = metadata['y_pred']
        dataset_name = metadata["dataset"]
        named_concepts = named_concept_bank(dataset=dataset_name)
        use_guideline = metadata.get("use_guideline", self.use_guideline)
        selected_concepts = metadata["selected_concepts"]
        species_name = metadata["species_name"]

        introduction = ''
        if dataset_name == "BREAST_US" or dataset_name == "DDSM":
            diagnosis = {0: "Benign", 1: "Cancer"}[y_pred]
            introduction += f"""You are given the final diagnostic prediction of a AI model for breast cancer diagnosis, which is {diagnosis}."""
            introduction += " Benign predictions can range from to BI-RADS 2 to BI-RADS 4, while malignant predictions correspond to BI-RADS 4A–5, reflecting different levels of suspicion for malignancy."
            introduction += " The model has also detected the following concepts in the image: "
        
        if dataset_name == "CUB":
            y_pred = int(y_pred.cpu().item())
            species_name = CUB.class_names[y_pred + 1]
            species_name = normalize_species_name(species_name)
            introduction = f"""You are given the final prediction of a AI model for bird species identification, which is {species_name}. 
            The model has also detected the following concepts in the image: """

        GUIDELINE = {
            "BUSBRA": BIRADS_DIAGNOSTIC_GUIDELINE_US,
            "BrEaST": BIRADS_DIAGNOSTIC_GUIDELINE_US,
            "BREAST_US": BIRADS_DIAGNOSTIC_GUIDELINE_US,
            "DDSM": BIRADS_DIAGNOSTIC_GUIDELINE_MG,
            "CUB": BIRDS_FIELD_GUIDE[species_name] if species_name else ''
        }[dataset_name]

        # Prepare the concept data for the prompt
        concept_data = ''
        for i in range(len(selected_concepts)):
            c_pred_i = metadata["concepts"][i] * 100
            if c_pred_i > 45.0:
                concept_data += f"{named_concepts[i].capitalize().replace('_', ' ')} ({c_pred_i}% confidence)\n"
            else:
                if dataset_name == "BREAST_US" and named_concepts[i] == 'regular_shape': # 
                    concept_data += f"Irregular shape ({c_pred_i}% confidence)\n"

        instructions = ''
        if dataset_name in ["BREAST_US", "DDSM"]:
            if use_guideline:
                instructions += f"""Assuming the diagnosis is correct, what are the implications of these concepts for the final diagnosis based on the BIRADS clinical guideline, provided below?
                Based on the guideline, explain the predicted concepts, analyze how those concepts align with the models' prediction, determine the most likely BI-RADS category and give follow-up recommendation. """
            else:
                instructions += f"""Assuming the diagnosis is correct, what are the implications of these concepts for the final diagnosis?
                Explain the predicted concepts, analyze how those concepts align with the models' prediction, determine the most likely BI-RADS category and give follow-up recommendation. """

        if dataset_name == "CUB":
            if use_guideline:
                instructions += f"""Assuming the final prediction is correct, what are the implications of these concepts for the final species classification based on an excerpt from the bird watching field guide, provided below?
                Based on the guideline, explain the predicted concepts, and analyze how those concepts align with the models' prediction. """
            else:
                instructions += f"""Assuming the final prediction is correct, what are the implications of these concepts for the final species classification?
                Explain the predicted concepts, and analyze how those concepts align with the models' prediction. """
        
        prompt_sections = [
            introduction.strip(),
            concept_data.strip(),
            instructions.strip(),
        ]

        if use_guideline:
            prompt_sections.append(str(GUIDELINE).strip())

        reasoning_prompt = "\n\n".join([section for section in prompt_sections if section])

        self.prompt = reasoning_prompt

        messages = [
            {"role": "user", "content": self.prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=metadata.get("enable_thinking", self.enable_thinking) # Switch between thinking and non-thinking modes. Default is True.
        )
       
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            use_cache=True,
            do_sample=False,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        think_end_ids = self.tokenizer.encode("</think>", add_special_tokens=False)

        def find_subsequence(sequence, subseq):
            for i in range(len(sequence) - len(subseq) + 1):
                if sequence[i:i+len(subseq)] == subseq:
                    return i + len(subseq)
            return None
        
        output_ids = generated_ids[0, model_inputs["input_ids"].shape[-1]:].tolist()

        think_end_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        split = find_subsequence(output_ids, think_end_ids)

        if split is None:
            thinking_ids = []
            content_ids = output_ids
        else:
            thinking_ids = output_ids[:split]
            content_ids = output_ids[split:]

        thinking_content = self.tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
        content = self.tokenizer.decode(content_ids, skip_special_tokens=True).strip()

        return thinking_content, content

    def get_prompt(self):
        return self.prompt

class Qwen3(LRM):
    pass

class ReasoningModel(nn.Module):
    def __init__(self, device='cuda', use_guideline=True, enable_thinking=True):
        super(ReasoningModel, self).__init__()
        self.model = Qwen3(device=device, 
                           use_guideline=use_guideline, 
                           enable_thinking=enable_thinking)
        self.device = device

    def generate_reasoning(self, metadata):
        text = self.model.run(metadata)
        return text

    def forward(self, metadata):
        return self.generate_reasoning(metadata)

    def get_prompt(self):
        return self.model.get_prompt()
