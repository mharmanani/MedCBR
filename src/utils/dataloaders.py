import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from random import uniform, random, gauss
from pydicom import dcmread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import LabelEncoder

DATA_DIR = '/home/harmanan/projects/aip-medilab/harmanan/breast_us/data/'

def _make_data_augmentations(augs_lst):
    augments = []
    for aug in augs_lst:
        if aug == 'speckle':
            augments.append(RandomSpeckleNoise())
        if aug == 'brightness':
            augments.append(RandomBrightness())
        if aug == 'flip':
            augments.append(RandomFlip())
        if aug == 'rotate':
            augments.append(RandomRotation())
    return augments

def birads_to_risk(birads):
    if int(birads[0]) < 4 or birads == '4a':
        return 0 # Benign
    if birads == '4b':
        return 1 # Low risk of malignancy
    if birads == '4c':
        return 2 # Moderate risk of malignancy
    if birads == '5':
        return 3 # High risk of malignancy

def compute_class_weights(df):
    """
    Computes class weights per task using inverse class frequency.

    Args:
        df (pd.DataFrame): A DataFrame where each column is a classification task.

    Returns:
        dict: {task_name: torch.Tensor of class weights}
    """
    class_weights = {}
    class_wgts = []

    concepts = [col for col in df.columns if col.startswith("concept_")]
    for c in concepts:
        class_counts = df[c].value_counts().sort_index()
        weights = 1.0 / class_counts  # Inverse frequency
        weights /= weights.sum()  # Normalize to sum to 1
        class_weights[c] = torch.tensor(weights.values, dtype=torch.float)
        class_wgts.append(class_weights[c])
    
    return class_wgts

def create_bus_concepts(df):
    """
    Create concepts for BUSBRA dataset based on the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing BUSBRA metadata.
        chosen_concepts (list): List of indices of concepts to create.

    Returns:
        pd.DataFrame: DataFrame with new concept columns.
    """
    def is_circumscribed_bus(x):
        return ("spiculated" not in x) \
            and ("microlobulated" not in x) \
            and ("indistinct" not in x) \
            and ("angular" not in x) \
            and ("uncircumscribed" not in x)
    
    df["concept_shadowing"] = df.Posterior_features.apply(lambda x: 1 if x == "shadowing" or x == "combined" else 0)
    df["concept_enhancement"] = df.Posterior_features.apply(lambda x: 1 if x == "enhancement" or x == "combined" else 0)
    df["concept_halo"] = df.Halo.apply(lambda x: 0 if x == 'no' else 1)
    df["concept_calc"] = df.Calcifications.apply(lambda x: 0 if x == 'no' else 1)
    df["concept_skin_thick"] = df.Skin_thickening.apply(lambda x: 1 if x == 'yes' else 0)
    df["concept_circumscribed"] = df.Margin.apply(lambda x: 1 if is_circumscribed_bus(x) else 0)
    df["concept_spiculated_mar"] = df.Margin.apply(lambda x: 1 if 'spiculated' in x else 0)
    df["concept_indistinct_mar"] = df.Margin.apply(lambda x: 1 if 'indistinct' in x else 0)
    df["concept_angular_mar"] = df.Margin.apply(lambda x: 1 if 'angular' in x else 0)
    df["concept_microlobulated_mar"] = df.Margin.apply(lambda x: 1 if 'microlobulated' in x else 0)
    df["concept_reg_shape"] = df.Shape.apply(lambda x: 0 if x in ['oval', 'round'] else 1)
    df["concept_echo_hyper"] = df.Echogenicity.apply(lambda x: 1 if x == "hyperechoic" else 0)
    df["concept_echo_hypo"] = df.Echogenicity.apply(lambda x: 1 if x == "hypoechoic" else 0)
    df["concept_echo_hetero"] = df.Echogenicity.apply(lambda x: 1 if x == "heterogeneous" else 0)
    df["concept_cystic"] = df.Echogenicity.apply(lambda x: 1 if "cystic" in x else 0)
    
    return df

def create_mvkl_concepts(df):
    """
    Create concepts for MVKL dataset based on the given DataFrame.
    """
    # mass shape: 
    df["concept_reg_shape"] = df.Shape.apply(lambda x: 0 if x.lower() in ['oval', 'round'] else 1)
    df["concept_irreg_shape"] = df.Shape.apply(lambda x: 1 if x.lower() in ['irregular', 'lobulated'] else 0)
    # mass margin:
    df["concept_circumscribed"] = df.Margin.apply(lambda x: 0 if 'not circumscribed' in x else 1)
    df["concept_spiculated_mar"] = df.Margin.apply(lambda x: 1 if 'spiculated' in x else 0)
    df["concept_obscured_mar"] = df.Margin.apply(lambda x: 1 if 'obscured' in x else 0)
    df["concept_microlobulated_mar"] = df.Margin.apply(lambda x: 1 if 'microlobulated' in x else 0)
    # mass size:
    df["concept_small_size"] = df.Size.apply(lambda x: 1 if x == 'small' else 0)
    df["concept_moderate_size"] = df.Size.apply(lambda x: 1 if x == 'moderate' else 0)
    df["concept_large_size"] = df.Size.apply(lambda x: 1 if x == 'large' else 0)
    # mass density:
    df["concept_low_density"] = df.Density.apply(lambda x: 1 if x == 'low' else 0)
    df["concept_median_density"] = df.Density.apply(lambda x: 1 if x == 'median' else 0)
    df["concept_high_density"] = df.Density.apply(lambda x: 1 if x == 'high' else 0)
    # calcifications - shape, size, distribution, density
    df["concept_popcorn_calc_shape"] = df.Calcification_Shape.apply(lambda x: 1 if x == 'popcorn' else 0)
    df["concept_branching_calc_shape"] = df.Calcification_Shape.apply(lambda x: 1 if x == 'branching' else 0)
    df["concept_crescentic_calc_shape"] = df.Calcification_Shape.apply(lambda x: 1 if x == 'coarse' else 0)
    df["concept_tiny_calc_size"] = df.Calcification_Size.apply(lambda x: 1 if x == 'small' else 0)
    df["concept_uneven_calc_size"] = df.Calcification_Size.apply(lambda x: 1 if x == 'uneven' else 0)
    df["concept_coarse_calc_size"] = df.Calcification_Size.apply(lambda x: 1 if x == 'coarse' else 0)
    df["concept_low_calc_density"] = df.Calcification_Density.apply(lambda x: 1 if x == 'low' else 0)
    df["concept_uneven_calc_density"] = df.Calcification_Density.apply(lambda x: 1 if x == 'uneven' else 0)
    df["concept_high_calc_density"] = df.Calcification_Density.apply(lambda x: 1 if x == 'high' else 0)
    df["concept_clustered_calc_dist"] = df.Calcification_Distribution.apply(lambda x: 1 if x == 'clustered' else 0)
    df["concept_scattered_calc_dist"] = df.Calcification_Distribution.apply(lambda x: 1 if x == 'scattered' else 0)
    df["concept_segmental_calc_dist"] = df.Calcification_Distribution.apply(lambda x: 1 if 'segmental' in x else 0)
    # other - Duct_sign', 'Comet_tail_sign', 'Nipple_retraction', 'Abnormal_Lymph_Node_Shadow', 'Abnormal_Blood_Vessel_Shadow'
    df["concept_halo"] = df.Halo.apply(lambda x: 0 if x == 'no' else 1)
    df["concept_skin_thick"] = df.Skin_thickening.apply(lambda x: 1 if x == 'yes' else 0)
    df["concept_distortion"] = df.Structural_distortion.apply(lambda x: 0 if x == 'no' else 1)
    df["concept_asymmetry"] = df.Asymmetry.apply(lambda x: 0 if x == 'no' else 1)
    df["concept_duct_sign"] = df.Duct_sign.apply(lambda x: 0 if x == 'no' else 1)
    df["concept_comet_tail_sign"] = df.Comet_tail_sign.apply(lambda x: 0 if x == 'no' else 1)
    df["concept_nipple_retraction"] = df.Nipple_retraction.apply(lambda x: 0 if x == 'no' else 1)
    df["concept_shadowing_lymph_node"] = df.Abnormal_Lymph_Node_Shadow.apply(lambda x: 0 if x == 'no' else 1)
    df["concept_shadowing_blood_vessel"] = df.Abnormal_Blood_Vessel_Shadow.apply(lambda x: 0 if x == 'no' else 1)

    return df

def create_ddsm_concepts(df):
    # mass shape
    df["concept_reg_shape"] = df["mass shape"].apply(lambda x: 1 if ('round' in x.lower() or 'oval' in x.lower()) else 0)
    df["concept_irreg_shape"] = df["mass shape"].apply(lambda x: 1 if 'irregular' in x.lower() else 0)
    df["concept_lobulated_shape"] = df["mass shape"].apply(lambda x: 1 if 'lobulated' in x.lower() else 0)
    # mass margin
    df["concept_circumscribed"] = df["mass margins"].apply(lambda x: 1 if 'circumscribed' in x.lower() else 0)
    df["concept_ill_defined_margins"] = df["mass margins"].apply(lambda x: 1 if 'ill-defined' in x.lower() else 0)
    df["concept_spiculated_margins"] = df["mass margins"].apply(lambda x: 1 if 'spiculated' in x.lower() else 0)
    df["concept_obscured_margins"] = df["mass margins"].apply(lambda x: 1 if 'obscured' in x.lower() else 0)
    df["concept_microlobulated_margins"] = df["mass margins"].apply(lambda x: 1 if 'microlobulated' in x.lower() else 0)
    # calc type
    df["concept_pleomorphic_calc"] = df["calc type"].apply(lambda x: 1 if 'pleomorphic' in x.lower() else 0)
    df["concept_amorphous_calc"] = df["calc type"].apply(lambda x: 1 if 'amorphous' in x.lower() else 0)
    df["concept_fine_linear_calc"] = df["calc type"].apply(lambda x: 1 if 'fine_linear' in x.lower() else 0)
    df["concept_branching_calc"] = df["calc type"].apply(lambda x: 1 if 'branching' in x.lower() else 0)
    df["concept_vascular_calc"] = df["calc type"].apply(lambda x: 1 if 'vascular' in x.lower() else 0)
    df["concept_coarse_calc"] = df["calc type"].apply(lambda x: 1 if 'coarse' in x.lower() else 0)
    df["concept_punctate_calc"] = df["calc type"].apply(lambda x: 1 if 'punctate' in x.lower() else 0)
    df["concept_lucent_calc"] = df["calc type"].apply(lambda x: 1 if 'lucent' in x.lower() else 0)
    df["concept_eggshell_calc"] = df["calc type"].apply(lambda x: 1 if 'eggshell' in x.lower() else 0)
    df["concept_round_calc"] = df["calc type"].apply(lambda x: 1 if 'round' in x.lower() else 0)
    df["concept_regular_calc"] = df["calc type"].apply(lambda x: 1 if 'regular' in x.lower() else 0)
    df["concept_dystrophic_calc"] = df["calc type"].apply(lambda x: 1 if 'dystrophic' in x.lower() else 0)
    # calcification distribution
    df["concept_clustered_calc_dist"] = df["calc distribution"].apply(lambda x: 1 if 'clustered' in x.lower() else 0)
    df["concept_segmental_calc_dist"] = df["calc distribution"].apply(lambda x: 1 if 'segmental' in x.lower() else 0)
    df["concept_linear_calc_dist"] = df["calc distribution"].apply(lambda x: 1 if 'linear' in x.lower() else 0)
    df["concept_scattered_calc_dist"] = df["calc distribution"].apply(lambda x: 1 if 'scattered' in x.lower() else 0)
    df["concept_regional_calc_dist"] = df["calc distribution"].apply(lambda x: 1 if 'regional' in x.lower() else 0)
    # other
    df["concept_low_breast_density"] = df["breast density"].apply(lambda x: 1 if x in [1,2] else 0)
    df["concept_moderate_breast_density"] = df["breast density"].apply(lambda x: 1 if x in [3] else 0)
    df["concept_high_breast_density"] = df["breast density"].apply(lambda x: 1 if x in [4] else 0)
    df['concept_architecture_distortion'] = df["mass shape"].apply(lambda x: 1 if 'distortion' in x.lower() else 0)
    df['concept_asymmetry'] = df["mass shape"].apply(lambda x: 1 if 'asymmetr' in x.lower() else 0)
    df['concept_lymph_node'] = df["mass shape"].apply(lambda x: 1 if 'lymph' in x.lower() else 0)

    return df

def named_concept_bank(dataset='breast_us'):
    from src.utils.concept_bank.CUB import CUB_CONCEPT_BANK
    #from src.utils.concept_bank.BREAST_US import BUS_CONCEPT_BANK

    dataset = dataset.lower()
    if dataset == 'breast_us':
        return [
            "Posterior acousting shadowing",
            "Posterior enhancement",
            "halo",
            "calcifications",
            "skin thickening",
            "Circumscribed margins",
            "Spiculated margins",
            "Indistinct margins",
            "Angular margins",
            "Microlobulated margins",
            "a regular shape",
            "Hyperechoic echo",
            "Hypoechoic",
            "Heterogeneous echo",
            "Complex cystic echo"
        ] # len = 15
    
    elif dataset == 'ddsm':
        return [
            "Regular shape",
            "Irregular shape",
            "Lobulated shape",
            "Circumscribed margins",
            "Ill-defined margins",
            "Spiculated margins",
            "Obscured margins",
            "Microlobulated margins",
            "Pleomorphic calcification",
            "Amorphous calcification",
            "Fine linear calcification",
            "Branching calcification",
            "Vascular calcification",
            "Coarse calcification",
            "Punctate calcification",
            "Lucent calcification",
            "Eggshell calcification",
            "Round calcification",
            "Regular calcification",
            "Dystrophic calcification",
            "Clustered calcification distribution",
            "Segmental calcification distribution",
            "Linear calcification distribution",
            "Scattered calcification distribution",
            "Regional calcification distribution",
            "Low breast density",
            "Moderate breast density",
            "High breast density",
            "Architectural distortion",
            "Asymmetry",
        ] # len = 30

    elif dataset == 'cub':
        return CUB_CONCEPT_BANK

def create_concepts(df, mode='breast_us', chosen_concepts=list(range(15))):
    if mode == 'breast_us':
        df = create_bus_concepts(df)
    if mode == 'mvkl':
        df = create_mvkl_concepts(df)
    if mode == 'ddsm':
        df = create_ddsm_concepts(df)
    if mode == 'cub':
        pass

    print(df.columns)

    concepts = [col for col in df.columns if col.startswith("concept_")]

    concept_df = df[[concepts[i] for i in chosen_concepts]]
    print(concept_df)
    df["concepts"] = concept_df.apply(lambda x: np.array([x[col] for col in concept_df.columns]), axis=1)
    
    return df

def make_report_from_concepts(concepts, modality="Breast Ultrasound", mode='breast_us'):
    named_concepts = named_concept_bank(dataset=mode)
    report = f"""
    **Patient Information:**
    [Insert Patient Information]

    **Date of Examination:**
    [Insert Date]

    **Imaging Modality:**
    {modality}

    **Image Description:**
    The image has the following characteristics:\n
    """

    if mode == 'cub':
        named_concepts = named_concept_bank(dataset=mode)
        report = f"""
        **Bird Identification Report**

        **Visual Attributes:**
        This bird exhibits the following observable characteristics:\n
        """

    for i in range(len(concepts)):
        if concepts[i] == 1:
            report += f"- {named_concepts[i]}.\n"
        else:
            if mode == 'breast_us':
                if named_concepts[i] == "Circumscribed margins":
                    report += f"- uncircumscribed margins\n"
                if named_concepts[i] == "a regular shape":
                    report += f"- an irregular shape\n"

    report += "\n"

    return report

import random

def generate_random_report(dataset="BREAST_US",
                           alpha=0.0, beta=1.0, length=70,
                           seed=None):
    """
    Generate a pseudo-report composed of short random sentences.
    The vocabulary automatically adapts to the chosen modality.
    """

    if seed is not None:
        random.seed(seed)

    modality = {
        'BREAST_US': "Ultrasound",
        'DDSM': "Mammography",
        'CUB': "Natural Image"
    }[dataset]

    # ------------------------------
    # Shared random language pool
    # ------------------------------
    t_rand = [
        # Generic words and phrases
        "forest", "mirror", "machine", "window", "pattern", "shadow", "object",
        "reflection", "light", "texture", "bridge", "river", "sky", "mountain",
        "story", "path", "stone", "ground", "motion", "color", "noise", "glass",
        "system", "signal", "data", "algorithm", "surface", "screen", "design",
        "energy", "camera", "image", "field", "shape", "direction", "form",
        "movement", "detail", "layer", "edge", "contrast", "depth", "composition",
        "background", "figure", "model", "pattern", "sequence", "signal", "feature",
        "component", "structure", "geometry", "texture", "variation", "frame",
        "scene", "object", "sample", "source", "dimension", "area", "surface",
        "object", "material", "shadow", "volume", "region", "highlight", "distance",
        "focus", "intensity", "environment", "artifact", "context", "measurement",
        "density", "frequency", "position", "orientation", "parameter",
        "dynamic", "response", "contrast", "behavior", "noise", "pattern",
        "temporal", "visual", "feature", "scale", "estimate", "distribution",
        "relation", "phase", "process", "observation", "spectrum", "simulation",
        "sequence", "representation", "analysis", "projection", "reconstruction",
        "random", "natural", "continuous", "static", "smooth", "complex", "simple",
        "uniform", "regular", "irregular", "sharp", "blurred", "fine", "coarse",
        "bright", "dim", "dark", "neutral", "soft", "strong", "faint", "stable",
        "variable", "gradual", "abrupt", "consistent", "scattered", "isolated",
        "clustered", "adjacent", "diffuse", "concentrated",
        # Small sentences/fragments
        "The pattern shifts slightly under changing light.",
        "A reflection appears across the surface.",
        "The structure extends beyond the main frame.",
        "Multiple regions exhibit varying contrast.",
        "Edges are blurred due to shallow focus.",
        "Subtle shadows suggest depth in the composition.",
        "The distribution of features is irregular.",
        "This area remains unchanged despite rotation.",
        "There is significant variation across layers.",
        "The object appears partially occluded by another.",
        "Some regions contain overlapping textures.",
        "A clear highlight separates the two areas.",
        "Intensity gradually decreases from center to edge.",
        "Small details are visible under magnification.",
        "Noise increases toward the boundary of the frame.",
        "The composition remains balanced despite asymmetry.",
        "A faint signal can be observed near the border.",
        "The observed effect is consistent with reflection.",
        "Light scattering contributes to overall brightness.",
        "Spatial alignment appears slightly off-center."
    ]

    # ------------------------------
    # Domain-specific vocabularies
    # ------------------------------
    if modality == "Ultrasound":
        descriptors = [
            "spiculated", "circumscribed", "indistinct", "angular", "lobulated",
            "well-defined", "irregular", "heterogeneous", "hypoechoic", "hyperechoic",
            "isoechoic", "complex", "shadowing", "enhancing"
        ]
        subjects = [
            "mass", "lesion", "nodule", "region", "margin", "calcification",
            "parenchyma", "duct", "fat", "septation", "tissue", "area"
        ]

    elif modality == "Mammography":
        descriptors = [
            "spiculated", "obscured", "microlobulated", "circumscribed",
            "architectural", "asymmetric", "focal", "dense", "scattered",
            "coarse", "clustered", "heterogeneous", "fine", "pleomorphic"
        ]
        subjects = [
            "mass", "asymmetry", "density", "calcification", "tissue", "architecture",
            "nodule", "region", "fatty area", "fibroglandular zone"
        ]

    elif dataset == "CUB":
        descriptors = [
            "blue", "rufous", "iridescent", "striped", "mottled", "spotted",
            "streaked", "glossy", "dull", "bright", "pale", "dark", "broad",
            "narrow", "hooked", "pointed", "long", "short"
        ]
        subjects = [
            "wing", "bill", "tail", "crown", "throat", "breast", "belly", "back",
            "eye", "plumage", "feather", "mantle", "flank", "rump"
        ]

    # ------------------------------
    # Sentence generation
    # ------------------------------
    total = alpha + beta
    alpha, beta = alpha / total, beta / total
    n_med = int(length * alpha)
    n_rand = length - n_med

    connectors = [
        "appears near", "is adjacent to", "is associated with",
        "is seen along", "lies beneath", "overlaps with", "contacts",
        "extends towards", "is consistent with"
    ]

    sentences = []
    while len(" ".join(sentences).split()) < length:
        if random.random() < alpha:
            d = random.choice(descriptors)
            s = random.choice(subjects)
            c = random.choice(connectors)
            r = random.choice(t_rand)
            sentences.append(f"The {d} {s} {c} the {r}.")
        else:
            sentence = random.choice(t_rand)
            if not sentence.endswith("."):
                sentence = sentence.capitalize() + "."
            sentences.append(sentence)

    # limit total words
    body = " ".join(sentences).split()[:length]
    report_text = " ".join(body)

    # ------------------------------
    # Report sections
    # ------------------------------
    if modality == "CUB":
        sections = [
            "**Observation:**",
            "[Insert Observation Context]",
            "",
            "**Date of Observation:**",
            "[Insert Date]",
            "",
            f"**Dataset Domain:** {modality}",
            "",
            "**Description:**",
            report_text,
            "",
            "**Notes:**",
            "The bird displays mixed characteristics across body regions."
        ]
    else:
        sections = [
            "**Patient Information:**",
            "Age: [##]   Sex: [M/F]",
            "",
            "**Date of Examination:**",
            "[Insert Date]",
            "",
            f"**Imaging Modality:** {modality}",
            "",
            "**Findings:**",
            report_text,
            "",
            "**Impression:**",
            "Overall appearance is variable and should be correlated clinically."
        ]

    return "\n".join(sections)

def oversample_cancer_data(df, sampling_ratio=1.0, seed=42, label_name='label'):
    """
    Oversample cancer cases to balance the dataset based on the sampling ratio.
    
    Args:
        df (pd.DataFrame): The dataset with a 'label' column where 1 indicates cancer and 0 indicates benign.
        sampling_ratio (float): Ratio of benign to cancer cases after oversampling.
        seed (int): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: The oversampled dataset.
    """
    np.random.seed(seed)
    benign_df = df[df[label_name] == 0]
    cancer_df = df[df[label_name] == 1]
    
    num_benign = benign_df.shape[0]
    num_cancer = cancer_df.shape[0]
    
    num_resample = int(num_benign * sampling_ratio) - num_cancer
    if num_resample > 0:
        cancer_resampled = cancer_df.sample(num_resample, replace=True, random_state=seed)
        df = pd.concat([df, cancer_resampled]).reset_index(drop=True)
    
    return df

def undersample_benign_data(df, sampling_ratio=1.0, seed=42, label_name='label'):
    np.random.seed(seed)
    benign_df = df[df[label_name] == 0]
    cancer_df = df[df[label_name] == 1]
    
    num_benign = benign_df.shape[0]
    num_cancer = cancer_df.shape[0]
    
    num_resample = int(num_cancer / sampling_ratio)
    if num_resample > 0:
        benign_resampled = benign_df.sample(num_resample, replace=True, random_state=seed)
        df = pd.concat([df, benign_resampled]).reset_index(drop=True)
    
    return df

def get_metadata_flag(name):
    if name.split("_") != 2:
        return ''
    else:
        return name.split("_")[1] if len(name.split("_")) > 1 else ''

def make_bus_data(conf):
    num_folds = conf.data.num_folds
    fold = conf.data.fold
    batch_sz = conf.data.batch_size
    style = conf.data.splitting
    sampling = conf.data.sampling
    sampling_ratio = conf.data.sampling_ratio
    augmentations = conf.data.augmentations
    try:
        chosen_concepts = conf.cbm.concepts 
    except: chosen_concepts = list(range(7))
    seed = conf.seed

    if conf.data.dataset == 'BrEaST':
        data_dir = os.path.join(DATA_DIR, 'BrEaST')
        return make_breast_datasets(
            data_dir=data_dir, style=style, fold=fold, batch_sz=batch_sz,
            num_folds=num_folds, sampling=sampling,
            sampling_ratio=sampling_ratio, augmentations=augmentations,
            chosen_concepts=chosen_concepts, seed=seed)
    if conf.data.dataset == 'BUSBRA':
        print('Loading BUSBRA dataset')
        data_dir = os.path.join(DATA_DIR, 'BUSBRA')
        return make_busbra(
            data_dir=data_dir, style=style, fold=fold, batch_sz=batch_sz,
            num_folds=num_folds, sampling=sampling,
            sampling_ratio=sampling_ratio, augmentations=augmentations,
            chosen_concepts=chosen_concepts, seed=seed)
    if conf.data.dataset == 'BREAST_US':
        return make_breast_and_busbra_dataset(
            style=style, fold=fold, batch_sz=batch_sz, image_sz=conf.data.image_size,
            num_folds=num_folds, sampling=sampling, datasets=conf.data.datasets,
            sampling_ratio=sampling_ratio, augmentations=augmentations,
            crop_rois=conf.data.crop_rois, use_llm_output=conf.data.use_llm_output,
            chosen_concepts=chosen_concepts, seed=seed)
    if conf.data.dataset == 'PAIRED':
        return make_paired_breast_and_busbra_dataset(
            style=style, fold=fold, batch_sz=batch_sz,
            num_folds=num_folds, sampling=sampling,
            sampling_ratio=sampling_ratio, augmentations=augmentations,
            chosen_concepts=chosen_concepts, seed=seed)
    if conf.data.dataset == 'DDSM':
        return make_ddsm_dataset(
            style=style, fold=fold, batch_sz=batch_sz, image_sz=conf.data.image_size,
            num_folds=num_folds, sampling=sampling, 
            sampling_ratio=sampling_ratio, augmentations=augmentations,
            use_llm_output=conf.data.use_llm_output,
            chosen_concepts=chosen_concepts, seed=seed)
    if conf.data.dataset == "CUB":
        return make_cub_dataset(
            style=style, fold=fold, batch_sz=batch_sz, image_sz=conf.data.image_size,
            num_folds=num_folds, sampling=sampling, use_llm_output=conf.data.use_llm_output,
            sampling_ratio=sampling_ratio, augmentations=augmentations,
            chosen_concepts=chosen_concepts, seed=seed, preprocessing=conf.data.preprocessing)
    

def make_busbra(data_dir, 
                style="kfold",
                fold=0,
                batch_sz=32,
                num_folds=5,
                sampling='oversample',
                sampling_ratio=1.0,
                augmentations=None,
                chosen_concepts=list(range(7)),
                seed=42):
    df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    
    try:
        df = create_concepts(df, chosen_concepts)
        df.rename(columns={'concepts': 'concepts_real'}, inplace=True)
    except:
        pass
    
    ids = df.Case.unique()

    if sampling:
        assert sampling in ['oversample', 'undersample']
        oversample_cancer = sampling == 'oversample'
        undersample_benign = sampling == 'undersample'

    birads_label_enc = LabelEncoder()
    df.BIRADS = birads_label_enc.fit_transform(df.BIRADS)

    cancer_label_enc = LabelEncoder()
    df.Pathology = cancer_label_enc.fit_transform(df.Pathology)

    df_by_pa = df[df.ID.apply(lambda x: not x.endswith('-r'))]
    df_by_pa = df_by_pa.reset_index()

    if style == "kfold":
        skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)

        for i, (train_idx, test_idx) in enumerate(
            skf.split(df_by_pa[["Case", "Pathology"]], df_by_pa["Pathology"])
        ):
            print(f"Fold {i}")
            print(f"Train: {len(train_idx)}")
            print(f"Test: {len(test_idx)}")
            if i == fold:
                train_pa, val_pa = df_by_pa.iloc[train_idx], df_by_pa.iloc[test_idx]
                val_pa, test_pa = train_test_split(
                    val_pa, test_size=0.5, random_state=seed, stratify=val_pa["Pathology"]
                )

                train_idx = df.Case.isin(train_pa.Case)
                val_idx = df.Case.isin(val_pa.Case)
                test_idx = df.Case.isin(test_pa.Case)

                train_tab, val_tab, test_tab = (
                    df[train_idx],
                    df[val_idx],
                    df[test_idx],
                )

                assert set(train_tab.Case) & set(val_tab.Case) == set()
                assert set(train_tab.Case) & set(test_tab.Case) == set()
                assert set(val_tab.Case) & set(test_tab.Case) == set()

                num_benign = train_tab[train_tab.Pathology == 0].shape[0]
                num_cancer = train_tab[train_tab.Pathology == 1].shape[0]
                if oversample_cancer:
                    print(f"Num. cancer cores: {num_cancer}")
                    num_resample = int(num_cancer * sampling_ratio)
                    print(f"Num. *resampled* cancer cores: {num_resample}")
                    train_tab = pd.concat([
                        train_tab,
                        train_tab[train_tab.Pathology == 1].sample(num_resample, replace=True)
                    ]).reset_index(drop=True)
                elif undersample_benign:
                    print(f"Num. benign cores: {num_benign}")
                    num_resample = int(num_benign / sampling_ratio)
                    print(f"Num. *resampled* benign cores: {num_resample}")
                    train_tab = pd.concat([
                        train_tab[train_tab.Pathology == 1],
                        train_tab[train_tab.Pathology == 0].sample(num_resample)
                    ]).reset_index(drop=True)

                train_ds, val_ds, test_ds = BUSBRA_Dataset(data_dir, train_tab, chosen_concepts=chosen_concepts, augs=augmentations, style=style),\
                            BUSBRA_Dataset(data_dir, val_tab, chosen_concepts=chosen_concepts, style=style),\
                            BUSBRA_Dataset(data_dir, test_tab, chosen_concepts=chosen_concepts, style=style)

                train_dl, val_dl, test_dl = make_dataloader(train_ds, batch_size=batch_sz), \
                                            make_dataloader(val_ds, batch_size=batch_sz), \
                                            make_dataloader(test_ds, batch_size=batch_sz)

                return train_dl, val_dl, test_dl
        
    elif style == "test_on_concepts" or style == "holdout":
        train_df = df[df.split == 'train']
        test_df = df[df.split == 'test']

        # no val split, split from train
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=seed)

        train_ds, val_ds, test_ds = (
                            BUSBRA_Dataset(data_dir, train_df, chosen_concepts=chosen_concepts, augs=augmentations),\
                            BUSBRA_Dataset(data_dir, val_df, chosen_concepts=chosen_concepts),\
                            BUSBRA_Dataset(data_dir, test_df, chosen_concepts=chosen_concepts))

        train_dl, val_dl, test_dl = (
            make_dataloader(train_ds, batch_size=batch_sz), \
            make_dataloader(val_ds, batch_size=batch_sz), \
            make_dataloader(test_ds, batch_size=batch_sz)
        )

        return train_dl, val_dl, test_dl

    else:
        raise NotImplementedError(f"Unknown style: {style}")

def make_breast_datasets(data_dir, 
                         style="kfold",
                         fold=0,
                         batch_sz=8,
                         num_folds=5,
                         sampling=None,
                         sampling_ratio=1.0,
                         augmentations=None,
                         chosen_concepts=list(range(7)),
                         seed=42):
    df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'), keep_default_na=False)
    df = df[df.Classification != 'normal']
    df = create_concepts(df, chosen_concepts)
    df['report'] = df.apply(lambda x: make_report_from_table(x, chosen_concepts), axis=1)
    ids = df['CaseID'].unique()

    if sampling:
        assert sampling in ['oversample', 'undersample']
    oversample_cancer = sampling == 'oversample'
    undersample_benign = sampling == 'undersample'

    from sklearn.preprocessing import LabelEncoder

    birads_label_enc = LabelEncoder()
    df.BIRADS = df.BIRADS.apply(lambda x: x.split(' ')[0]) # remove the 'a', 'b', 'c' suffix
    df.BIRADS = birads_label_enc.fit_transform(df.BIRADS)

    cancer_label_enc = LabelEncoder()
    df.Classification = cancer_label_enc.fit_transform(df.Classification)

    train_df, test_df = train_test_split(df, 
                                         test_size=0.15, 
                                         random_state=seed)
    test_idx = df.CaseID.isin(test_df.CaseID)

    if style == "kfold":
        skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)

        for i, (train_idx, val_idx) in enumerate(
            skf.split(train_df, train_df["Classification"])
        ):
            print(f"Fold {i}")
            print(f"Train: {len(train_idx)}")
            print(f"Val: {len(val_idx)}")
            print(f"Test: {len(test_idx)}")
            if i == fold:
                train_df, val_df = train_df.iloc[train_idx], train_df.iloc[val_idx]
                
                train_idx = df.CaseID.isin(train_df.CaseID)
                val_idx = df.CaseID.isin(val_df.CaseID)
                
                train_tab, val_tab, test_tab = (
                    df[train_idx],
                    df[val_idx],
                    df[test_idx],
                )

                assert set(train_tab.CaseID) & set(val_tab.CaseID) == set()
                assert set(train_tab.CaseID) & set(test_tab.CaseID) == set()
                assert set(val_tab.CaseID) & set(test_tab.CaseID) == set()

                num_benign = train_tab[train_tab.Classification == 0].shape[0]
                num_cancer = train_tab[train_tab.Classification == 1].shape[0]
                if oversample_cancer:
                    print(f"Num. cancer cores: {num_cancer}")
                    train_tab = oversample_cancer_data(train_tab, label_name='Classification', sampling_ratio=sampling_ratio)
                    num_resample = train_tab[train_tab.Classification == 1].shape[0]
                    print(f"Num. *resampled* cancer cores: {num_resample}")
                elif undersample_benign:
                    print(f"Num. benign cores: {num_benign}")
                    train_tab = undersample_benign_data(train_tab, label_name='Classification', sampling_ratio=sampling_ratio)
                    num_resample = train_tab[train_tab.Classification == 0].shape[0]
                    print(f"Num. *resampled* benign cores: {num_resample}")

                train_ds, val_ds, test_ds = BrEaST_Dataset(data_dir, train_tab, augs=augmentations),\
                            BrEaST_Dataset(data_dir, val_tab),\
                            BrEaST_Dataset(data_dir, test_tab)

                train_dl, val_dl, test_dl = make_dataloader(train_ds, batch_size=batch_sz), \
                                            make_dataloader(val_ds, batch_size=batch_sz), \
                                            make_dataloader(test_ds, batch_size=batch_sz)

                return train_dl, val_dl, test_dl
        
    elif style == "holdout_noval":
        train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=seed)
        train_df = df[df['CaseID'].isin(train_ids)]
        test_df = df[df['CaseID'].isin(test_ids)]

        train_ds, test_ds = BrEaST_Dataset(data_dir, train_df, augs=augmentations),\
                            BrEaST_Dataset(data_dir, test_df)

        train_dl, test_dl = make_dataloader(train_ds, batch_size=batch_sz), \
                            make_dataloader(test_ds, batch_size=batch_sz)

        return train_dl, test_dl, test_dl

    else:
        train_ids, val_ids = train_test_split(ids, test_size=0.3, random_state=seed)
        val_ids, test_ids = train_test_split(val_ids, test_size=0.6, random_state=seed)

        train_df = df[df['CaseID'].isin(train_ids)]
        val_df = df[df['CaseID'].isin(val_ids)]
        test_df = df[df['CaseID'].isin(test_ids)]

        return train_df, val_df, test_df

def make_breast_and_busbra_dataset(style="kfold",
                         fold=0,
                         batch_sz=8,
                         num_folds=5,
                         sampling=None,
                         sampling_ratio=1.0,
                         augmentations=None,
                         chosen_concepts=list(range(15)),
                         datasets=["BrEaST", "BUSBRA"],
                         crop_rois=False,
                         use_llm_output=True,
                         image_sz=256,
                         seed=42):
    dir_breast = os.path.join(DATA_DIR, f'BrEaST{"_cropped" if crop_rois else ""}')
    dir_busbra = os.path.join(DATA_DIR, f'BUSBRA{"_cropped" if crop_rois else ""}')
    print(f"Loading breast data from {dir_breast}")
    print(f"Loading busbra data from {dir_busbra}")
    df_breast = pd.read_csv(os.path.join(dir_breast, f'metadata.csv'))
    df_breast = df_breast[df_breast.Classification != 'normal']
    df_breast = create_concepts(df_breast, mode="breast_us", chosen_concepts=chosen_concepts)
    
    df_busbra = pd.read_csv(os.path.join(dir_busbra, f'metadata.csv'))
    df_busbra = create_concepts(df_busbra, mode="breast_us", chosen_concepts=chosen_concepts)
    df_busbra = df_busbra[df_busbra.Shape != '-1'].reset_index(drop=True)
    df_busbra.rename(columns={'Case': 'CaseID'}, inplace=True)

    cancer_label_enc = LabelEncoder()
    df_busbra.Pathology = cancer_label_enc.fit_transform(df_busbra.Pathology)
    df_breast.Classification = cancer_label_enc.fit_transform(df_breast.Classification)

    birads_label_enc = LabelEncoder()
    df_breast.BIRADS = df_breast.BIRADS.apply(lambda x: str(x)[0])  # remove the 'a', 'b', 'c' suffix
    df_breast.BIRADS = birads_label_enc.fit_transform(df_breast.BIRADS)
    df_busbra.BIRADS = birads_label_enc.fit_transform(df_busbra.BIRADS)

    df_busbra["CaseID"] = df_busbra["CaseID"].apply(lambda x: f"BUSBRA-00{x}")
    df_breast["CaseID"] = df_breast["CaseID"].apply(lambda x: f"BrEaST-00{x}")

    df_busbra.rename(columns={'Pathology': 'Classification'}, inplace=True)

    if style == "kfold":
        df_all = pd.concat([df_breast, df_busbra], ignore_index=True)

        # Step 1: one label per patient
        caseid_label_df = df_all.groupby("CaseID").first().reset_index()[["CaseID", "Classification"]]        

        # Step 2: train-test split (case-level)
        all_caseids = caseid_label_df["CaseID"].tolist()
        train_ids, test_ids = train_test_split(
            caseid_label_df["CaseID"],
            test_size=0.2,
            stratify=caseid_label_df["Classification"],
            random_state=seed
        )

        # Step 3: k-fold on case-level train set
        train_cases = caseid_label_df[caseid_label_df["CaseID"].isin(train_ids)].reset_index(drop=True)
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

        for i, (train_index, val_index) in enumerate(skf.split(train_cases, train_cases["Classification"])):
            if i == fold:
                # Step 4: get case IDs per fold
                train_case_ids = train_cases.iloc[train_index]["CaseID"]
                val_case_ids = train_cases.iloc[val_index]["CaseID"]

                # Step 5: get actual image rows from df_all
                train_tab = df_all[df_all["CaseID"].isin(train_case_ids)]
                val_tab = df_all[df_all["CaseID"].isin(val_case_ids)]
                test_tab = df_all[df_all["CaseID"].isin(test_ids)]

                # Verify no leakage
                assert set(train_tab.CaseID) & set(val_tab.CaseID) == set()
                assert set(train_tab.CaseID) & set(test_tab.CaseID) == set()
                assert set(val_tab.CaseID) & set(test_tab.CaseID) == set()

                if oversample_cancer_data:
                    print(f"Num. cancer cores in train: {train_tab[train_tab.Classification == 1].shape[0]}")
                    train_tab = oversample_cancer_data(train_tab, sampling_ratio=sampling_ratio, label_name='Classification')
                    print(f"Num. *resampled* cancer cores in train: {train_tab[train_tab.Classification == 1].shape[0]}")
                elif undersample_benign_data:
                    print(f"Num. benign cores in train: {train_tab[train_tab.Classification == 0].shape[0]}")
                    train_tab = undersample_benign_data(train_tab, sampling_ratio=sampling_ratio, label_name='Classification')
                    print(f"Num. *resampled* benign cores in train: {train_tab[train_tab.Classification == 0].shape[0]}")

                # Optional: separate back out into breast/busbra
                busbra_train = df_busbra[df_busbra["CaseID"].isin(train_case_ids)]
                busbra_val = df_busbra[df_busbra["CaseID"].isin(val_case_ids)]
                busbra_test = df_busbra[df_busbra["CaseID"].isin(test_ids)]

                breast_train = df_breast[df_breast["CaseID"].isin(train_case_ids)]
                breast_val = df_breast[df_breast["CaseID"].isin(val_case_ids)]
                breast_test = df_breast[df_breast["CaseID"].isin(test_ids)]

                train_ds, val_ds, test_ds = (
                            BrEaST_and_BUSBRA_Dataset(data_dir=DATA_DIR, datasets=["BrEaST", "BUSBRA"],
                                                      df_breast=breast_train, df_busbra=busbra_train,
                                                      crop_rois=crop_rois, image_size=image_sz, 
                                                      augs=augmentations, use_llm_output=use_llm_output),\
                            BrEaST_and_BUSBRA_Dataset(data_dir=DATA_DIR, datasets=["BrEaST", "BUSBRA"],
                                                      df_breast=breast_val, crop_rois=crop_rois, image_size=image_sz,
                                                      df_busbra=busbra_val, use_llm_output=use_llm_output),\
                            BrEaST_and_BUSBRA_Dataset(data_dir=DATA_DIR, datasets=["BUSBRA"],
                                                      df_breast=breast_test, df_busbra=busbra_test, 
                                                      crop_rois=crop_rois, image_size=image_sz))

                train_ds.breast_metadata.to_csv(f"/home/harmanan/scratch/train_breast_fold{fold}.csv")
                val_ds.breast_metadata.to_csv(f"/home/harmanan/scratch/val_breast_fold{fold}.csv")
                test_ds.breast_metadata.to_csv(f"/home/harmanan/scratch/test_breast_fold{fold}.csv")

                train_ds.busbra_metadata.to_csv(f"/home/harmanan/scratch/train_busbra_fold{fold}.csv")
                val_ds.busbra_metadata.to_csv(f"/home/harmanan/scratch/val_busbra_fold{fold}.csv")
                test_ds.busbra_metadata.to_csv(f"/home/harmanan/scratch/test_busbra_fold{fold}.csv")
                
                train_dl, val_dl, test_dl = (make_dataloader(train_ds, batch_size=batch_sz), 
                                            make_dataloader(val_ds, batch_size=batch_sz), 
                                            make_dataloader(test_ds, batch_size=batch_sz))
                
                return train_dl, val_dl, test_dl

def make_ddsm_dataset(style="kfold",
                      fold=0,
                      batch_sz=64,
                      num_folds=5,
                      sampling=None,
                      sampling_ratio=1.0, 
                      chosen_concepts=list(range(25)),
                      image_sz=256,
                      augmentations=None,
                      use_llm_output=True,
                      seed=42):
    dir_ddsm = os.path.join(DATA_DIR, 'DDSM/')
    df_ddsm = pd.read_csv(os.path.join(dir_ddsm, 'metadata.csv'), keep_default_na=False)
    df_ddsm = create_concepts(df_ddsm, mode='ddsm', chosen_concepts=chosen_concepts)
    df_ddsm['y'] = df_ddsm.pathology.apply(lambda x: 1 if 'malignant' in x.lower() else 0)
    df_ddsm['BIRADS'] = df_ddsm.assessment.apply(lambda x: int(x))
    df_ddsm['split'] = df_ddsm["dcm_name"].apply(lambda x: "train" if "Training" in x else "test")
    df_ddsm['img_name'] = df_ddsm["ROI_ID"].apply(lambda x: f"/home/harmanan/projects/aip-medilab/harmanan/data/CBIS-DDSM-png/{x}.png")
    df_ddsm['report'] = df_ddsm.apply(lambda x: make_report_from_concepts(x['concepts'], modality="Mammography", mode='ddsm'), axis=1)
    
    if style == "kfold":
        train_df = df_ddsm[df_ddsm.split == 'train']
        test_df = df_ddsm[df_ddsm.split == 'test']
        skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)

        # split train_df by patient_id
        train_ids = train_df['patient_id'].unique()
        test_ids = test_df['patient_id'].unique()
        for i, (train_idx, val_idx) in enumerate(
            skf.split(train_df, train_df["y"])
        ):
            if i == fold:
                print(f"Fold {i}")
                train_df, val_df = train_df.iloc[train_idx], train_df.iloc[val_idx]

                print(f"Train: {len(train_df)}")
                print(f"Val: {len(val_df)}")
                print(f"Test: {len(test_df)}")

                #assert set(train_df.patient_id) & set(val_df.patient_id) == set()
                #assert set(train_df.patient_id) & set(test_df.patient_id) == set()
                #assert set(val_df.patient_id) & set(test_df.patient_id) == set()

                train_ds = DDSM_ROI_Dataset(data_dir=dir_ddsm, df=train_df, augs=augmentations, image_size=image_sz, use_llm_output=use_llm_output)
                val_ds = DDSM_ROI_Dataset(data_dir=dir_ddsm, df=val_df, image_size=image_sz, use_llm_output=use_llm_output)
                test_ds = DDSM_ROI_Dataset(data_dir=dir_ddsm, df=test_df, image_size=image_sz, use_llm_output=use_llm_output)

                train_dl = make_dataloader(train_ds, batch_size=batch_sz)
                val_dl = make_dataloader(val_ds, batch_size=batch_sz)
                test_dl = make_dataloader(test_ds, batch_size=batch_sz)

                return train_dl, val_dl, test_dl
    
    if style == 'kfold_patients': # same as breast-and-busbra
        # Step 1: one label per patient
        caseid_label_df = df_ddsm.groupby("patient_id").first().reset_index()[["patient_id", "y"]]
        
        # Step 2: train-test split (case-level)
        train_ids, test_ids = train_test_split(
            caseid_label_df["patient_id"],
            test_size=0.2,
            stratify=caseid_label_df["y"],
            random_state=seed
        )
        
        # Step 3: k-fold on case-level train set
        train_cases = caseid_label_df[caseid_label_df["patient_id"].isin(train_ids)].reset_index(drop=True)
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for i, (train_index, val_index) in enumerate(skf.split(train_cases, train_cases["y"])):
            if i == fold:
                # Step 4: get case IDs per fold
                train_case_ids = train_cases.iloc[train_index]["patient_id"]
                val_case_ids = train_cases.iloc[val_index]["patient_id"]
                
                # Step 5: get actual image rows from df_all
                train_tab = df_ddsm[df_ddsm["patient_id"].isin(train_case_ids)]
                val_tab = df_ddsm[df_ddsm["patient_id"].isin(val_case_ids)]
                test_tab = df_ddsm[df_ddsm["patient_id"].isin(test_ids)]
                
                # Verify no leakage
                assert set(train_tab.patient_id) & set(val_tab.patient_id) == set()
                assert set(train_tab.patient_id) & set(test_tab.patient_id) == set()
                assert set(val_tab.patient_id) & set(test_tab.patient_id) == set()
                
                if sampling == 'oversample':
                    print(f"Num. cancer cores in train: {train_tab[train_tab.y == 1].shape[0]}")
                    train_tab = oversample_cancer_data(train_tab, sampling_ratio=sampling_ratio, label_name='y')
                    print(f"Num. *resampled* cancer cores in train: {train_tab[train_tab.y == 1].shape[0]}")
                elif sampling == 'undersample':
                    print(f"Num. benign cores in train: {train_tab[train_tab.y == 0].shape[0]}")
                    train_tab = undersample_benign_data(train_tab, sampling_ratio=sampling_ratio, label_name='y')
                    print(f"Num. *resampled* benign cores in train: {train_tab[train_tab.y == 0].shape[0]}")
                
                
                train_ds = DDSM_ROI_Dataset(data_dir=dir_ddsm, df=train_tab, augs=augmentations, image_size=image_sz, use_llm_output=use_llm_output)
                val_ds = DDSM_ROI_Dataset(data_dir=dir_ddsm, df=val_tab, image_size=image_sz, use_llm_output=use_llm_output)
                test_ds = DDSM_ROI_Dataset(data_dir=dir_ddsm, df=test_tab, image_size=image_sz, use_llm_output=use_llm_output)

                train_ds.metadata.to_csv(f"/home/harmanan/scratch/train_fold{fold}.csv")
                val_ds.metadata.to_csv(f"/home/harmanan/scratch/val_fold{fold}.csv")
                test_ds.metadata.to_csv(f"/home/harmanan/scratch/test_fold{fold}.csv")
                
                train_dl = make_dataloader(train_ds, batch_size=batch_sz)
                val_dl = make_dataloader(val_ds, batch_size=batch_sz)
                test_dl = make_dataloader(test_ds, batch_size=batch_sz)
                
                return train_dl, val_dl, test_dl

def make_cub_dataset(style="holdout",
                      fold=0,
                      preprocessing='imagenet',
                      batch_sz=64,
                      num_folds=5,
                      sampling=None,
                      sampling_ratio=1.0, 
                      chosen_concepts=list(range(25)),
                      image_sz=256,
                      use_llm_output=True,
                      augmentations=None,
                      seed=42):
    dir_cub = os.path.join(DATA_DIR, 'CUB/')
    df_cub = pd.read_csv(os.path.join(dir_cub, 'metadata.csv'))
    df_cub = create_concepts(df_cub, mode='cub', chosen_concepts=chosen_concepts)

    species_label_enc = LabelEncoder()
    df_cub["Species"] = species_label_enc.fit_transform(df_cub.species_id)

    if style == "kfold":
        pass

    elif style == "holdout":
        # holdout a busbra test set
        cub_train_df = df_cub[df_cub.split == 'train']
        cub_test_df = df_cub[df_cub.split == 'test']
        cub_val_df = df_cub[df_cub.split == 'val']

        if sampling == 'divide':
            N = 5  # number of splits
            fold = fold
            assert 0 <= fold < N, f"fold must be in [0, {N-1}]"

            # Divide test_df into N parts
            subset_size = len(cub_test_df) // N
            start_idx = fold * subset_size
            end_idx = (fold + 1) * subset_size if fold < N - 1 else len(cub_test_df)

            # Select the appropriate subset for this run
            cub_test_df = cub_test_df.iloc[start_idx:end_idx].reset_index(drop=True)
            print(f"[INFO] Using test subset {fold+1}/{N} ({len(cub_test_df)} samples)")

        # now, we have the train and val dataframes for both datasets
        # build datasets
        train_ds = CUB_Dataset(data_dir=DATA_DIR, df=cub_train_df, augs=augmentations, image_size=image_sz, use_llm_output=use_llm_output, preprocessing=preprocessing)
        val_ds = CUB_Dataset(data_dir=DATA_DIR, df=cub_val_df,  image_size=image_sz, use_llm_output=use_llm_output, preprocessing=preprocessing)
        test_ds = CUB_Dataset(data_dir=DATA_DIR, df=cub_test_df, image_size=image_sz, use_llm_output=use_llm_output, preprocessing=preprocessing)

        # build dataloaders
        train_dl = make_dataloader(train_ds, batch_size=batch_sz)
        val_dl = make_dataloader(val_ds, batch_size=batch_sz)
        test_dl = make_dataloader(test_ds, batch_size=batch_sz)

        return train_dl, val_dl, test_dl




def make_paired_breast_and_busbra_dataset(style="kfold",
                                          fold=0,
                                          batch_sz=64,
                                          num_folds=5,
                                          sampling=None,
                                          sampling_ratio=1.0,
                                          augmentations=None,
                                          chosen_concepts=list(range(12)),
                                          seed=42):
    dir_breast = os.path.join(DATA_DIR, 'BrEaST/')
    dir_busbra = os.path.join(DATA_DIR, 'BUSBRA/')
    df_breast = pd.read_csv(os.path.join(dir_breast, 'metadata.csv'))
    df_breast = df_breast[df_breast.Classification != 'normal']
    df_breast = create_concepts(df_breast, chosen_concepts)
    
    df_busbra = pd.read_csv(os.path.join(dir_busbra, 'metadata.csv'))

    cancer_label_enc = LabelEncoder()
    df_busbra.Pathology = cancer_label_enc.fit_transform(df_busbra.Pathology)
    df_breast.Classification = cancer_label_enc.fit_transform(df_breast.Classification)

    birads_label_enc = LabelEncoder()
    df_breast.BIRADS = df_breast.BIRADS.apply(lambda x: x.split(' ')[0]) # remove the 'a', 'b', 'c' suffix
    df_breast.BIRADS = birads_label_enc.fit_transform(df_breast.BIRADS)
    df_busbra.BIRADS = birads_label_enc.fit_transform(df_busbra.BIRADS)

    if style == "kfold":
        breast_ids = df_breast['CaseID'].unique()
        busbra_ids = df_busbra['ID'].unique()

        breast_train_ids, breast_test_ids = train_test_split(breast_ids, test_size=0.1, random_state=seed)
        busbra_train_ids, busbra_test_ids = train_test_split(busbra_ids, test_size=0.2, random_state=seed)

        breast_train_tab  = df_breast[df_breast['CaseID'].isin(breast_train_ids)]
        busbra_train_tab  = df_busbra[df_busbra['ID'].isin(busbra_train_ids)]
        breast_test_tab = df_breast[df_breast['CaseID'].isin(breast_test_ids)]
        busbra_test_tab = df_busbra[df_busbra['ID'].isin(busbra_test_ids)]

        skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)

        # first, build the breast dataframes (train and val)
        for i, (train_idx, val_idx) in enumerate(
            skf.split(breast_train_tab, breast_train_tab["Classification"])
        ):
            print(f"Fold {i}")
            print(f"Train: {len(train_idx)}")
            print(f"Val: {len(val_idx)}")
            print(f"Test: {len(breast_test_ids)}")

            if i == fold:
                breast_train_df, breast_val_df = breast_train_tab.iloc[train_idx], breast_train_tab.iloc[val_idx]

                breast_train_idx = df_breast.CaseID.isin(breast_train_df.CaseID)
                breast_val_idx = df_breast.CaseID.isin(breast_val_df.CaseID)
                
                breast_train_tab, breast_val_tab = (
                    breast_train_tab[breast_train_idx],
                    breast_train_tab[breast_val_idx],
                )

                assert set(breast_train_tab.CaseID) & set(breast_val_tab.CaseID) == set()
                assert set(breast_train_tab.CaseID) & set(breast_test_tab.CaseID) == set()
                assert set(breast_val_tab.CaseID) & set(breast_test_tab.CaseID) == set()

                break
        # then, build the busbra dataframes (train and val)
        for i, (train_idx, val_idx) in enumerate(
            skf.split(busbra_train_tab, busbra_train_tab["Pathology"])
        ):
            print(f"Fold {i}")
            print(f"Train: {len(train_idx)}")
            print(f"Val: {len(val_idx)}")
            print(f"Test: {len(busbra_test_ids)}")
            if i == fold:
                busbra_train_df, busbra_val_df = busbra_train_tab.iloc[train_idx], busbra_train_tab.iloc[val_idx]
                busbra_train_idx = df_busbra.ID.isin(busbra_train_df.ID)
                busbra_val_idx = df_busbra.ID.isin(busbra_val_df.ID)
                busbra_train_tab, busbra_val_tab = (
                    busbra_train_tab[busbra_train_idx],
                    busbra_train_tab[busbra_val_idx],
                )
                assert set(busbra_train_tab.ID) & set(busbra_val_tab.ID) == set()
                assert set(busbra_train_tab.ID) & set(busbra_test_tab.ID) == set()
                assert set(busbra_val_tab.ID) & set(busbra_test_tab.ID) == set()
                break
        # now, we have the train and val dataframes for both datasets
        # build datasets
        train_ds = BrEaST_and_BUSBRA_Dataset(data_dir=DATA_DIR, df_breast=breast_train_tab, df_busbra=busbra_train_tab, augs=augmentations)
        val_ds = BrEaST_and_BUSBRA_Dataset(data_dir=DATA_DIR, df_breast=breast_val_tab, df_busbra=busbra_val_tab)
        test_ds = BrEaST_and_BUSBRA_Dataset(data_dir=DATA_DIR, df_breast=breast_test_tab, df_busbra=busbra_test_tab)

        # build dataloaders
        train_dl = make_dataloader(train_ds, batch_size=batch_sz)
        val_dl = make_dataloader(val_ds, batch_size=batch_sz)
        test_dl = make_dataloader(test_ds, batch_size=batch_sz)
        
        return train_dl, val_dl, test_dl
    
    elif style == "holdout":
        # holdout a busbra test set
        busbra_train_df = df_busbra[df_busbra.split == 'train']
        busbra_test_df = df_busbra[df_busbra.split == 'test']

        breast_train_ids, breast_test_ids = train_test_split(breast_ids, test_size=0.1, random_state=seed)
        breast_train_df = df_breast[df_breast['CaseID'].isin(breast_train_ids)]
        breast_test_df = df_breast[df_breast['CaseID'].isin(breast_test_ids)]

        # use kfold on breast and busbra train set
        skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
        for i, (train_idx, val_idx) in enumerate(
            skf.split(busbra_train_df, busbra_train_df["Pathology"])
        ):
            print(f"Fold {i}")
            print(f"Train: {len(train_idx)}")
            print(f"Val: {len(val_idx)}")
            if i == fold:
                busbra_train_df, busbra_val_df = busbra_train_df.iloc[train_idx], busbra_train_df.iloc[val_idx]
                busbra_train_idx = df_busbra.ID.isin(busbra_train_df.ID)
                busbra_val_idx = df_busbra.ID.isin(busbra_val_df.ID)
                busbra_train_tab, busbra_val_tab = (
                    busbra_train_df[busbra_train_idx],
                    busbra_train_df[busbra_val_idx],
                )
                assert set(busbra_train_tab.ID) & set(busbra_val_tab.ID) == set()
                assert set(busbra_train_tab.ID) & set(busbra_test_df.ID) == set()
                assert set(busbra_val_tab.ID) & set(busbra_test_df.ID) == set()
                break

        for i, (train_idx, val_idx) in enumerate(
            skf.split(df_breast, df_breast["Classification"])
        ):
            print(f"Fold {i}")
            print(f"Train: {len(train_idx)}")
            print(f"Val: {len(val_idx)}")
            if i == fold:
                breast_train_df, breast_val_df = df_breast.iloc[train_idx], df_breast.iloc[val_idx]
                breast_train_idx = df_breast.CaseID.isin(breast_train_df.CaseID)
                breast_val_idx = df_breast.CaseID.isin(breast_val_df.CaseID)
                breast_train_tab, breast_val_tab = (
                    df_breast[breast_train_idx],
                    df_breast[breast_val_idx],
                )
                assert set(breast_train_tab.CaseID) & set(breast_val_tab.CaseID) == set()
                assert set(breast_train_tab.CaseID) & set(busbra_test_df.ID) == set()
                assert set(breast_val_tab.CaseID) & set(busbra_test_df.ID) == set()
                break
        
        # now, we have the train and val dataframes for both datasets
        # build datasets
        train_ds = BrEaST_and_BUSBRA_Dataset(data_dir=DATA_DIR, df_breast=breast_train_tab, df_busbra=busbra_train_tab, augs=augmentations)
        val_ds = BrEaST_and_BUSBRA_Dataset(data_dir=DATA_DIR, df_breast=breast_val_tab, df_busbra=busbra_val_tab)
        test_ds = BrEaST_and_BUSBRA_Dataset(data_dir=DATA_DIR, df_breast=breast_test_df, df_busbra=busbra_test_df)

        return train_dl, val_dl, test_dl

    elif style == "holdout_noval":
        raise NotImplementedError("Holdout no validation is not implemented for paired datasets.")

def make_mvkl_dataset(style="kfold",
                      fold=0,
                      batch_sz=64,
                      num_folds=5,
                      sampling=None,
                      sampling_ratio=1.0,
                      augmentations=None,
                      chosen_concepts=list(range(33)),
                      seed=42):
    dir_mvkl = os.path.join(DATA_DIR, 'MVKL')
    df_mvkl = pd.read_csv(os.path.join(dir_mvkl, 'metadata.csv'), keep_default_na=False)
    df_mvkl = create_concepts(df_mvkl, mode='mammogram', chosen_concepts=chosen_concepts)

    cancer_label_enc = LabelEncoder()
    df_mvkl['Pathology'] = cancer_label_enc.fit_transform(df_mvkl.CancerNoisy)
    # df_mvkl.BIRADS = df_mvkl.BIRADS.apply(lambda x: str(x)[0])  # remove the 'a', 'b', 'c' suffix
    birads_label_enc = LabelEncoder()
    df_mvkl.BIRADS = birads_label_enc.fit_transform(df_mvkl.BIRADS)

    if style == "kfold":
        raise NotImplementedError("K-fold cross-validation is not implemented for MVKL dataset.")

    if style == 'holdout':
        # holdout a test set
        mvkl_train_df = df_mvkl[df_mvkl.split == 'train']
        mvkl_val_df = df_mvkl[df_mvkl.split == 'val']
        mvkl_test_df = df_mvkl[df_mvkl.split == 'test']

        # build datasets
        train_ds = MVKL_Dataset(data_dir=dir_mvkl, df=mvkl_train_df, augs=augmentations)
        val_ds = MVKL_Dataset(data_dir=dir_mvkl, df=mvkl_val_df)
        test_ds = MVKL_Dataset(data_dir=dir_mvkl, df=mvkl_test_df)

        # build dataloaders
        train_dl = make_dataloader(train_ds, batch_size=batch_sz)
        val_dl = make_dataloader(val_ds, batch_size=batch_sz)
        test_dl = make_dataloader(test_ds, batch_size=batch_sz)

        return train_dl, val_dl, test_dl

class CUB_Dataset(Dataset):
    def __init__(self, data_dir, df, image_size=256, use_llm_output=True,
                 preprocessing='imagenet', chosen_concepts=[1], augs=None, style="kfold"):
        self.data_dir = data_dir
        self.metadata = df
        self.augmentations = augs
        self.augment = RandomTransform(
            transforms=_make_data_augmentations(augs),
            p=0.2
        ) if augs else None

        self.translate = RandomTranslation()
        self.style = style

        self.files = {}
        self.image_size = image_size

        resize_size = image_size + 32  # 256 when image_size=224 (standard)

        if self.augmentations:
            # TRAIN
            self.transform = T.Compose([
                T.RandomResizedCrop(
                    image_size, scale=(0.7, 1.0), ratio=(3/4, 4/3)
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
            ])
        else:
            # VAL/TEST
            self.transform = T.Compose([
                T.Resize(resize_size),
                T.CenterCrop(image_size),
            ])

        if preprocessing == 'imagenet':
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.transform = T.Compose([self.transform, T.Normalize(mean=mean, std=std)])

        for i in tqdm(range(1, len(self.metadata)), "Loading images..."):
            k = max(self.files.keys()) + 1 if len(self.files) > 0 else 0
            #try:
            img_name = f'{self.metadata.image_path.iloc[i]}'
            species = self.metadata["Species"].iloc[i]
            species_name = self.metadata["species"].iloc[i]
            concepts = self.metadata["concepts"].iloc[i]
            
            try: 
                llm_output = self.metadata["llm_output"].iloc[i]
            except: 
                llm_output = make_report_from_concepts(concepts, mode="cub")
            
            if not use_llm_output:
                llm_output = generate_random_report(dataset="CUB")

            
            os.stat(img_name)
            self.files[k] = [img_name, '', species, species_name, concepts, llm_output]
            #except FileNotFoundError:
            #    continue

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        (img_name, 
        mask_name, 
        species,
        species_name, 
        concepts,
        llm_output) = self.files[idx]
        
        image = plt.imread(img_name)
        try:
            H, W, C = image.shape
        except:
            H, W = image.shape
            image = image.reshape(H,W,1)
            image = np.repeat(image, 3, axis=2)

        mask = np.zeros((H, W))

        from skimage.transform import resize

        image = resize(image, (self.image_size, self.image_size))
        mask = resize(mask, (self.image_size, self.image_size))

        if idx in  [0, 10, 20, 100, 200, 500]: # sanity check some images
            plt.savefig(f"/home/harmanan/projects/aip-medilab/harmanan/breast_us/cub_debug_{idx}.png")

        plt.savefig(f"/home/harmanan/scratch/cub_debug/{idx}.png")

        # convert to torch
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        if self.augmentations and 'translation' in self.augmentations:
            image, mask = self.translate(image, mask)
        
        return {
            'img':image, 
            'img_name': img_name,
            'mask': mask,
            'mask_name': mask_name,
            'species_name': species_name,
            'label': species,
            'concepts': concepts,
            'llm_output': llm_output
        }

class BUSBRA_Dataset(Dataset):
    def __init__(self, data_dir, df, chosen_concepts=[1], augs=None, style="kfold"):
        self.data_dir = data_dir
        self.metadata = df
        self.augmentations = augs
        self.transform = RandomTransform(
            transforms=_make_data_augmentations(augs),
            p=0.2
        ) if augs else None

        self.translate = RandomTranslation()
        self.style = style

        self.files = {}

        for i in tqdm(range(1, len(self.metadata)), "Loading images..."):
            k = max(self.files.keys()) + 1 if len(self.files) > 0 else 0
            try:
                img_name = os.path.join(data_dir, f'Images/{self.metadata.ID.iloc[i]}.png')
                mask_name = os.path.join(data_dir, f'Masks/{self.metadata.ID.iloc[i]}.png'.replace('bus', 'mask'))
                birads = self.metadata["BIRADS"].iloc[i]
                diagnosis = self.metadata["Pathology"].iloc[i]
                #concepts = np.zeros(len(chosen_concepts))
                concepts_real = self.metadata["concepts_real"].iloc[i]

                if sum(concepts_real) == -1 and self.style != "test_on_concepts": # no ground truth, use the probability guide
                    prob_guide = self.metadata["prob_guide"].iloc[i]
                    if ',' in prob_guide:
                        prob_guide = prob_guide.replace(',', '')
                    prob_guide = [float(c) for c in list(prob_guide.replace('[','').replace(']','').split(' ')) if len(c) > 0]
                    prob_guide = np.array(prob_guide)
                    concepts = prob_guide # overwrite the concepts with the probability guide
                else: 
                    concepts = concepts_real
                
                try: 
                    llm_output = self.metadata["llm_output"].iloc[i]
                except: 
                    llm_output = 'Missing report'
                    
                
                os.stat(img_name)
                os.stat(mask_name)
                self.files[k] = [img_name, mask_name, diagnosis, birads, concepts, llm_output]
            except FileNotFoundError:
                continue

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        (img_name, 
        mask_name, 
        diagnosis,
        birads, 
        concepts,
        llm_output) = self.files[idx]
        
        image = plt.imread(img_name)
        H, W = image.shape
        image = image.reshape(H,W,1)
        mask = plt.imread(mask_name)

        from skimage.transform import resize

        image = resize(image, (256, 256))
        mask = resize(mask, (256, 256))

        # convert to torch
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        image = torch.concat([image]*3)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        if self.augmentations and 'translation' in self.augmentations:
            image, mask = self.translate(image, mask)
        
        return {
            'img':image, 
            'img_name': img_name,
            'mask': mask,
            'mask_name': mask_name,
            'birads': birads,
            'label': diagnosis,
            'concepts': concepts,
            'llm_output': llm_output
        }

class BrEaST_Dataset(Dataset):
    def __init__(self, data_dir, df, augs=None):
        self.data_dir = data_dir
        self.metadata = df
        self.augmentations = augs
        self.transform = RandomTransform(
            transforms=_make_data_augmentations(augs),
            p=0.2
        ) if augs else None

        self.translate = RandomTranslation()
        self.flip = RandomFlip()
        self.crop = RandomCrop()
        self.rotate = RandomRotation()

        self.files = {}
        self.class_weights = compute_class_weights(df)

        for i in tqdm(range(1, len(self.metadata)), "Loading images..."):
            make_idx = lambda x: (3 - len(str(x)))  * '0' + str(x)
            idx = make_idx(i)
            k = max(self.files.keys()) + 1 if len(self.files) > 0 else 0

            try:
                img_name = os.path.join(data_dir, f'case{idx}.png')
                mask_name = os.path.join(data_dir, f'case{idx}_tumor.png')
                birads = self.metadata["BIRADS"].iloc[i]
                diagnosis = self.metadata["Classification"].iloc[i]
                concepts = self.metadata["concepts"].iloc[i]
                llm_output = self.metadata["llm_output"].iloc[i]
                if str(llm_output) == 'nan' or llm_output == 'None':
                    llm_output = make_report_from_concepts(concepts)
                print(f"Generated report for case {idx}")
                os.stat(img_name)
                os.stat(mask_name)
                self.files[k] = [img_name, mask_name, diagnosis, birads, concepts, llm_output]
            except FileNotFoundError:
                continue

        print(f'Loaded {len(self.files)} images')
        print(self.files.keys())
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        (img_name, 
        mask_name, 
        diagnosis,
        birads, 
        concepts,
        llm_output) = self.files[idx]
        
        image = plt.imread(img_name)[:,:,:3]
        mask = plt.imread(mask_name)

        from skimage.transform import resize

        image = resize(image, (256, 256, 3))
        mask = resize(mask, (256, 256))

        # convert to torch
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        if self.augmentations and 'translation' in self.augmentations:
            image, mask = self.translate(image, mask)
        
        return {
            'img':image, 
            'img_name': img_name,
            'mask': mask,
            'mask_name': mask_name,
            'birads': birads,
            'label': diagnosis,
            'concepts': concepts,
            'llm_output': llm_output
        }

class BrEaST_and_BUSBRA_Dataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 df_breast, 
                 df_busbra, 
                 datasets=["BrEaST", "BUSBRA"],
                 augs=None, 
                 crop_rois=False,
                 image_size=256,
                 use_llm_output=True
                 ):
        self.data_dir = data_dir
        self.breast_metadata = df_breast
        self.busbra_metadata = df_busbra
        self.augmentations = augs
        self.transform = RandomTransform(
            transforms=_make_data_augmentations(augs),
            p=0.2
        ) if augs else None

        self.translate = RandomTranslation()
        self.flip = RandomFlip()
        self.crop = RandomCrop()
        self.rotate = RandomRotation()

        self.files = {}
        self.class_weights = compute_class_weights(df_breast)
        self.crop_rois = crop_rois
        self.image_size = image_size

        # 'var': [BrEaST, BUSBRA]
        label_names = {
            'img_name': ['Image', 'ID'],
            'y': ['Classification', 'Pathology'],
        }

        if 'BrEaST' in datasets:
            for i in tqdm(range(1, len(self.breast_metadata)), "Loading BrEaST images..."):
                make_idx = lambda x: (3 - len(str(x)))  * '0' + str(x)
                idx = make_idx(i)
                k = max(self.files.keys()) + 1 if len(self.files) > 0 else 0
                try:
                    #img_name = os.path.join(data_dir, f'BrEaST{}/case{idx}.png')
                    img_name = os.path.join(data_dir, f"BrEaST{'_cropped' if self.crop_rois else ''}/case{idx}.png")
                    #mask_name = os.path.join(data_dir, f'BrEaST/case{idx}_tumor.png')
                    mask_name = os.path.join(data_dir, f"BrEaST{'_cropped' if self.crop_rois else ''}/case{idx}_tumor.png")
                    birads = self.breast_metadata["BIRADS"].iloc[i]
                    diagnosis = self.breast_metadata["Classification"].iloc[i]
                    concepts = self.breast_metadata["concepts"].iloc[i]
                    llm_output = self.breast_metadata["llm_output"].iloc[i]
                    if str(llm_output) == 'nan' or llm_output == 'None' or not use_llm_output:
                        llm_output = make_report_from_concepts(concepts)
                        if not use_llm_output:
                            llm_output = generate_random_report("BREAST_US")
                    os.stat(img_name)
                    os.stat(mask_name)
                    self.files[k] = [img_name, mask_name, diagnosis, birads, concepts, llm_output]
                except FileNotFoundError:
                    continue
        else:
            print("Skipping BrEaST dataset")
        
        if 'BUSBRA' in datasets:
            for i in tqdm(range(1, len(self.busbra_metadata)), "Loading BUSBRA images..."):
                k = max(self.files.keys()) + 1 if len(self.files) > 0 else 0
                try:
                    #img_name = os.path.join(data_dir, f'BUSBRA/Images/{self.busbra_metadata.ID.iloc[i]}.png')
                    #mask_name = os.path.join(data_dir, f'BUSBRA/Masks/{self.busbra_metadata.ID.iloc[i]}.png'.replace('bus', 'mask'))
                    img_name = os.path.join(data_dir, f'BUSBRA{"_cropped" if self.crop_rois else ""}/Images/{self.busbra_metadata.ID.iloc[i]}.png')
                    mask_name = os.path.join(data_dir, f'BUSBRA{"_cropped" if self.crop_rois else ""}/Masks/{self.busbra_metadata.ID.iloc[i]}.png'.replace('bus', 'mask'))
                    birads = self.busbra_metadata["BIRADS"].iloc[i]
                    diagnosis = self.busbra_metadata["Classification"].iloc[i]
                    concepts = self.busbra_metadata["concepts"].iloc[i]
                    llm_output = self.busbra_metadata["llm_output"].iloc[i]
                    if str(llm_output) == 'nan' or llm_output == 'None' or not use_llm_output:
                        llm_output = make_report_from_concepts(concepts)
                        if not use_llm_output:
                            llm_output = generate_random_report("BREAST_US")
                    os.stat(img_name)
                    os.stat(mask_name)
                    self.files[k] = [img_name, mask_name, diagnosis, birads, concepts, llm_output]
                except FileNotFoundError:
                    print(f"[{img_name}] not found")
                    continue
        else:
            print("Skipping BUSBRA dataset")

        print(f'Loaded {len(self.files)} images')
        print(self.files.keys())
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        (img_name, 
        mask_name, 
        diagnosis,
        birads, 
        concepts,
        llm_output) = self.files[idx]
        
        try:
            image = plt.imread(img_name)[:,:,:3]
        except IndexError:
            image = plt.imread(img_name)
        
        mask = plt.imread(mask_name)
        # if more than 1 channel, take the mean
        if len(mask.shape) > 2 and mask.shape[2] > 1:
            mask = mask.mean(axis=2, keepdims=True)

        from skimage.transform import resize

        H = W = self.image_size
        image = resize(image, (H, W, 3))
        mask = resize(mask, (H, W))

        # convert to torch
        image = image.transpose((2, 0, 1))
        if len(mask.shape) > 2:
            mask = mask.transpose((2, 0, 1))
        else:
            mask = mask.reshape(1, H, W)
        
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        #if self.transform:
        #    image = self.transform(image)
        if self.augmentations:
            if 'translation' in self.augmentations:
                image, mask = self.translate(image, mask)
            if 'flip' in self.augmentations:
                image, mask = self.flip(image, mask)
            if 'rotate' in self.augmentations:
                image, mask = self.rotate(image, mask)
            if 'crop' in self.augmentations:
                image, mask = self.crop(image, mask)

        return {
            'img':image, 
            'img_name': img_name,
            'mask': mask,
            'mask_name': mask_name,
            'birads': birads,
            'label': diagnosis,
            'concepts': concepts,
            'llm_output': llm_output
        }
        

class BrEaST_and_BUSBRA_Paired_Dataset(Dataset):
    def __init__(self, data_dir, df_breast, df_busbra, augs=None):
        self.data_dir = data_dir
        self.breast_metadata = df_breast
        self.busbra_metadata = df_busbra
        self.augmentations = augs
        self.transform = RandomTransform(
            transforms=_make_data_augmentations(augs),
            p=0.2
        ) if augs else None

        self.translate = RandomTranslation()

        self.breast_files = {}
        self.busbra_files = {}
        self.class_weights = compute_class_weights(df_breast)

        # pair breast and busbra images based on the BIRADS score

        self.busbra_metadata['concepts'] = self.busbra_metadata['ID'].apply(lambda x: -1 * np.ones_like(self.breast_metadata["concepts"].iloc[0]))
        
        # rename columns to be the same - id, img_name, birads, y, concepts
        rename_columns = lambda df: df.rename(columns={
            'CaseID': 'ID',
            'Image_filename': 'img_name',
            'BIRADS': 'birads',
            'Classification': 'y',
            'Pathology': 'y',
            'concepts': 'concepts'
        }, inplace=True)

        print("BUSBRA shape: ", self.busbra_metadata.shape)
        print("BrEaST shape: ", self.breast_metadata.shape)

        rename_columns(self.busbra_metadata)
        rename_columns(self.breast_metadata)

        # concatenate dataframes
        self.metadata = pd.concat([self.breast_metadata, self.busbra_metadata], ignore_index=True)
        self.metadata['birads'] = self.metadata['birads'].apply(lambda x: int(str(x).split(' ')[0])) # remove the 'a', 'b', 'c' suffix
        self.metadata['group'] = self.metadata.apply(lambda x: f"{x['birads']}_{x['y']}", axis=1)

        self.breast_metadata['group'] = self.breast_metadata.apply(lambda x: f"{x['birads']}_{x['y']}", axis=1)
        self.busbra_metadata['group'] = self.busbra_metadata.apply(lambda x: f"{x['birads']}_{x['y']}", axis=1)

        group_encoder = LabelEncoder()
        # fit to metadata
        group_encoder.fit(self.metadata['group'])
        # transform breast and busbra metadata
        self.breast_metadata['group'] = group_encoder.transform(self.breast_metadata['group'])
        self.busbra_metadata['group'] = group_encoder.transform(self.busbra_metadata['group'])

        print(self.breast_metadata['group'].value_counts())
        print(self.busbra_metadata['group'].value_counts())

        for i in tqdm(range(1, len(self.breast_metadata)), "Loading BrEaST images..."):
            make_idx = lambda x: (3 - len(str(x)))  * '0' + str(x)
            idx = make_idx(i)
            k = max(self.breast_files.keys()) + 1 if len(self.breast_files) > 0 else 0
            try:
                img_name = os.path.join(data_dir, f'BrEaST/case{idx}.png')
                mask_name = os.path.join(data_dir, f'BrEaST/case{idx}_tumor.png')
                birads = self.breast_metadata["birads"].iloc[i]
                diagnosis = self.breast_metadata["y"].iloc[i]
                concepts = self.breast_metadata["concepts"].iloc[i]
                group = self.breast_metadata["group"].iloc[i]
                os.stat(img_name)
                os.stat(mask_name)
                self.breast_files[k] = [img_name, mask_name, diagnosis, birads, concepts, group]
            except FileNotFoundError:
                continue

        for i in tqdm(range(1, len(self.busbra_metadata)), "Loading BUSBRA images..."):
            k = max(self.busbra_files.keys()) + 1 if len(self.busbra_files) > 0 else 0
            try:
                img_name = os.path.join(data_dir, f'BUSBRA/Images/{self.busbra_metadata.ID.iloc[i]}.png')
                mask_name = os.path.join(data_dir, f'BUSBRA/Masks/{self.busbra_metadata.ID.iloc[i]}.png'.replace('bus', 'mask'))
                birads = self.busbra_metadata["birads"].iloc[i]
                diagnosis = self.busbra_metadata["y"].iloc[i]
                concepts = -1.0 * np.ones_like(self.breast_metadata["concepts"].iloc[0])
                group = self.busbra_metadata["group"].iloc[i]
                os.stat(img_name)
                os.stat(mask_name)
                self.busbra_files[k] = [img_name, mask_name, diagnosis, birads, concepts, group]
            except FileNotFoundError:
                print(f"File not found: {img_name} or {mask_name}")
                continue
            except Exception as e:
                print(f"Some other error occurred: {e}")
                print(f"Image: {img_name}")
                continue

        print(f'Loaded {len(self.breast_files) + len(self.busbra_files)} images')
        print(self.breast_files.keys())
        print(self.busbra_files.keys())

            
    def __len__(self):
        return len(self.breast_files)
    
    def __getitem__(self, idx):
        # get the breast image
        (img_name,
        mask_name,
        diagnosis,
        birads,
        concepts,
        group) = self.breast_files[idx]

        print(f"group: {group}")

        # get the busbra images from the same group
        busbra_images_same_group = self.busbra_metadata[self.busbra_metadata['group'] == group]
        
        # randomly select one busbra image from the same group
        #busbra_images_same_group = busbra_images_same_group.sample(n=1, random_state=42)
        # get the index of the busbra image

        print(busbra_images_same_group)
        
        busbra_idx = busbra_images_same_group.index[0]

        print(busbra_idx)
        print(self.busbra_files.keys())

        (busbra_img_name,
        busbra_mask_name,
        busbra_diagnosis,
        busbra_birads,
        busbra_concepts,
        busbra_group) = self.busbra_files[busbra_idx]

        # load the breast image         
        try:
            image = plt.imread(img_name)[:,:,:3]
        except IndexError:
            image = plt.imread(img_name)

        # load the busbra image
        try:
            busbra_image = plt.imread(busbra_img_name)[:,:,:3]
        except IndexError:
            busbra_image = plt.imread(busbra_img_name)
        
        mask = plt.imread(mask_name)
        # if more than 1 channel, take the mean
        if len(mask.shape) > 2 and mask.shape[2] > 1:
            mask = mask.mean(axis=2, keepdims=True)

        from skimage.transform import resize

        image = resize(image, (256, 256, 3))
        mask = resize(mask, (256, 256))

        busbra_image = resize(busbra_image, (256, 256, 1))
        # transform the busbra image to 3 channels
        busbra_image = np.concatenate([busbra_image]*3, axis=2)
        busbra_image = busbra_image.reshape(256, 256, 3)

        # convert to torch
        image = image.transpose((2, 0, 1))
        if len(mask.shape) > 2:
            mask = mask.transpose((2, 0, 1))
        else:
            mask = mask.reshape(1, 256, 256)
        
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        busbra_image = busbra_image.transpose((2, 0, 1))
        busbra_image = torch.tensor(busbra_image, dtype=torch.float32)
        
        #if self.transform:
        #    image = self.transform(image)
        if self.augmentations:
            if 'translation' in self.augmentations:
                image, mask = self.translate(image, mask)
            if 'flip' in self.augmentations:
                image, mask = self.flip(image, mask)
            if 'rotate' in self.augmentations:
                image, mask = self.rotate(image, mask)
            if 'crop' in self.augmentations:
                image, mask = self.crop(image, mask)

        return {
            'img': (image, busbra_image),
            'img_name': (img_name, busbra_img_name),
            'mask': (mask, busbra_image),
            'mask_name': (mask_name, busbra_mask_name),
            'birads': (birads, busbra_birads),
            'label': diagnosis,
            'concepts': (concepts, busbra_concepts),
            'group': group
        }
        
class DDSM_ROI_Dataset(Dataset):
    def __init__(self, data_dir, df, image_size=256, augs=None, use_llm_output=True):
        self.data_dir = data_dir
        self.metadata = df
        self.augmentations = augs
        self.transform = RandomTransform(
            transforms=_make_data_augmentations(augs),
            p=0.2
        ) if augs else None

        self.translate = RandomTranslation()
        self.image_size = image_size

        self.files = {}
        self.class_weights = compute_class_weights(df)

        print(self.metadata)

        for i in tqdm(range(1, len(self.metadata)), "Loading images..."):
            k = max(self.files.keys()) + 1 if len(self.files) > 0 else 0
            try:
                img_name = os.path.join(data_dir, f"{self.metadata.ROI_ID.iloc[i]}.png")
                mask_name = ''
                birads = self.metadata["BIRADS"].iloc[i]
                diagnosis = self.metadata["y"].iloc[i]
                concepts = self.metadata["concepts"].iloc[i]
                roi_id = self.metadata["ROI_ID"].iloc[i]
                try:
                    llm_output = self.metadata["llm_output"].iloc[i]
                except:
                    llm_output = make_report_from_concepts(concepts, modality='Mammography', mode='ddsm')
                
                if not use_llm_output:
                    print('no llm')
                    llm_output = generate_random_report(dataset="DDSM")
                os.stat(img_name)
                self.files[k] = [img_name, mask_name, diagnosis, birads, concepts, llm_output, roi_id]
            except FileNotFoundError:
                continue

        print(f'Loaded {len(self.files)} images')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        (img_name, 
        mask_name, 
        diagnosis,
        birads, 
        concepts,
        llm_output,
        roi_id) = self.files[idx]
        
        print(img_name)
        image = plt.imread(img_name)[:,:,:3]  # ensure 3 channels
        # mask is not available in DDSM dataset

        from skimage.transform import resize

        image = resize(image, (self.image_size, self.image_size, 3))

        # convert to torch
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        if self.augmentations and 'translation' in self.augmentations:
            image, mask = self.translate(image, mask)

        return {
            'img': image, 
            'img_name': img_name,
            'mask': mask,
            'mask_name': mask_name,
            'birads': birads,
            'label': diagnosis,
            'concepts': concepts,
            'llm_output': llm_output,
            'roi_id': roi_id
        }

class MVKL_Dataset(Dataset):
    def __init__(self, data_dir, df, image_size=256, augs=None):
        self.data_dir = data_dir
        self.metadata = df
        self.augmentations = augs
        self.transform = RandomTransform(
            transforms=_make_data_augmentations(augs),
            p=0.2
        ) if augs else None

        self.translate = RandomTranslation()
        self.image_size = image_size

        self.files = {}
        self.class_weights = compute_class_weights(df)

        for i in tqdm(range(1, len(self.metadata)), "Loading images..."):
            k = max(self.files.keys()) + 1 if len(self.files) > 0 else 0
            try:
                img_name = os.path.join(data_dir, f'img/{self.metadata.img_name.iloc[i]}')
                mask_name = ''
                birads = self.metadata["BIRADS"].iloc[i]
                diagnosis = self.metadata["Pathology"].iloc[i]
                concepts = self.metadata["concepts"].iloc[i]
                llm_output = self.metadata["llm_output"].iloc[i]
                if str(llm_output) == 'nan' or llm_output == 'None':
                    llm_output = make_report_from_concepts(concepts)
                os.stat(img_name)
                self.files[k] = [img_name, mask_name, diagnosis, birads, concepts, llm_output]
            except FileNotFoundError:
                continue

        print(f'Loaded {len(self.files)} images')
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        (img_name, 
        mask_name, 
        diagnosis,
        birads, 
        concepts,
        llm_output) = self.files[idx]
        
        image = plt.imread(img_name)
        # mask is not available in MVKL dataset

        from skimage.transform import resize

        image = resize(image, (self.image_size, self.image_size, 3))

        # convert to torch
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)  # dummy mask
        
        if self.transform:
            image = self.transform(image)
        if self.augmentations and 'translation' in self.augmentations:
            image, mask = self.translate(image, mask)
        
        return {
            'img':image, 
            'img_name': img_name,
            'mask': mask,
            'mask_name': mask_name,
            'birads': birads,
            'label': diagnosis,
            'concepts': concepts,
            'llm_output': llm_output
        }


def make_dataloader(dataset, batch_size=32, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




class RandomBlur:
    def __init__(self, radius=(1, 3)):
        self.radius = radius

    def __call__(self, image):
        from torchvision.transforms.functional import gaussian_blur

        blur_radius = uniform(self.radius[0], self.radius[1])

        return gaussian_blur(image, kernel_size=int(blur_radius * 2 + 1))

class RandomBrightness:
    def __init__(self, brightness_factor=(0.5, 1.5)):
        self.brightness_factor = brightness_factor

    def __call__(self, image):
        brightness = uniform(self.brightness_factor[0], self.brightness_factor[1])
        return image * brightness

class RandomGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        noise = torch.normal(mean=self.mean, std=self.std, size=image.shape).to(image.device)
        return image + noise

class RandomSpeckleNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        noise = torch.normal(mean=self.mean, std=self.std, size=image.shape).to(image.device)
        return image + image * noise



class RandomTransform:
    def __init__(self, transforms, p=0.2):
        self.transforms = transforms
        self.p = p

    def __call__(self, *images):
        outputs = []
        self.augs = T.Compose(self.transforms)
        for image in images:
            outputs.append(self.augs(image))

        return outputs[0] if len(outputs) == 1 else outputs

class RandomFlip:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, *images):
        outputs = []
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=self.p),
            T.RandomVerticalFlip(p=self.p)

        ])
        for image in images:
            outputs.append(self.augs(image))

        return outputs[0] if len(outputs) == 1 else outputs
       
class RandomRotation:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, *images):
        outputs = []
        angle = np.random.randint(0, 360)
        self.augs = T.Compose([
            T.RandomRotation(angle),

        ])
        for image in images:
            if self.p < np.random.rand():
                outputs.append(self.augs(image))
            else:
                outputs.append(image)

        return outputs[0] if len(outputs) == 1 else outputs

class RandomCrop:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, *images):
        outputs = []
        self.augs = T.Compose([
            T.RandomCrop(size=200, padding=1, padding_mode='constant')

        ])
        for image in images:
            if self.p < np.random.rand():
                outputs.append(self.augs(image))
            else:
                outputs.append(image)

        # upsample to 256x256
        for i in range(len(outputs)):
            if outputs[i].shape[-2:] != (256, 256):
                outputs[i] = T.Resize((256, 256))(outputs[i])

        return outputs[0] if len(outputs) == 1 else outputs
    
class RandomTranslation:
    def __init__(self, translation=(0.1, 0.1)):
        self.translation = translation

    def __call__(self, *images):
        from torchvision.transforms.functional import affine
        from random import uniform

        h_factor, w_factor = uniform(
            -self.translation[0], self.translation[0]
        ), uniform(-self.translation[1], self.translation[1])

        outputs = []
        for image in images:
            H, W = image.shape[-2:]
            translate_x = int(w_factor * W)
            translate_y = int(h_factor * H)
            outputs.append(
                affine(
                    image,
                    angle=0,
                    translate=(translate_x, translate_y),
                    scale=1,
                    shear=0,
                )
            )

        return outputs[0] if len(outputs) == 1 else outputs