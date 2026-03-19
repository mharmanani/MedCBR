<div align="center">
    <img src=".img/medCBR_logo_wide.png" width="50%">
</div>

# Vision-Language Models Encode Clinical Guidelines for Concept-Based Reasoning in Medical Image Analysis

<div>
<p>
    <a href="https://arxiv.org/abs/2603.08921"><img src="https://img.shields.io/badge/-2603.08921-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
    <a href="https://harmanani.com/MedCBR"><img src="https://img.shields.io/badge/-Project%20Page-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
    <a href="#"><img src="https://img.shields.io/badge/-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"></a>
    <a href="https://pytorch.org"><img src="https://img.shields.io/badge/-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="License: MIT"></a>
    <img src="https://komarev.com/ghpvc/?username=mharmanani&repo=medcbr&style=for-the-badge&label=visitors&color=blueviolet" alt="Visitors">
</p>
</div>

#### Abstract
Concept Bottleneck Models (CBMs) are a prominent framework for interpretable AI that map learned visual features onto a set of meaningful concepts, to be used for task-specific downstream predictions. Their sequential structure enhances transparency by connecting model predictions to the underlying concepts that support them. In medical imaging, where transparency is essential, CBMs offer an appealing foundation for explainable model design. However, their discrete concept representations overlook broader clinical context such as diagnostic guidelines and expert heuristics, reducing reliability in complex cases. We propose MedCBR, a concept-based reasoning framework that integrates clinical guidelines with vision–language and reasoning models. Labeled clinical descriptors are transformed into guideline-conformant text, and a concept-based model is trained with a multi-task objective combining multi-modal contrastive alignment, concept supervision, and diagnostic classification to jointly ground image features, concepts, and pathology. A reasoning model then converts these predictions into structured clinical narratives that explain the diagnosis, emulating expert reasoning based on established guidelines. MedCBR achieves superior diagnostic and concept-level performance, with AUROCs of 94.2% on ultrasound and 84.0% on mammography. Further experiments were also performed on non-medical datasets, with 86.1% accuracy. Our framework enhances interpretability and forms an end-to-end bridge from medical image analysis to decision-making.

---

###  Dataset setup
##### BUSBRA & BrEaST
1. Download both datasets
2. Move the images into the correct directory
3. Preprocess the data by cropping to lesion ROIs
```python3
utils/preprocess.py -d busbra -src <src_directory> -dst <target_directory>
```

##### CBIS-DDSM
1. Download the dataset
2. Preprocess the data by cropping to lesion ROIs, resizing, and grouping ROIs by patient
3. Move the images into the correct directory

##### CUB-200-2011
1. Download the data
2. Move the images into the correct directory

--- 

### Model Training
Our code base supports the following models. Any model not mentioned here was trained using code provided by its creators. 

#### Supported Models
**Black-box Vision Models**:
- CLIP RN50 
- CLIP ViT-B/32
- CLIP ViT-L/14
- SigLIP
- BiomedCLIP

**Concept-based models**:
- Original CBM (Koh et al., 2020)
- CLIP CBM

To run a model using k-fold cross-validation, we used:
```bash
scripts/run_job.sh -y <yaml_name> -g <wandb_group_name> -f <fold> -t <time>
```
To run a model using different random seeds, we used:
```bash
scripts/run_job.sh -y <yaml_name> -g <wandb_group_name> -s <seed> -t <time>
```

#### Unsupported Models
The following models must be cloned from their respective repositories and run according to the instructions provided by their authors:
- PCBM, PCBMh (Yuksekgonul et al., ICLR 2023)
- Label-free CBM (Oikarinen et al., ICLR 2023)
- AdaCBM (Chowdhury et al., MICCAI 2024)

---

### Synthetic Report Generation

To generate a synthetic report, we run the following code:
```bash
scripts/run_job.sh -y <dataset>/qwen2vl -g <wandb_group_name> -f <fold> -t <time>
```
This will run the Qwen2.5VL LVLM on the test set of that fold. The 3 YAMLs `busbra/qwen2vl`, `ddsm/qwen2vl`, and `cub/qwen2vl` specify the hyperparameters to use for this run. 

The `src/run_qwenvl.py` file contains the logic used to run the LVLM on the data to generate reports. The model and prompt are both implemented in `src/models/qwen.py`, and the guidelines are recorded in `src/utils/clinical_guideline.py` (although some of the guidelines are non-clinical).

### CLIP Concept Training
To train a multi-task concept model using CLIP, we use the following code:
```bash
scripts/run_job.sh -y <dataset>/medcbr -g <wandb_group_name> -f <fold> -t <time>
```
This will train a train a CLIP model with ViT-L/14 backbone on the data. The hyperparameters are found in YAMLs `busbra/medcbr`, `ddsm/medcbr`, and `cub/medcbr`. In particular, the hyperparameter
```yaml
clip:
    ...
    use_llm_output: true
    ...
```
is responsible for enabling CLIP-based training on the guideline-conformant reports generated by the LVLM. The weights for the multi-task loss are also recorded as:
```yaml
  clip_weight: 1.0
  det_weight: 5.0
  concept_weight: 1.0
```

To train a CLIP CBM baseline, use the following script:
```bash
scripts/run_job.sh -y <dataset>/clip_cbm -g <wandb_group_name> -f <fold> -t <time>
```
which has the following values in the YAML:
```yaml
  clip_weight: 0.0
  det_weight: 1.0
  concept_weight: 1.0
  use_llm_output: false
```

### Clinical Reasoning
Finally, to generate the reasoning, run the following:
```bash
scripts/run_job.sh -y <dataset>/medcbr_reason -g <wandb_group_name> -f <fold> -t <time>
```
This will instantiate a CLIP ViT-L/14 vision backbone trained using the previous step, and use its predictions as input to a Qwen3 LRM. The `src/run_reasoning.py` file details the process of generating the clinical narratives, and the LRM is implemented in `src/models/reasoning.py`.
