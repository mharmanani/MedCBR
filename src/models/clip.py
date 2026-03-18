import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open_clip 

class BaseCLIPModel(nn.Module):
    """
    Base model to load CLIP-like architectures and compute image-text embeddings and CLIP loss.
    Includes automatic handling of long text inputs via chunking + average pooling.
    """
    def __init__(self, model_name, conf, pretrained=None):
        super(BaseCLIPModel, self).__init__()
        self.device = conf.device
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )

        self.preprocess = self.preprocess_val  # default to val preprocess; can switch to train preprocess in forward() if needed
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.to(self.device)
        self.context_length = getattr(conf, "context_length", 77)  # default CLIP context length

    # ------------------------------------------------------
    # Helper: encode long text via chunk-and-average
    # ------------------------------------------------------
    def encode_long_text(self, text, freeze_text=False, max_len=None):
        """
        Tokenize long text into chunks (<= max_len tokens each),
        encode each chunk separately, then average embeddings.
        """
        max_len = self.context_length if max_len is None else max_len

        words = text.split()
        chunks = [" ".join(words[i:i + max_len]) for i in range(0, len(words), max_len)]

        text_tokens = self.tokenizer(chunks, context_length=max_len).to(self.device)

        if freeze_text:
            with torch.no_grad():
                text_embeds = self.model.encode_text(text_tokens)
        else:
            text_embeds = self.model.encode_text(text_tokens)

        return text_embeds.mean(dim=0, keepdim=True)

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------
    def forward(self, img_names, text_reports):
        text_reports = [t[0] if isinstance(t, list) else t for t in text_reports]

        # Use train or val transform depending on mode
        preprocess = self.preprocess_train if self.training else self.preprocess_val

        images = torch.stack([
            preprocess(Image.open(img).convert("RGB"))
            for img in img_names
        ]).to(self.device)

        image_feats = self.model.encode_image(images)

        text_feats = torch.cat([self.encode_long_text(t) for t in text_reports], dim=0)

        logit_scale = self.model.logit_scale.exp()
        return image_feats, text_feats, logit_scale

    # ------------------------------------------------------
    # CLIP loss
    # ------------------------------------------------------
    def apply_clip_loss(self, image_features, text_features, temperature=0.07):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits_per_image = image_features @ text_features.T / temperature
        logits_per_text = text_features @ image_features.T / temperature

        labels = torch.arange(image_features.size(0), device=image_features.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        return (loss_i2t + loss_t2i) / 2


class CLIPClassifierModel(BaseCLIPModel):
    """
    Extension of BaseCLIPModel to include concept prediction and downstream classification.
    """
    def __init__(self, model_name, conf, pretrained=None):
        super(CLIPClassifierModel, self).__init__(model_name, conf, pretrained)

        self.architecture = conf.cbm.architecture
        self.conf = conf

        if self.architecture == 'bottleneck':
            self.classifier = nn.Sequential(
                nn.Linear(conf.cbm.num_concepts, conf.cbm.num_classes)
            ).to(self.device)
        elif self.architecture == 'multitask':
            self.classifier = nn.Sequential(
                nn.Linear(conf.clip.d, conf.cbm.num_classes)
            ).to(self.device)
        elif self.architecture == 'fusion':
            self.classifier = nn.Sequential(
                nn.Linear(conf.clip.d + conf.cbm.num_concepts, conf.cbm.num_classes)
            ).to(self.device)
        else:
            self.classifier = nn.Linear(conf.clip.d, conf.cbm.num_classes).to(self.device)

        self.birads_classifier = nn.Sequential(
            nn.Linear(conf.cbm.num_concepts, conf.cbm.num_birads)
        ).to(self.device)

        self.concept_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(conf.clip.d, conf.clip.adapter_dim),
                nn.ReLU(),
                nn.Linear(conf.clip.adapter_dim, 1)
            ).to(self.device)
            for _ in range(conf.cbm.num_concepts)
        ])

    def get_downstream_pred(self, image_feats):
        if self.architecture == 'bottleneck':
            concept_preds = self.get_concept_preds(image_feats)
            return self.classifier(concept_preds)
        elif self.architecture == 'multitask':
            return self.classifier(image_feats)
        elif self.architecture == 'fusion':
            concept_preds = self.get_concept_preds(image_feats)
            combined_feats = torch.cat((image_feats, concept_preds), dim=1)
            return self.classifier(combined_feats)
        return self.classifier(image_feats)

    def get_concept_preds(self, image_feats):
        return torch.cat([clf(image_feats) for clf in self.concept_classifiers], dim=1)

    def get_birads_pred(self, concept_preds):
        return self.birads_classifier(concept_preds)

    def eval(self):
        super().eval()
        self.classifier.eval()
        self.birads_classifier.eval()
        for clf in self.concept_classifiers:
            clf.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.classifier.train(mode)
        self.birads_classifier.train(mode)
        for clf in self.concept_classifiers:
            clf.train(mode)
        return self


class BiomedCLIP(CLIPClassifierModel):
    def __init__(self, conf):
        super().__init__(
            model_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            pretrained=None,
            conf=conf
        )


class SigLIP(CLIPClassifierModel):
    def __init__(self, conf):
        super().__init__(
            model_name='ViT-B-16-SigLIP2-384',
            pretrained='webli',
            conf=conf
        )


class CLIPViT(CLIPClassifierModel):
    def __init__(self, conf):
        super().__init__(
            model_name='ViT-B-32',
            pretrained='openai',
            conf=conf
        )
        self.context_length = 77


class CLIPViTL(CLIPClassifierModel):
    def __init__(self, conf):
        super().__init__(
            model_name='ViT-L-14',
            pretrained='laion2b_s32b_b82k',
            conf=conf
        )
        self.context_length = 77


class CLIPRN50(CLIPClassifierModel):
    def __init__(self, conf):
        super().__init__(
            model_name='RN50',
            pretrained='openai',
            conf=conf
        )
        self.context_length = 77