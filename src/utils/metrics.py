import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.concept_bank.BREAST_US import BREAST_US_CONCEPT_BANK
from src.utils.concept_bank.DDSM import DDSM_CONCEPT_BANK
from src.utils.concept_bank.CUB import CUB_CONCEPT_BANK


CONCEPT_BANK = {
    "BREAST_US": {i: c for i, c in enumerate(BREAST_US_CONCEPT_BANK)},
    "BrEaST": {i: c for i, c in enumerate(BREAST_US_CONCEPT_BANK)},
    "BUSBRA": {i: c for i, c in enumerate(BREAST_US_CONCEPT_BANK)},
    "DDSM": {i: c for i, c in enumerate(DDSM_CONCEPT_BANK)},
    "CUB": {i: c for i, c in enumerate(CUB_CONCEPT_BANK)},
}


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_int01_tensor(x):
    if isinstance(x, list):
        x = torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return (x > 0.5).int()


def _safe_binary_auc(y_true, y_score):
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)
    if len(np.unique(y_true)) <= 1:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_bal_acc_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    bal_acc_values = [balanced_accuracy_score(y_true, y_score > thr) for thr in thresholds]
    best_idx = int(np.argmax(bal_acc_values))
    return float(thresholds[best_idx]), float(bal_acc_values[best_idx])


def _get_concept_names(dataset="breast_us"):
    bank = CONCEPT_BANK.get(dataset, {})
    return {idx: str(concept).replace(" ", "_").lower() for idx, concept in bank.items()}


def top_k_accuracy(y_true, y_pred, k=5):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim != 2:
        raise ValueError(f"y_pred must be 2D (N, C), got shape {y_pred.shape}")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"N mismatch: y_true {y_true.shape[0]} vs y_pred {y_pred.shape[0]}")

    c = y_pred.shape[1]
    if not (1 <= k <= c):
        raise ValueError(f"k must be in [1, {c}], got {k}")

    topk_idx = np.argpartition(y_pred, -k, axis=1)[:, -k:]
    correct = (topk_idx == y_true[:, None]).any(axis=1)
    return float(correct.mean())


def dice_score_coefficient(y_pred, y_true, threshold=None, eps=1e-7):
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)

    if threshold is not None:
        y_pred = (y_pred > threshold).astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2 * intersection + eps) / (union + eps)


def jaccard_index(y_pred, y_true, eps=1e-7):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return (intersection + eps) / (union + eps)


def visualize_predictions(x, y, y_pred):
    x = x[0].permute(1, 2, 0).cpu().numpy()
    y = y[0].permute(1, 2, 0).cpu().numpy()
    y_pred = y_pred[0].permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(x, cmap="gray", interpolation="none")
    ax[1].imshow(x, cmap="gray")
    ax[1].imshow(y, cmap="jet", interpolation="none", alpha=0.5)
    ax[2].imshow(x, cmap="gray")
    ax[2].imshow(y_pred, cmap="jet", interpolation="none", alpha=0.5)
    return fig, ax


def show_confmat(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large")
    return fig, ax


def show_reconstruction(y_true, y_pred):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
    ax[0].imshow(y_true[0, 0], cmap="gray")
    ax[0].set_title("Ground Truth")
    ax[1].imshow(y_pred[0, 0], cmap="gray")
    ax[1].set_title("MAE Prediction")
    return fig, ax


def compute_conceptwise_metrics(c_true, c_pred, selected_concepts=None, desc="test", dataset="breast_us"):
    c_pred = torch.from_numpy(c_pred) if isinstance(c_pred, np.ndarray) else c_pred
    c_true = _to_int01_tensor(c_true)

    num_categories = c_true.shape[1]
    concept_names = _get_concept_names(dataset)

    if selected_concepts is None:
        selected_concepts = list(range(num_categories))

    accuracy_per_concept = []
    bal_accuracy_per_concept = []
    auroc_per_concept = []

    for i in range(num_categories):
        pred_concept = c_pred[:, i]
        label_concept = c_true[:, i]

        binary_preds = (pred_concept >= 0.5).float()
        accuracy = (binary_preds == label_concept).float().mean().item()

        thresh, max_bal_acc = _safe_bal_acc_threshold(_to_numpy(label_concept), _to_numpy(pred_concept))
        print("Max balanced accuracy: {:.2f} at threshold {:.2f}".format(max_bal_acc, thresh))

        bal_accuracy = balanced_accuracy_score(_to_numpy(label_concept), _to_numpy(pred_concept) > thresh)
        auroc = _safe_binary_auc(_to_numpy(label_concept), _to_numpy(pred_concept))

        accuracy_per_concept.append(float(accuracy))
        bal_accuracy_per_concept.append(float(bal_accuracy))
        auroc_per_concept.append(float(auroc))

    metrics = {}
    for i in range(len(accuracy_per_concept)):
        concept_idx = int(selected_concepts[i]) if i < len(selected_concepts) else i
        concept_name = concept_names.get(concept_idx, f"concept_{concept_idx}")
        metrics[f"{concept_name}_accuracy"] = accuracy_per_concept[i]
        metrics[f"{concept_name}_bal_accuracy"] = bal_accuracy_per_concept[i]
        metrics[f"{concept_name}_auroc"] = auroc_per_concept[i]

    metrics["concept_auc"] = float(np.nanmean(auroc_per_concept))
    metrics["concept_acc"] = float(np.mean(accuracy_per_concept))
    metrics["concept_bacc"] = float(np.mean(bal_accuracy_per_concept))

    return metrics


def compute_classification_metrics(y_true, y_prob, desc="test", tune_threshold=True, multi_class=False, save_threshold=None):
    y_true = _to_numpy(y_true).reshape(-1)
    y_prob = _to_numpy(y_prob)

    if multi_class:
        y_prob = torch.softmax(torch.from_numpy(y_prob), dim=1).numpy()
        y_pred = np.argmax(y_prob, axis=1)

        try:
            auroc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError as e:
            print(f"Warning: AUROC calculation failed: {e}")
            auroc = float("nan")

        return {
            "auc": float(auroc),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "bal_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }

    y_prob = torch.sigmoid(torch.from_numpy(y_prob)).numpy()

    thresh, max_bal_acc = _safe_bal_acc_threshold(y_true, y_prob[:, 1])
    if save_threshold:
        torch.save(thresh, save_threshold)

    thresh40 = min(
        np.unique(y_prob[:, 1]),
        key=lambda x: abs(recall_score(y_true, y_prob[:, 1] > x, pos_label=0) - 0.6),
    )
    sens_at_40 = recall_score(y_true, y_prob[:, 1] > thresh40, pos_label=1)

    thresh60 = min(
        np.unique(y_prob[:, 1]),
        key=lambda x: abs(recall_score(y_true, y_prob[:, 1] > x, pos_label=0) - 0.4),
    )
    sens_at_60 = recall_score(y_true, y_prob[:, 1] > thresh60, pos_label=1)

    thresh80 = min(
        np.unique(y_prob[:, 1]),
        key=lambda x: abs(recall_score(y_true, y_prob[:, 1] > x, pos_label=0) - 0.2),
    )
    sens_at_80 = recall_score(y_true, y_prob[:, 1] > thresh80, pos_label=1)

    auroc = roc_auc_score(y_true, y_prob[:, 1])
    bal_accuracy = balanced_accuracy_score(y_true, y_prob[:, 1] > thresh)
    sensitivity = recall_score(y_true, y_prob[:, 1] > 0.5, pos_label=1)
    specificity = recall_score(y_true, y_prob[:, 1] > 0.5, pos_label=0)

    return {
        "auc": float(auroc),
        "bal_accuracy": float(bal_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "sensitivity_at_40": float(sens_at_40),
        "sensitivity_at_60": float(sens_at_60),
        "sensitivity_at_80": float(sens_at_80),
    }


def compute_multiclass_metrics(b_true, b_prob, desc="test", tune_threshold=True):
    b_true = _to_numpy(b_true)
    b_prob = _to_numpy(b_prob)

    metrics = {}
    num_classes = b_prob.shape[1]

    for i in range(num_classes):
        y_true_bin = (b_true == i).astype(int)
        y_score_bin = b_prob[:, i]

        fpr, tpr, threshold_list = roc_curve(y_true_bin, y_score_bin)

        if tune_threshold:
            bal_acc_list = [balanced_accuracy_score(y_true_bin, y_score_bin > thr) for thr in threshold_list]
            best_thr = threshold_list[int(np.argmax(bal_acc_list))]
        else:
            best_thr = 0.5

        auroc = roc_auc_score(y_true_bin, y_score_bin)
        bal_acc = balanced_accuracy_score(y_true_bin, y_score_bin > best_thr)
        bal_acc_05 = balanced_accuracy_score(y_true_bin, y_score_bin > 0.5)

        metrics[f"birads_{i}_auroc"] = float(auroc)
        metrics[f"birads_{i}_bal_accuracy"] = float(bal_acc)
        metrics[f"birads_{i}_bal_accuracy_05"] = float(bal_acc_05)

    return metrics