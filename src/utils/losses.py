import torch
import torch.nn as nn

import torch.nn.functional as F

def _make_loss_function(conf):
    return {
        'concept_weighted_ce_loss': concept_weighted_ce_loss,
        'concept_weighted_bce_loss': concept_weighted_bce_loss,
        'concept_unweighted_ce_loss': concept_unweighted_ce_loss,
        'concept_mse_loss': concept_mse_loss,
        'weighted_multi_task_ce_loss': weighted_multi_task_ce_loss,
        'unweighted_multi_task_ce_loss': unweighted_multi_task_ce_loss,
        'ce_and_concept_mse_loss': joint_ce,
        'cross_entropy': F.cross_entropy,
        'birads_ce': F.cross_entropy,
        'birads_multi_task_ce_loss': birads_multi_task_ce_loss,
        'birads_ce_and_concept_mse_loss': birads_ce_and_concept_mse_loss,
        'supcon': SupConLoss(),
    }[conf.loss]

def dice_loss(y_pred, y_true, eps=1e-7):
    """
    Sørensen–Dice coefficient, also known as the Dice similarity index.
    The dice loss is defined as 1 - dice coefficient.
    """
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = 1 - (2 * intersection + eps) / (union + eps)
    return dice

def bce_dice_loss(y_pred, y_true, eps=1e-7):
    """
    The sum of binary cross-entropy loss and the dice loss.
    """
    bce = F.binary_cross_entropy(y_pred, y_true)
    dice = dice_loss(y_pred, y_true)
    return bce + dice

def joint_ce(y_pred, y_true, c_pred, c_true, gamma=0.8, device='cuda'):
    """
    The sum of the BIRADS cross-entropy and concept cross-entropy.
    """
    l_birads = F.cross_entropy(y_pred, y_true)
    l_concepts = concept_mse_loss(c_pred, c_true, device=device)
    return l_birads + gamma*l_concepts

import torch

def concept_mse_loss(c_pred, c_true, device='cuda'):
    """
    Computes mean squared error loss for concept tasks.

    Args:
        c_pred (torch.Tensor): Shape (B, N_c, C_c), concept task logits.
        c_true (torch.Tensor): Shape (B, N_c), concept task labels.
        device (str): Device for computation (default: 'cuda').

    Returns:
        torch.Tensor: Total MSE loss.
    """
    c_pred = c_pred.to(device)
    c_true = c_true.to(device)
    return F.mse_loss(c_pred, c_true)

def one_hot_encode(labels, N):
    """
    Converts a B-dimensional label vector into a BxN one-hot encoded matrix.

    Args:
        labels (torch.Tensor): Shape (B,), containing class indices in {0, ..., N-1}.
        N (int): Number of classes.

    Returns:
        torch.Tensor: Shape (B, N), one-hot encoded labels.
    """
    B = labels.shape[0]
    one_hot = torch.zeros((B, N), dtype=torch.float32, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

def concept_unweighted_ce_loss(c_pred, c_true, **kwargs):
    """
    Computes unweighted concept cross-entropy loss.

    Args:
        c_pred (torch.Tensor): Shape (B, N_c, C_c), concept task logits.
        c_true (torch.Tensor): Shape (B, N_c), concept task labels.
        device (str): Device for computation (default: 'cuda').

    Returns:
        torch.Tensor: Total unweighted loss.
    """
    ce_tot = 0
    B, N_c = c_pred.shape  # c_pred is (B, N_c, C_c), where C_c varies per task
    for i in range(N_c):
        ci_pred = c_pred[:, i].unsqueeze(1)  # Shape (B, C_c) - logits for task i
        ci_pred = torch.cat((1-ci_pred, ci_pred), dim=1)
        ci_true = c_true[:, i]     # Shape (B,) - true labels for task i
        
        ce_i = F.cross_entropy(ci_pred, ci_true.long())
        ce_tot += ce_i

    return ce_tot * 1.0 / N_c

def concept_weighted_ce_loss(c_pred, 
                           c_true, 
                           concept_weights,
                           device='cuda'):
    wce_tot = 0
    B, N_c = c_pred.shape  # c_pred is (B, N_c, C_c), where C_c varies per task

    assert len(concept_weights) == N_c, \
        f"Expected {N_c} concept weights, got {len(concept_weights)}"

    for i in range(N_c):
        ci_pred = c_pred[:, i].unsqueeze(1)  # Shape (B, C_c) - logits for task i
        ci_pred = torch.cat((1-ci_pred, ci_pred), dim=1)
        ci_true = c_true[:, i]     # Shape (B,) - true labels for task i
        ci_wgt = concept_weights[i]
        
        ce_i = F.cross_entropy(ci_pred, ci_true.long())
        wce_tot += ce_i * ci_wgt

    return wce_tot 


def concept_weighted_bce_loss(c_pred, c_true, concept_weights, device='cuda'):
    """
    Args:
        c_pred: Tensor of shape (B, N_c) - predicted logits for binary concepts
        c_true: Tensor of shape (B, N_c) - true binary labels (0 or 1)
        concept_weights: Tensor of shape (N_c,) - weight for each concept
    Returns:
        Scalar weighted BCE loss
    """
    assert c_pred.shape == c_true.shape, "Shape mismatch between predictions and targets"
    assert c_pred.shape[1] == len(concept_weights), "Mismatch in number of concepts"

    c_pred = c_pred.to(device)
    c_true = c_true.to(device)
    concept_weights = torch.tensor(concept_weights, dtype=torch.float32, device=device)

    # Compute BCE loss per sample per concept
    bce_loss = F.binary_cross_entropy_with_logits(c_pred, c_true, reduction='none')  # shape: (B, N_c)

    # Apply concept weights (broadcast over batch dimension)
    weighted_loss = bce_loss * concept_weights  # shape: (B, N_c)

    # Mean over batch and concepts
    return weighted_loss.mean()


def birads_multi_task_ce_loss(y_pred, y_true, b_pred, b_true, device='cuda', gamma=1.0):
    y_ce = F.cross_entropy(y_pred, y_true)
    b_ce = F.cross_entropy(b_pred, b_true)
    return y_ce + gamma * b_ce

def weighted_multi_task_ce_loss(y_pred, 
                           y_true, 
                           c_pred, 
                           c_true, 
                           concept_weights,
                           device='cuda',
                           gamma=0.8):
    """
    Computes weighted multi-task cross-entropy loss.

    Args:
        y_pred (torch.Tensor): Shape (B, C_main), main task logits.
        y_true (torch.Tensor): Shape (B,), main task labels.
        c_pred (torch.Tensor): Shape (B, N_c, C_c), concept task logits.
        c_true (torch.Tensor): Shape (B, N_c), concept task labels.
        concept_weights (list of torch.Tensor): List of class weight tensors for each concept task.
        device (str): Device for computation (default: 'cuda').
        gamma (float): Weighting factor for concept loss.

    Returns:
        torch.Tensor: Total weighted loss.
    """
    wce_tot = concept_weighted_ce_loss(c_pred, c_true, device=device, concept_weights=concept_weights)

    print('debugging loss function')
    print(y_pred)
    print(y_pred.shape)
    print(y_true)
    print(y_true.shape)

    return gamma * wce_tot + F.cross_entropy(y_pred, y_true)

def unweighted_multi_task_ce_loss(y_pred, 
                                  y_true, 
                                  c_pred, 
                                  c_true, 
                                  gamma=0.8,
                                  device='cuda'):
    """
    Computes unweighted multi-task cross-entropy loss.

    Args:
        y_pred (torch.Tensor): Shape (B, C_main), main task logits.
        y_true (torch.Tensor): Shape (B,), main task labels.
        c_pred (torch.Tensor): Shape (B, N_c, C_c), concept task logits.
        c_true (torch.Tensor): Shape (B, N_c), concept task labels.
        device (str): Device for computation (default: 'cuda').

    Returns:
        torch.Tensor: Total unweighted loss.
    """
    wce_tot = concept_unweighted_ce_loss(c_pred, c_true, device=device)
    if y_pred.shape != y_true.shape:
        y_true = y_true.flatten()
    return gamma*wce_tot + F.cross_entropy(y_pred, y_true)

def birads_ce_and_concept_mse_loss(y_pred, y_true,
                                   b_pred, b_true,
                                   c_pred, c_true, device='cuda', gamma=0.8):
    """
    Computes the sum of BIRADS cross-entropy loss and concept MSE loss.
    Args:
        y_pred (torch.Tensor): Shape (B, C_main), main task logits.
        y_true (torch.Tensor): Shape (B,), main task labels.
        b_pred (torch.Tensor): Shape (B, C_birads), BIRADS task logits.
        b_true (torch.Tensor): Shape (B,), BIRADS task labels.
        c_pred (torch.Tensor): Shape (B, N_c, C_c), concept task logits.
        c_true (torch.Tensor): Shape (B, N_c), concept task labels.
        device (str): Device for computation (default: 'cuda').
        gamma (float): Weighting factor for concept loss.
    Returns:
        torch.Tensor: Total loss.
    """
    y_ce = F.cross_entropy(y_pred, y_true)
    b_ce = F.cross_entropy(b_pred, b_true)
    c_mse = concept_mse_loss(c_pred, c_true, device=device)
    return 0.5 * (1-gamma) * y_ce + 0.5 * (1-gamma) * b_ce + gamma * c_mse

def sinkhorn(out, epsilon=1e-9, num_iters=3, device='cuda'):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    #dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(num_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        #dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()



"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, eps=1e-12):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        print(features.shape)
        print(labels.shape)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        if torch.isnan(loss).any():
            print("NaN detected in SupConLoss")

        return loss