import torch
import numpy as np
from tamer.datamodule.vocab import vocab


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    mask = np.ones(m, dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    i, j = m, n
    operations = []
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            mask[i-1] = 0
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            j -= 1
        else:
            if i > 0 and j > 0:
                if s1[i - 1] == s2[j - 1]:
                    operations.append({'action': 'KEEP', 'index': i, 'value': 'SELF'})
                else:
                    operations.append({'action': 'REPLACE', 'index': i, 'value': s2[j-1]})
                    mask[i-1] = 0
            i -= 1
            j -= 1


    return mask

def generate_new_label(
        pred: list,
        gt: list,
        device: torch.device
    )->torch.Tensor:
    """ generate new label contains error mask
    
    Args:
        pred (list)
        gt (list)

    Returns:
        error_mask (list)
    """
    error_mask_list = []
    
    for p, g in zip(pred, gt):
        error_mask = edit_distance(p, g)
        error_mask_list.append(error_mask)

    return error_mask_list

def generate_new_label_set(
        pred: list,
        gt: list,
        device: torch.device
    )->torch.Tensor:
    """ generate new label contains error mask
    
    Args:
        pred (list)
        gt (list)

    Returns:
        pred_paddded (Tensor)
        pred_mask (Tensor)
        error_mask (Tensor)
    """
    error_mask_list = []
    
    for p, g in zip(pred, gt):
        error_mask = edit_distance(p, g)
        error_mask_list.append(error_mask)

    max_len = max([len(p) for p in pred])
    n_samples = len(pred)

    pred_padded = torch.zeros(n_samples, max_len, dtype=torch.long, device=device)
    pred_mask = torch.ones(n_samples, max_len, dtype=torch.bool, device=device)
    error_mask = torch.full((n_samples, max_len), -1, dtype=torch.long, device=device)

    for i, p in enumerate(pred):
        pred_padded[i, :len(p)] = torch.tensor(p, device=device)
        pred_mask[i, :len(p)] = 0
        error_mask[i, :len(p)] = torch.tensor(error_mask_list[i], device=device)

    return pred_padded, pred_mask, error_mask


def generate_default_label_set(
        n_samples: int,
        device: torch.device
    )->torch.Tensor:
    """ generate default label set
    
    Args:
        n_samples (int)
        device (torch.device)

    Returns:
        pred_paddded (Tensor)
        pred_mask (Tensor)
        error_mask (Tensor)
    """
    pred_padded = torch.full((n_samples, 1), vocab.MSK_IDX, dtype=torch.long, device=device)
    pred_mask = torch.zeros(n_samples, 1, dtype=torch.bool, device=device)
    error_mask = torch.zeros(n_samples, 1, dtype=torch.long, device=device)

    return pred_padded, pred_mask, error_mask

def out_hat_to_pred(
        out_hat: torch.Tensor,
        threshold: float
    )->torch.Tensor:
    """ convert out_hat to pred
    
    Args:
        out_hat (Tensor)
        threshold (float)
        probility 

    Returns:
        pred (Tensor)
    """
    out_hat = out_hat[: out_hat.size(0) // 2]
    top2_val, top2_indices = out_hat.topk(2, dim=-1)
    diff = top2_val[:, :, 0] - top2_val[:, :, 1]
    negative_examples = top2_indices[:, :, 0].clone()
    negative_examples[diff < threshold] = top2_indices[:, :, 1][diff < threshold]
    pred = negative_examples.tolist()

    for i in range(len(pred)):
        if vocab.EOS_IDX in pred[i]:
            pred[i] = pred[i][:pred[i].index(vocab.EOS_IDX)]
            if len(pred[i]) == 0:
                pred[i] = [vocab.MSK_IDX]

    return pred