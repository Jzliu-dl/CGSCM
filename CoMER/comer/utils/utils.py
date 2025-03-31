from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from comer.datamodule import vocab
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric


class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
        for pred, truth in zip(indices_hat, indices):
            pred = vocab.indices2label(pred)
            truth = vocab.indices2label(truth)

            is_same = pred == truth

            if is_same:
                self.rec += 1

            self.total_line += 1

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line
        return exp_rate


def ce_loss(
    output_hat: torch.Tensor,
    output: torch.Tensor,
    ignore_idx: int = vocab.PAD_IDX,
    reduction: str = "mean",
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction)
    return loss

def focal_loss_with_ignore(
    confidence: torch.Tensor,
    target_confidence: torch.Tensor,
    ignore_value: int = -1,
    reduction: str = "mean",
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """
    Compute focal loss for confidence scores with ignore value.

    Args:
        confidence (torch.Tensor): [batch, len, 1]
        target_confidence (torch.Tensor): [batch, len]
        ignore_value (int): Value to ignore in the target tensor
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'
        gamma (float): Focusing parameter for Focal Loss. Default: 2.0
        alpha (float): Balance parameter for positive and negative classes. Default: 0.25

    Returns:
        torch.Tensor: loss value
    """
    # 调整维度
    flat_confidence = rearrange(confidence, "b l -> (b l)")
    flat_target = rearrange(target_confidence, "b l -> (b l)").float()

    # 创建掩码，忽略填充值
    mask = (flat_target != ignore_value).float()

    # 计算预测概率（sigmoid 激活）
    prob = torch.sigmoid(flat_confidence)
    
    # 根据 target 计算 alpha 和 pt
    alpha_t = flat_target * alpha + (1 - flat_target) * (1 - alpha)
    prob_t = flat_target * prob + (1 - flat_target) * (1 - prob)

    # 计算 Focal Loss 的动态缩放因子
    focal_weight = alpha_t * (1 - prob_t).pow(gamma)

    # 计算 BCE Loss
    bce_loss = F.binary_cross_entropy_with_logits(flat_confidence, flat_target, reduction="none")

    # 应用 Focal Loss 的权重
    loss = focal_weight * bce_loss

    # 应用掩码
    loss = loss * mask

    # 计算损失
    if reduction == "mean":
        loss = loss.sum() / mask.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def to_tgt_output(
    tokens: Union[List[List[int]], List[LongTensor]],
    direction: str,
    device: torch.device,
    pad_to_len: Optional[int] = None,
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : Union[List[List[int]], List[LongTensor]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    if isinstance(tokens[0], list):
        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    if direction == "l2r":
        tokens = tokens
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]

    length = max(lens) + 1
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1 : (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_bi_tgt_out(
    tokens: List[List[int]], device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out
