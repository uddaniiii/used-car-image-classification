import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

# -----------------------------
# 1. 기본 Cross Entropy
# -----------------------------
def cross_entropy(reduction='mean'):
    return nn.CrossEntropyLoss(reduction=reduction)

# -----------------------------
# 2. Label Smoothing
# -----------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# -----------------------------
# 3. Focal Loss
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# -----------------------------
# 4. Asymmetric Loss (ASL)
# -----------------------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x = torch.softmax(x, dim=-1)
        xs_pos = x.gather(dim=-1, index=y.unsqueeze(1)).squeeze(1)
        xs_neg = 1 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = torch.clamp(xs_neg + self.eps, max=1 - self.clip)

        loss = -torch.log(xs_pos + self.eps) * (1 - xs_pos) ** self.gamma_pos \
               - torch.log(xs_neg + self.eps) * xs_neg ** self.gamma_neg
        return loss.mean()

# -----------------------------
# 5. Poly1 Cross Entropy
# -----------------------------
class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
        poly1 = ce + self.epsilon * (1 - pt)
        return poly1.mean()

# -----------------------------
# 6. Bi-Tempered Logistic Loss
# -----------------------------
def log_t(u, t):
    if t == 1.0:
        return torch.log(u)
    else:
        return (u ** (1 - t) - 1) / (1 - t)

def exp_t(u, t):
    if t == 1.0:
        return torch.exp(u)
    else:
        return torch.clamp((1 + (1 - t) * u), min=0) ** (1 / (1 - t))

class BiTemperedLoss(nn.Module):
    def __init__(self, t1=0.7, t2=1.3, label_smoothing=0.1):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        num_classes = logits.size(-1)
        labels_onehot = F.one_hot(labels, num_classes).float()
        if self.label_smoothing > 0.0:
            labels_onehot = (1 - self.label_smoothing) * labels_onehot + self.label_smoothing / num_classes

        probabilities = F.softmax(logits, dim=-1)
        loss = (
            torch.sum(
                labels_onehot * (log_t(labels_onehot, self.t1) - log_t(probabilities, self.t1)), dim=-1
            )
            - 1 / (2 - self.t1) * torch.sum(
                labels_onehot ** (2 - self.t1), dim=-1
            )
            + 1 / (2 - self.t2) * torch.sum(
                probabilities ** (2 - self.t2), dim=-1
            )
        )
        return loss.mean()

# -----------------------------
# 7. Taylor Cross Entropy Loss
# -----------------------------
class TaylorCrossEntropyLoss(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.n = n

    def forward(self, pred, target):
        one_hot = F.one_hot(target, num_classes=pred.size(1)).float()
        prob = F.softmax(pred, dim=1)
        taylor_approx = 1 - sum([(torch.pow(prob, i + 1)) / math.factorial(i + 1) for i in range(self.n)])
        return (one_hot * taylor_approx).sum(dim=1).mean()

# -----------------------------
# 8. Symmetric Cross Entropy Loss
# -----------------------------
class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred_softmax = F.softmax(pred, dim=1)
        log_pred = torch.log(pred_softmax + 1e-7)
        one_hot = F.one_hot(target, num_classes=pred.size(1)).float()
        ce = F.cross_entropy(pred, target)
        rce = -torch.sum(pred_softmax * torch.log(one_hot + 1e-7), dim=1).mean()
        return self.alpha * ce + self.beta * rce

# -----------------------------
# 9. Confidence Penalty Loss
# -----------------------------
class ConfidencePenaltyLoss(nn.Module):
    def __init__(self, base_loss=nn.CrossEntropyLoss(), penalty_weight=0.1):
        """
        base_loss: 기본 손실함수 (보통 nn.CrossEntropyLoss)
        penalty_weight: entropy penalty 계수 (lambda)
        """
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else nn.CrossEntropyLoss()
        self.penalty_weight = penalty_weight

    def forward(self, logits, target):
        # 기본 손실 계산
        base_loss_val = self.base_loss(logits, target)

        # softmax 확률 계산
        probs = F.softmax(logits, dim=1)
        
        # entropy 계산: -sum(p * log p)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        # 페널티 항 더하기 (기본 loss - lambda * entropy)
        loss = base_loss_val - self.penalty_weight * entropy
        return loss

# -----------------------------
# 10. Max-Entropy Loss
# -----------------------------
class MaxEntropyLoss(nn.Module):
    def __init__(self, lambda_entropy=0.5):
        super(MaxEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_entropy = lambda_entropy

    def forward(self, logits, targets):
        # 기본 CE Loss
        ce_loss = self.ce(logits, targets)

        # 확률로 변환 (softmax)
        probs = F.softmax(logits, dim=1)

        # Entropy 계산: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

        # Total Loss
        loss = ce_loss + self.lambda_entropy * entropy
        return loss
    
# -----------------------------
# Loss Builder
# -----------------------------
def get_loss_fn(name,reduction='mean'):
    name = name.lower()
    if name == 'ce':
        return cross_entropy(reduction=reduction)
    elif name == 'labelsmoothing':
        return LabelSmoothingCrossEntropy()
    elif name == 'focal':
        return FocalLoss(gamma=2.0)
    elif name == 'asl':
        return AsymmetricLoss()
    elif name == 'poly1':
        return Poly1CrossEntropyLoss()
    elif name == 'bitempered':
        return BiTemperedLoss()
    elif name == 'taylor':
        return TaylorCrossEntropyLoss()
    elif name == 'sce':
        return SymmetricCrossEntropyLoss()
    elif name == 'confidence':
        return ConfidencePenaltyLoss()
    elif name == 'me':
        return MaxEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")
