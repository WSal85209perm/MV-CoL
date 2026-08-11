"""Per-view contrastive objectives used by MV-CoL."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer projection head used only by the stage-one metric losses."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features.float())


def supervised_contrastive_loss(
    projected_features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Paper-style SupCon for one view only.

    The projection output is L2-normalized. For each anchor, other samples with
    the same class label are positives; every non-anchor sample is in the
    contrastive denominator. The anchor itself is explicitly excluded.
    """
    if projected_features.ndim != 2:
        raise ValueError("projected_features must have shape [batch, dimension]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    features = F.normalize(projected_features.float(), p=2, dim=1)
    labels = labels.reshape(-1)
    batch_size = features.shape[0]
    if labels.shape[0] != batch_size:
        raise ValueError("labels and projected_features must have the same batch size")
    if batch_size < 2:
        return features.sum() * 0.0

    logits = features @ features.T / temperature
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    contrast_mask = ~self_mask
    positive_mask = labels[:, None].eq(labels[None, :]) & contrast_mask

    # Mask before logsumexp so the anchor can never contribute to its own
    # denominator. logsumexp keeps the paper's ratio numerically stable.
    log_denominator = torch.logsumexp(logits.masked_fill(self_mask, -torch.inf), dim=1, keepdim=True)
    log_probability = logits - log_denominator
    positive_count = positive_mask.sum(dim=1)
    valid_anchors = positive_count > 0
    if not torch.any(valid_anchors):
        return features.sum() * 0.0

    mean_positive_log_probability = (
        (log_probability * positive_mask).sum(dim=1) / positive_count.clamp_min(1)
    )
    return -mean_positive_log_probability[valid_anchors].mean()


def batch_hard_triplet_loss(
    projected_features: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Triplet loss for one view using batch-hard mining.

    For every valid anchor, the farthest same-label sample is the positive and
    the closest different-label sample is the negative. Euclidean distance is
    measured in the L2-normalized projected space, and the optimized term is
    max(0, D(a,p) - D(a,n) + margin). No samples from another view enter this
    mining operation.
    """
    if projected_features.ndim != 2:
        raise ValueError("projected_features must have shape [batch, dimension]")
    if margin < 0:
        raise ValueError("margin cannot be negative")

    features = F.normalize(projected_features.float(), p=2, dim=1)
    labels = labels.reshape(-1)
    batch_size = features.shape[0]
    if labels.shape[0] != batch_size:
        raise ValueError("labels and projected_features must have the same batch size")
    if batch_size < 2:
        return features.sum() * 0.0

    distances = torch.cdist(features, features, p=2)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    negative_mask = ~labels[:, None].eq(labels[None, :])
    valid_anchors = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not torch.any(valid_anchors):
        return features.sum() * 0.0

    hardest_positive = distances.masked_fill(~positive_mask, -torch.inf).max(dim=1).values
    hardest_negative = distances.masked_fill(~negative_mask, torch.inf).min(dim=1).values
    losses = F.relu(hardest_positive - hardest_negative + margin)
    return losses[valid_anchors].mean()


class PerViewJointLoss(nn.Module):
    """Implement ``L = L_ce + lambda_1 L_sup + lambda_2 L_tri``.

    Each component is computed independently for Views A, B, and C before its
    three values are averaged. In particular, this class never concatenates
    samples from different prompt views into one metric-learning batch.
    """

    def __init__(
        self,
        supcon_weight: float,
        triplet_weight: float,
        temperature: float,
        margin: float,
        triplet_mining: str,
    ) -> None:
        super().__init__()
        if triplet_mining != "batch_hard":
            raise ValueError("the released implementation supports triplet_mining='batch_hard'")
        self.supcon_weight = supcon_weight
        self.triplet_weight = triplet_weight
        self.temperature = temperature
        self.margin = margin
        # The paper states the triplet objective but not its sampling rule. The
        # released reference implementation explicitly uses batch-hard mining.
        self.triplet_mining = triplet_mining

    def forward(
        self,
        logits_by_view: Sequence[torch.Tensor],
        projections_by_view: Sequence[torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(logits_by_view) != 3 or len(projections_by_view) != 3:
            raise ValueError("MV-CoL requires exactly three independently processed views")

        ce_losses = [F.cross_entropy(logits.float(), labels) for logits in logits_by_view]
        supcon_losses = [
            supervised_contrastive_loss(features, labels, self.temperature)
            for features in projections_by_view
        ]
        triplet_losses = [
            batch_hard_triplet_loss(features, labels, self.margin)
            for features in projections_by_view
        ]
        ce = torch.stack(ce_losses).mean()
        supcon = torch.stack(supcon_losses).mean()
        triplet = torch.stack(triplet_losses).mean()
        # The paper assigns an implicit coefficient of one to L_ce. Only
        # lambda_1 and lambda_2 are configurable.
        total = ce + self.supcon_weight * supcon + self.triplet_weight * triplet
        return total, {"ce": ce.detach(), "supcon": supcon.detach(), "triplet": triplet.detach()}
