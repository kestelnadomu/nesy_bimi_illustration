import torch
from torch import nn

import fuzzy_ops as ops

# --- Operators ---
And = ops.And("product")
Or = ops.Or("product")
Not = ops.Not("std")
Implies = ops.Implies("reichenbach")
Forall = ops.Forall("pmean", p=1)
Exists = ops.Exists("product", p=6)
Equiv = ops.Equiv(and_op=And, implies_op=Implies)

Independent = ops.Independent("eta_squared")
Equal = ops.Equal("gaussian")


# --- Losses ---
class BaselineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, y_true, y_pred, a_batch=None):
        return self.bce(y_pred, y_true)


class SeparationVarianceLoss(nn.Module):
    """BCE + variance of group-wise predictions (simple proxy for Var TPR/FPR)."""
    def __init__(self, reg_weight=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.reg_weight = reg_weight

    def forward(self, y_true, y_pred, a_batch):
        base = self.bce(y_pred, y_true)
        # compute group means of predictions
        groups = torch.unique(a_batch)
        means = []
        for g in groups:
            mask = (a_batch == g)
            if mask.sum() == 0:
                continue
            means.append(y_pred[mask].mean())
        if len(means) <= 1:
            reg = torch.tensor(0.0, device=y_pred.device)
        else:
            means = torch.stack(means)
            reg = torch.var(means)
        return base + self.reg_weight * reg


class EqualOpportunityVarianceLoss(nn.Module):
    """BCE + variance across groups of mean prediction on positive true labels (proxy for TPR variance)."""
    def __init__(self, reg_weight=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.reg_weight = reg_weight

    def forward(self, y_true, y_pred, a_batch):
        base = self.bce(y_pred, y_true)
        mask = (y_true == 1)
        if mask.sum() == 0:
            return base
        groups = torch.unique(a_batch[mask])
        means = []
        for g in groups:
            gm = (a_batch[mask] == g)
            if gm.sum() == 0:
                continue
            means.append(y_pred[mask][gm].mean())
        if len(means) <= 1:
            reg = torch.tensor(0.0, device=y_pred.device)
        else:
            means = torch.stack(means)
            reg = torch.var(means)
        return base + self.reg_weight * reg


class SeparationSymbolicLoss(nn.Module):
    def __init__(self, reg_weight=1.0):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self, y_true, y_pred, a_batch):
        accuracy = Forall(Equal(y_true, y_pred))

        outcomes = torch.unique(y_true)
        separation = Forall( # quantify over number of outcomes
            torch.tensor([Forall( # quantify over batch size
                Implies(
                    Equal(y_true, o),
                    Independent(y_pred, a_batch)
                )
            ) for o in outcomes]) # Quickfix for quantification
        )

        return 1 - And(accuracy, separation)
    

class EqualOpportunitySymbolicLoss(nn.Module):
    def __init__(self, reg_weight=1.0):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self, y_true, y_pred, a_batch):
        accuracy = Forall(Equal(y_true, y_pred))

        eo = Forall(# quantify over batch size
            Implies(
                Equal(y_true, 1),
                Independent(y_pred, a_batch)
            )
        )

        return 1 - And(accuracy, eo)
    

class SufficiencySymbolicLoss(nn.Module):
    def __init__(self, reg_weight=1.0):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self, y_true, y_pred, a_batch):
        accuracy = Forall(Equal(y_true, y_pred))

        outcomes = torch.unique(y_true)
        sufficiency = Forall( # quantify over number of outcomes
            torch.tensor([Forall( # quantify over batch size
                Implies(
                    Equal(y_pred, o),
                    Independent(y_true, a_batch)
                )
            ) for o in outcomes]) # Quickfix for quantification
        )

        return 1 - And(accuracy, sufficiency)

loss_experiment = {
    "baseline": BaselineLoss(),
    "separation_variance": SeparationVarianceLoss(),
    "separation_symbolic": SeparationSymbolicLoss(),
    "equal_opportunity_variance": EqualOpportunityVarianceLoss(),
    "equal_opportunity_symbolic": EqualOpportunitySymbolicLoss(),
}