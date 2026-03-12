from typing import Literal
import torch
import torch.nn.functional as F


##################
# Operators
##################

class Operator:
    def __init__(self, type):
        self.type = type

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
    

class Not(Operator):
    def __init__(self, type: Literal["godel", "std"] = "std"):
        super().__init__(type)

    def __call__(self,x):
        if self.type == "godel":
            return x == 0
        elif self.type == "std":
            return 1 - x
        else:
            raise ValueError(f"Unknown Not type: {self.type}")

class And(Operator):
    def __init__(self, type: Literal["godel", "lukasiewicz", "product"] = "product"):
        super().__init__(type)

    def __call__(self, a, b):
        if self.type == "godel":
            return torch.min(a, b)
        elif self.type == "lukasiewicz":
            return torch.clamp(a + b - 1, min=0.0)
        elif self.type == "product":
            return a * b
        else:
            raise ValueError(f"Unknown And type: {self.type}")

class Or(Operator):
    def __init__(self, type: Literal["godel", "lukasiewicz", "product"] = "godel"):
        super().__init__(type)

    def __call__(self, a, b):
        if self.type == "godel":
            return torch.max(a, b)
        elif self.type == "lukasiewicz":
            return torch.clamp(a + b, max=1.0)
        elif self.type == "product":
            return a + b - a * b
        else:
            raise ValueError(f"Unknown Or type: {self.type}")

class Implies(Operator):
    def __init__(self, type: Literal["godel", "lukasiewicz", "product", "reichenbach"] = "godel"):
        super().__init__(type)

    def __call__(self, a, b):
        if self.type == "godel":
            return torch.where(a <= b, torch.ones_like(a), b)
        elif self.type == "lukasiewicz":
            return torch.clamp(1 - a + b, min=0.0, max=1.0)
        elif self.type == "product":
            return torch.where(a <= b, torch.ones_like(a), b / a)
        elif self.type == "reichenbach":
            return 1 - a + a * b
        else:
            raise ValueError(f"Unknown Implies type: {self.type}")

class Equiv(Operator):
    """Returns an operator that computes: And(Implies(x,y),Implies(y,x))"""
    def __init__(self, and_op, implies_op):
        super().__init__(type = "std")
        self.and_op = and_op
        self.implies_op = implies_op
    
    def __call__(self, x, y):
        return self.and_op(self.implies_op(x,y), self.implies_op(y,x))
    

class Exists(Operator):
    def __init__(self, type: Literal["godel", "lukasiewicz", "product", "pmean"] = "godel", p: int = 6):
        super().__init__(type)
        self.p = p

    def __call__(self, x):
        if self.type == "godel":
            return torch.max(x)
        elif self.type == "lukasiewicz":
            return torch.clamp(torch.sum(x), max=1.0)
        elif self.type == "product":
            return 1 - torch.prod(1 - x)
        elif self.type == "pmean":
            return torch.pow(torch.mean(torch.pow(x, self.p)), 1/self.p)
        else:
            raise ValueError(f"Unknown Exists type: {self.type}")

class Forall(Operator):
    def __init__(self, type: Literal["godel", "lukasiewicz", "product", "pmean"] = "godel", p: int = 1):
        super().__init__(type)
        self.p = p # only relevant for "pmean"

    def __call__(self, x):
        if self.type == "godel":
            return torch.min(x)
        elif self.type == "lukasiewicz":
            return torch.clamp(torch.sum(x) - (len(x) - 1), min=0.0)
        elif self.type == "product":
            return torch.prod(x)
        elif self.type == "pmean":
            return 1 - torch.pow(torch.mean(torch.pow(1 - x, self.p)), 1/self.p)
        else:
            raise ValueError(f"Unknown Forall type: {self.type}")
    


##################
# Predicates
##################

class Predicate:
    def __init__(self, type):
        self.type = type

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class Independent(Predicate):
    def __init__(self, type: Literal["eta_squared", "pearson"]):
        super().__init__(type)

    def __call__(self, num, cat, eps = 1e-5):
        if self.type == "eta_squared":
            # Placeholder: 1 - variance of group means / total variance
            groups = torch.unique(cat)
            grand_mean = num.mean()
            ss_total = ((num - grand_mean) ** 2).sum()
            ss_within = 0.0
            for g in groups:
                mask = (cat == g)
                group_vals = num[mask]
                if group_vals.numel() == 0:
                    continue
                group_mean = group_vals.mean()
                ss_within += ((group_vals - group_mean) ** 2).sum()
            if ss_total == 0:
                return torch.tensor(1.0, device=num.device)
            return 1 - (ss_within / (ss_total + eps))
        elif self.type == "pearson":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown Independent type: {self.type}")

class Equal(Predicate):
    def __init__(self, type: Literal["gaussian"] = "gaussian"):
        super().__init__(type)

    def __call__(self, a, b):
        if self.type == "gaussian":
            # Gaussian similarity (can be replaced with exp(-d^2) or similar)
            return torch.exp(-torch.square(a - b))
        else:
            raise ValueError(f"Unknown Equal type: {self.type}")