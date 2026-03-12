from typing import Literal
import aif360
import ltn
import tensorflow as tf
from tensorflow.keras import losses

# --- LTN Connectives and Quantifiers ---x
Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Equiv = ltn.Wrapper_Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.And_Prod(),ltn.fuzzy_ops.Implies_Reichenbach()))
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=1),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=6),semantics="exists")
formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=1))
# formula_aggregator2 = ltn.fuzzy_ops.Aggreg_pMeanError(p=1)
# formula_aggregator_weighted = CustomWrapperFormulaAggregator(Aggreg_pMeanError_weighted(p=1))
eq = ltn.Predicate.Lambda(
    # lambda args: tf.exp(-0.05*tf.sqrt(tf.reduce_sum(tf.square(args[0]-args[1]),axis=1)))
    lambda args: 1 / (1 + 0.5 * (tf.reduce_sum(tf.square(args[0] - args[1]), axis=1)))
)

def independence(nominal, metric):
    """
    Calculate 1 - eta^2 (effect size) to measure independence between nominal and metric variables.
    Eta^2 = 1 - (SS_within / SS_total)
    """
    groups = tf.unique(nominal)[0]
    grand_mean = tf.reduce_mean(metric)
    ss_total = tf.reduce_sum(tf.square(metric - grand_mean))

    ss_within = 0.0
    for group in groups:
        group_mask = tf.equal(nominal, group)
        group_values = tf.boolean_mask(metric, group_mask)
        group_mean = tf.reduce_mean(group_values)
        ss_within += tf.reduce_sum(tf.square(group_values - group_mean))

    return 1 - (ss_within / ss_total)

ind = ltn.Predicate.Lambda(independence)

# --- Loss definitions for experimental conditions ---

class Loss:
    """Base Loss class for experiment conditions."""
    def __init__(self, name):
        self.name = name
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

# 1. Baseline: Post-Processing Threshold Adjustment (Hardt et al. 2016)
class BaselineLoss(Loss):
    def __init__(self):
        super().__init__("baseline")
    def __call__(self, y_true, y_pred):
        # Standard binary cross-entropy
        return losses.binary_crossentropy(y_true, y_pred)
    
class _ExperimentalLoss(Loss):
    def __init__(self, name, type: Literal["variance", "symbolic"] = "variance", reg_weight = 1.0):
        super().__init__(name)
        
        if type not in ["variance", "symbolic"]:
            raise ValueError("Invalid type for ExperimentalLoss. Must be 'variance' or 'symbolic'.") 
        self.type = type
        self.reg_weight = reg_weight

    def __call__(self):
        # Placeholder for experimental loss calculation
        raise NotImplementedError

# 2. Separation (Variance-based): Var_a(TPR_a) + Var_a(FPR_a)
class SeparationLoss(_ExperimentalLoss):
    def __init__(self, type: Literal["variance", "symbolic"] = "variance", reg_weight = 1.0):
        super().__init__("separation", type, reg_weight)
        
    @tf.function
    def __call__(self, y_true, y_pred, sensitive):
        if self.type == "variance":
            base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # Standard BCE loss
            
            # Calculate variance of predictions across sensitive groups for the positive class
            mean_total = tf.reduce_mean(y_pred)
            groups, _ = tf.unique(sensitive)
            # Mean prediction for each sensitive group among the positive class (y_true == 1)
            mean_sensitive = tf.stack([tf.reduce_mean(y_pred[sensitive == a]) for a in groups])
            diff = tf.square(mean_sensitive - mean_total)

            # BCE loss + regularization term for variance of TPR and FPR across sensitive groups
            return base_loss + self.reg_weight * tf.reduce_mean(diff)
        
        if self.type == "symbolic":
            outcomes, _ = tf.unique(y_true)[0]  # Unique outcomes (e.g., 0 and 1)
            y_true = ltn.Variable("y_true", y_true)
            y_pred = ltn.Variable("y_pred", y_pred)
            sensitive = ltn.Variable("sensitive", sensitive)
            outcomes = ltn.Variable("outcomes", outcomes)

            accuracy = Equiv(y_true, y_pred)  # Equivalence loss

            # ∀x∈X,o∈O: y(x)=o ⟹ P(x)⊥a(x), e.g. with O={0,1}
            separation = Forall([y_pred, outcomes], Implies(eq(y_true, outcomes), ind(y_pred, sensitive)))  # Separation condition for each group
            
            # Combine accuracy with separation constraint
            kb = formula_aggregator([accuracy, separation])  
            return 1.0-kb.tensor

# 3. Equal Opportunity: Var_a(TPR_a)
class EqualOpportunityLoss(_ExperimentalLoss):
    def __init__(self, type: Literal["variance", "symbolic"] = "variance", reg_weight = 1.0):
        super().__init__("equal_opportunity", type, reg_weight)

    @tf.function  
    def __call__(self, y_true, y_pred, sensitive):
        if self.type == "variance":
            base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # Standard BCE loss

            # Focus on positive class (TPR)
            mask = tf.equal(y_true, 1)
            y_pred = tf.boolean_mask(y_pred, mask)
            sensitive = tf.boolean_mask(sensitive, mask)

            # Calculate variance of predictions across sensitive groups for the positive class
            mean_total = tf.reduce_mean(y_pred)
            groups, _ = tf.unique(sensitive)
            # Mean prediction for each sensitive group among the positive class (y_true == 1)
            mean_sensitive = tf.stack([tf.reduce_mean(y_pred[sensitive == a]) for a in groups])
            diff = tf.square(mean_sensitive - mean_total)

            # BCE loss + regularization term for variance of TPR across sensitive groups
            return base_loss + self.reg_weight * tf.reduce_mean(diff)
        
        if self.type == "symbolic":
            # Variables
            y_true = ltn.Variable("y_true", y_true)
            y_pred = ltn.Variable("y_pred", y_pred)
            sensitive = ltn.Variable("sensitive", sensitive)

            accuracy = Equiv(y_true, y_pred)  # Equivalence loss

            # ∀x∈X: y(x)=1 ⟹ P(x)⊥a(x) (# Equal Opportunity condition for each group)
            eo = Forall([y_pred], Implies(eq(y_true, 1), ind(y_pred, sensitive)))  

            # Combine accuracy with equal opportunity constraint
            kb = formula_aggregator([accuracy, eo])  
            return 1.0-kb.tensor

# 4. Sufficiency: Switch y and y_pred in formulae
class SufficiencyLoss(_ExperimentalLoss):
    def __init__(self, type: Literal["variance", "symbolic"] = "variance", reg_weight = 1.0):
        super().__init__("sufficiency", type, reg_weight)
    
    @tf.function
    def __call__(self, y_true, y_pred, sensitive):
        if self.type == "variance":
            raise NotImplementedError("Variance-based Sufficiency loss is not implemented.")
        
        if self.type == "symbolic":
            accuracy = Equiv(y_true, y_pred)  # Equivalence loss
            preds, _ = tf.unique(y_pred)[0]  # Unique outcomes (e.g., 0 and 1)
            y_true = ltn.Variable("y_true", y_true)
            y_pred = ltn.Variable("y_pred", y_pred)
            sensitive = ltn.Variable("sensitive", sensitive)
            preds = ltn.Variable("preds", preds)

            # ∀x∈X,o∈O: P(x)=o ⟹ y(x)⊥a(x) (Sufficiency condition for each group)
            sufficiency = Forall([y_true, preds], Implies(eq(y_pred, preds), ind(y_true, sensitive)))

            # Combine accuracy with sufficiency constraint
            kb = formula_aggregator([accuracy, sufficiency])  
            return 1.0-kb.tensor