# Neurosymbolic Architectures for Algorithmic Fairness

## Abstract
Bias is a pervasive issue in Machine Learning, particularly in domains like automated decision-making (ADM), where it can lead to unfair treatment of individuals or groups based on sensitive attributes. Accounting for it requires knowledge and reasoning about how bias can affect the decision process and how to constrain this process in order to decrease its vulnerability to societal and statistical bias. In the field of bias mitigation, a broad set of constraining techniques has been developed to address the issue of biased predictions. Usually, such a technique is an architecture or procedure particularly designed for a use case or a distinct definition of fairness. In application however, practitioners face complex realities requiring flexible, complex reasoning about constraints, yet the link to integrative approaches that combine symbolic reasoning with logical constraints and statistical learning is still missing. Although there exist several neurosymbolic architectures able to incorporate knowledge and constraints into a model, only few attempts have been made to use them to apply fairness constraints to model predictions. This work tries to bridge this gap by mapping neurosymbolic architectures to bias mitigation techniques. We categorize these architectures based on their potential application in pre-processing, in-processing, and post-processing. By doing so, we aim to provide a structured overview of the current set of existing neurosymbolic architectures for bias mitigation, and highlight important underexplored directions and promising research avenues at the intersection of neurosymbolic AI and algorithmic fairness.

## Keywords
Neurosymbolic AI, Algorithmic Fairness, Bias Mitigation, Trustworthy AI

## Illustration

### Data
TODO (Fairground data (folk tables, public coverage))
@article{ding2021retiring,
  title={Retiring adult: New datasets for fair machine learning},
  author={Ding, Frances and Hardt, Moritz and Miller, John and Schmidt, Ludwig},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={6478--6490},
  year={2021}
}

### Quantitative and Qualitative Criteria

#### Quantitative
* Accuracy
* Equality of Opportunity
* TODO

#### Qualitative
* Flexibility
* TODO



### Experimental Conditions
* Baseline: Hardt et al. (2016): Post-Processing Threshold Adjustment
* Same network, different loss: TODO
  * Separation: $Var_a(TPR_a) + Var_a(FPR_a)$
  * Equal Opportunity: $Var_a(TPR_a) = \frac{1}{|A|} \sum_{a \in A} (TPR_a - \overline{TPR})^2 = \frac{1}{|A|} \sum_{a \in A} (\frac{1}{|Y^+_a|} \sum_{y^+_a \in Y^+_a} \textbf{P}(x) - \frac{1}{|Y^+|} \sum_{y^+ \in Y^+} \textbf{P}(x))^2$
  * Sufficiency: Switch is non-trivial, since discrete values are inherent to function
* Neurosymbolic Loss: TODO
  * Independence Function: $a \bot b \equiv 1-\eta^2_{a, b} = 1 - \frac{\sum_{a \in A} n_a (\mu_a - \bar{\mu})^2}{\sum_{i \in N} (x_i - \bar{\mu})^2}$ (multiclass-metric correlation)
  * Equality Function: $a = b \equiv \exp(-d(a,b)) = \exp(-(a-b)^2)$
  * Separation
    * Abstract symbolic: $\forall x \in X, o \in O: \text{y}(x) = o \implies \textbf{P}(x) \bot \text{a}(x)$, e.g. with $O = \{0, 1\}$
    * Concrete calculation:
      * Gödel: $\inf_{x \in X} \inf_{o \in O} \min(1, o-y(x) + \textbf{P}(x) \bot \text{a}(x))$
  * Equal Opportunity
    * Abstract symbolic: $\forall x \in X: \text{y}(x) = 1 \implies \textbf{P}(x) \bot \text{a}(x)$
    * Concrete calculation: 
      * Gödel: $1 - \inf_{x \in X} \text{ifelse}[\text{eq}(y(x), 1) \leq \textbf{P}(x) \bot \text{a}(x), 1, \textbf{P}(x) \bot \text{a}(x)]$
      * Lukasiewicz: $1 - \min[\sum_{x \in X} \min[1, 1 - \text{eq}(y(x), 1) + \textbf{P}(x) \bot \text{a}(x)], |X|-1]$
      * Product: $1 - \prod_{x \in X} \min[1, \frac{\textbf{P}(x) \bot \text{a}(x)}{\text{eq}(y(x), 1)}]$
      * Reichenbach: $1 - \prod_{x \in X} \min[1, 1 - \text{eq}(y(x), 1) + \text{eq}(y(x), 1) \times \textbf{P}(x) \bot \text{a}(x)]$
  * Sufficiency:
    * Switch is trivial, just switch $y$ and $\hat{y}$ in formulae
    * LTN maps discrete values to metric ones inherently