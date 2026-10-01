# Differential Privacy Mechanisms: Sensitivity Calibration, Laplace Noise, and Bayesian Suspicion Modeling

Author: Rishindra Mateti  
Department of Computer Science, Wright State University  
Email: mateti.7@wright.edu | research@rishindramateti.com  

---

## Overview

This repository contains research manuscripts, empirical implementations, and mathematical models evaluating Differential Privacy (DP) mechanisms, global query sensitivity calibration, and adversarial posterior suspicion inference across structured numerical datasets.

All research papers and presentations in this repository are available as compiled, publication-ready PDF documents alongside reproducible Python scripts.

---

## Mathematical Foundations

### 1. Pure $\epsilon$-Differential Privacy
A randomized mechanism $\mathcal{M}$ provides $\epsilon$-differential privacy if for all neighboring datasets $D, D'$ differing on at most one individual record, and all query output subsets $S \subseteq \text{Range}(\mathcal{M})$:

$$P[\mathcal{M}(D) \in S] \le e^{\epsilon} \cdot P[\mathcal{M}(D') \in S]$$

where $\epsilon > 0$ denotes the privacy budget parameter. Smaller values of $\epsilon$ guarantee stronger privacy protection at the expense of output utility.

### 2. Global Sensitivity Calibration & Laplace Mechanism
For a real-valued aggregation query $f: \mathcal{D} \rightarrow \mathbb{R}$, global $L_1$ sensitivity $\Delta f$ represents the maximum possible change resulting from adding or removing any single individual record:

$$\Delta f = \max_{D, D'} \|f(D) - f(D')\|_1$$

For computing the sample mean across $n$ records with domain bounds $[a, b]$, global sensitivity is calibrated as:

$$\Delta f = \frac{b - a}{n}$$

The Laplace mechanism introduces calibrated zero-mean noise drawn from a Laplace distribution scaled inversely to $\epsilon$:

$$\mathcal{M}_L(D) = f(D) + Y, \quad Y \sim \text{Laplace}\left(0, \frac{\Delta f}{\epsilon}\right)$$

### 3. Bayesian Posterior Suspicion Model
Given an adversary with prior suspicion $\pi = P(D = D_{\text{in}})$, the updated posterior probability that an individual record belongs to dataset $D$ given observed noisy output release $O = \mathcal{A}(D)$ is derived via Bayes' theorem:

$$P(D = D_{\text{in}} \mid \mathcal{A}(D) = O) = \frac{e^{\epsilon} \cdot \pi}{e^{\epsilon} \cdot \pi + (1 - \pi)}$$

As the privacy budget $\epsilon \rightarrow 0$, the posterior probability converges toward the prior suspicion $\pi$, providing mathematical information-theoretic deniability.

---

## Repository Structure

```text
Differential-Privacy-Mechanisms/
|
|-- Differential_Privacy_Survey_Rishindra.pdf       # Comprehensive IEEE-format survey paper
|-- part1_updated_suspicion.py                      # Bayesian posterior suspicion model implementation
|-- part2_laplace_mechanism.py                      # Laplace mechanism & empirical accuracy-privacy trade-off
|-- README.md                                       # Repository documentation and formulations
`-- .gitignore                                      # Build artifact exclusions
```

---

## Research Paper Summary

**`Differential_Privacy_Survey_Rishindra.pdf`**  
- **Title:** Advancements of Differential Privacy in Modern Applications: A Comprehensive Survey  
- **Author:** Rishindra Mateti  
- **Scope:** Evaluates theoretical privacy bounds and empirical utility tradeoffs across edge computing, differential private stochastic gradient descent (DP-SGD), localized privacy models, and distributed cloud analytics.

---

## Code Implementations & Execution

### Prerequisites

```bash
pip install numpy pandas matplotlib openpyxl
```

### 1. Bayesian Suspicion Update Simulation (`part1_updated_suspicion.py`)
Computes and plots an adversary's updated posterior suspicion across privacy budgets $\epsilon \in [1, 7]$ for a balanced prior baseline ($\pi = 0.5$):

```bash
python part1_updated_suspicion.py
```
Output: Generates `suspicion_probability_plot.png`.

### 2. Empirical Laplace Mechanism on UCI Dataset (`part2_laplace_mechanism.py`)
Downloads the real-world UCI Energy Efficiency benchmark dataset (`ENB2012`), computes exact global sensitivity on surface area features, injects calibrated Laplace noise across privacy budgets $\epsilon \in [0.1, 0.5, 1.0, 2.0, 5.0]$, computes Mean Squared Error (MSE) over 1,000 iterations per budget, and plots the empirical accuracy curve ($\frac{1}{\text{MSE}}$ vs $\epsilon$):

```bash
python part2_laplace_mechanism.py
```

#### Empirical Results on UCI Energy Efficiency Dataset:
- **Baseline Feature Mean:** 671.71
- **Global Sensitivity ($\Delta f$):** 0.3828
- **$\epsilon = 0.1$:** $\text{MSE} \approx 27.74$ (High Noise, Maximum Privacy)
- **$\epsilon = 0.5$:** $\text{MSE} \approx 1.20$
- **$\epsilon = 1.0$:** $\text{MSE} \approx 0.315$
- **$\epsilon = 2.0$:** $\text{MSE} \approx 0.069$
- **$\epsilon = 5.0$:** $\text{MSE} \approx 0.012$ (Negligible Noise, High Utility)

---

## Citation

```bibtex
@misc{mateti2025differentialprivacy,
  author = {Mateti, Rishindra},
  title = {Advancements of Differential Privacy in Modern Applications: A Comprehensive Survey},
  year = {2025},
  institution = {Wright State University},
  url = {https://github.com/rishindra-mateti-tech/Differential-Privacy-Mechanisms}
}
```

---

## License

This project is licensed under the MIT License.
