import numpy as np
import matplotlib.pyplot as plt

# Initial suspicion probability: attacker believes with 50% confidence
prior = 0.5

# Values of epsilon (ε) to test
epsilons = [1, 2, 3, 4, 5, 6, 7]

# Let's assume:
# P[A(D_out) = O] = 1  (we set this to 1 as a baseline)
# P[A(D_in) = O] = exp(ε)  (from differential privacy bound)

# Compute posterior suspicion using Bayes' theorem
def updated_suspicion(epsilon, prior):
    numerator = np.exp(epsilon) * prior
    denominator = numerator + (1 - prior)
    return numerator / denominator

# Compute results
posterior_probs = [updated_suspicion(eps, prior) for eps in epsilons]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(epsilons, posterior_probs, marker='o', color='blue', label='Updated Suspicion P(D=Din | A(D)=O)')
plt.title('Updated Suspicion Probability vs ε')
plt.xlabel('ε (Privacy Budget)')
plt.ylabel('Updated Suspicion Probability')
plt.xticks(epsilons)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("suspicion_probability_plot.png")
plt.show()
