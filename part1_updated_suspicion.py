import sys
import numpy as np
import matplotlib.pyplot as plt

def updated_suspicion(epsilon: float, prior: float = 0.5) -> float:
    """Compute posterior suspicion probability under epsilon-DP using Bayes' theorem."""
    numerator = np.exp(epsilon) * prior
    denominator = numerator + (1.0 - prior)
    return float(numerator / denominator)

def main():
    prior = 0.5
    epsilons = [1, 2, 3, 4, 5, 6, 7]
    posterior_probs = [updated_suspicion(eps, prior) for eps in epsilons]

    print("--- Bayesian Suspicion Update Results ---")
    print(f"Prior Suspicion: {prior:.2f}")
    for eps, prob in zip(epsilons, posterior_probs):
        print(f"Epsilon = {eps}: Updated Suspicion = {prob:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, posterior_probs, marker="o", color="blue", linewidth=2, label="P(D=Din | A(D)=O)")
    plt.title("Adversarial Suspicion Probability vs Privacy Budget (ε)")
    plt.xlabel("ε (Privacy Budget)")
    plt.ylabel("Updated Suspicion Probability")
    plt.xticks(epsilons)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    output_img = "suspicion_probability_plot.png"
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved successfully to {output_img}")
    if "--show" in sys.argv:
        plt.show()

if __name__ == "__main__":
    main()
