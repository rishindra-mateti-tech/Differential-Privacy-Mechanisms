import os
import sys
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
DATASET_FILE = "ENB2012_data.xlsx"

def ensure_dataset(url: str = DATASET_URL, filename: str = DATASET_FILE) -> str:
    """Download the benchmark dataset if not already cached locally."""
    if not os.path.exists(filename):
        print(f"Fetching dataset from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(filename, "wb") as f:
            f.write(resp.read())
        print(f"Dataset saved to {filename}")
    return filename

def laplace_mechanism(true_value: float, sensitivity: float, epsilon: float) -> float:
    """Inject zero-mean Laplace noise scaled to sensitivity / epsilon."""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return float(true_value + noise)

def main():
    dataset_path = ensure_dataset()
    df = pd.read_excel(dataset_path)

    # Column X2 represents Surface Area in the UCI Energy Efficiency dataset
    surface_area = df["X2"].values
    n = len(surface_area)
    true_avg = float(np.mean(surface_area))
    sensitivity = float((np.max(surface_area) - np.min(surface_area)) / n)

    print("--- Differential Privacy Laplace Mechanism on UCI Dataset ---")
    print(f"Dataset Records (n): {n}")
    print(f"True Average Surface Area: {true_avg:.4f}")
    print(f"Global Sensitivity (Delta f): {sensitivity:.6f}\n")

    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    mse_list = []
    num_trials = 1000

    np.random.seed(42)
    print(f"Evaluating {num_trials} Monte Carlo trials per privacy budget:")
    for eps in epsilons:
        noisy_samples = np.array([laplace_mechanism(true_avg, sensitivity, eps) for _ in range(num_trials)])
        mse = float(np.mean((noisy_samples - true_avg) ** 2))
        mse_list.append(mse)
        print(f"  Epsilon = {eps:4.1f} | Empirical MSE = {mse:10.6f} | Accuracy (1/MSE) = {1.0/mse:10.4f}")

    # Plot Accuracy (1/MSE) vs Epsilon
    plt.figure(figsize=(8, 5))
    accuracies = [1.0 / m for m in mse_list]
    plt.plot(epsilons, accuracies, marker="o", color="green", linewidth=2, label="Utility / Accuracy (1/MSE)")
    plt.title("Empirical Utility vs Privacy Budget (ε)")
    plt.xlabel("ε (Privacy Budget)")
    plt.ylabel("Accuracy (1 / MSE)")
    plt.xticks(epsilons)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    output_img = "accuracy_vs_epsilon.png"
    plt.savefig(output_img, dpi=300)
    print(f"\nAccuracy plot saved successfully to {output_img}")
    if "--show" in sys.argv:
        plt.show()

if __name__ == "__main__":
    main()
