# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request

# Download the UCI Energy Efficiency dataset
dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
filename = 'ENB2012_data.xlsx'
urllib.request.urlretrieve(dataset_url, filename)

# Read the dataset
df = pd.read_excel(filename)

# Check available columns (optional, for debug)
print("Available Columns:", df.columns)

# Extract the "Surface Area" column
# "X2" corresponds to Surface Area
surface_area = df['X2'].values
n = len(surface_area)

# True average (without noise)
true_avg = np.mean(surface_area)
print(f"True Average Surface Area: {true_avg}")

# Laplace Mechanism
def laplace_mechanism(true_value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_value + noise

# Sensitivity for average = (max - min) / n
sensitivity = (np.max(surface_area) - np.min(surface_area)) / n
print(f"Sensitivity: {sensitivity}")

# Values of epsilon to test
epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
mse_list = []

# Perform experiments
np.random.seed(42)  # For reproducibility
for epsilon in epsilons:
    noisy_averages = []
    for _ in range(1000):  # Repeat to get stable MSE
        noisy_avg = laplace_mechanism(true_avg, sensitivity, epsilon)
        noisy_averages.append(noisy_avg)
    noisy_averages = np.array(noisy_averages)
    
    mse = np.mean((noisy_averages - true_avg) ** 2)
    mse_list.append(mse)
    print(f"Epsilon: {epsilon}, MSE: {mse}")

# Plotting Accuracy (1/MSE) vs Epsilon
plt.figure(figsize=(8, 5))
plt.plot(epsilons, [1/m for m in mse_list], marker='o', color='green', label='Accuracy (1/MSE)')
plt.title('Accuracy vs Privacy Budget (ε)')
plt.xlabel('ε (Privacy Budget)')
plt.ylabel('Accuracy (1 / MSE)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_vs_epsilon.png")
plt.show()
