import numpy as np
import matplotlib.pyplot as plt

T = 1.0
n_steps = 1000
n_paths = 1000
dt = T / n_steps



rng = np.random.default_rng(42)
Z = rng.normal(size=(n_paths, n_steps))
dW = np.sqrt(dt) * Z
W = np.cumsum(dW, axis=1)
W = np.insert(W, 0, 0, axis=1)


print("Mean of W(1):", W[:, -1].mean())
print("Variance of W(1):", W[:, -1].var())

quadratic_variation = np.sum(dW**2, axis=1)

print("Mean quadratic variation:", quadratic_variation.mean())
print("Variance of quadratic variation:", quadratic_variation.var())

time = np.linspace(0, T, n_steps + 1)

plt.figure(figsize=(10, 6))
plt.plot(time, W[:20].T, linewidth=1)

plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Time t")
plt.ylabel("W(t)")
plt.title("20 Simulated Brownian Motion Paths")
plt.grid(alpha=0.3)
plt.show()

empirical_std = W.std(axis=0)
theoretical_std = np.sqrt(time)

plt.figure(figsize=(8, 5))
plt.plot(time, empirical_std, label="Empirical SD")
plt.plot(time, theoretical_std, "--", label="Theoretical sqrt(t)")

plt.xlabel("Time t")
plt.ylabel("Standard deviation")
plt.title("Brownian Motion: Empirical vs Theoretical Spread")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

S_initial = 100.0
mu = 0.05
sigma = 0.20

S = np.empty((n_paths, n_steps + 1))
S[:, 0] = S_initial
for i in range(n_steps):
    S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[:, i])

print("Minimum simulated price:", S.min())
print("Mean terminal price:", S[:, -1].mean())
print("Theoretical mean:", S_initial * np.exp(mu * T))

plt.figure(figsize=(10, 6))
plt.plot(time, S[:20].T, linewidth=1)

plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Time t")
plt.ylabel("S(t)")
plt.title("20 Simulated Stock Price Paths")
plt.grid(alpha=0.3)
plt.show()

