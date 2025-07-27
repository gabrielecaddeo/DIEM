import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.linalg import norm
from scipy.stats import kstest, chi2, ztest
import os
from scipy.spatial.distance import cdist
from scipy.linalg import null_space

def DIEM_Stat(N, maxV, minV, fig_flag):
    d = []
    dort = []

    for _ in range(int(1e5)):
        a = (maxV - minV) * np.random.rand(N, 1) + minV
        b = (maxV - minV) * np.random.rand(N, 1) + minV
        tmp = null_space(a.T)
        ort = tmp[:, 0].reshape(-1, 1)
        d.append(cdist(a.T, b.T, metric='euclidean')[0][0])
        dort.append(cdist(a.T, ort.T, metric='euclidean')[0][0])

    d = np.array(d)
    dort = np.array(dort)
    exp_center = np.median(d)
    vard = np.var(d)
    orth_med = (maxV - minV) * (np.median(dort) - exp_center) / vard
    adjusted_dist = (maxV - minV) * (d - exp_center) / vard
    std_one = np.std(adjusted_dist)
    min_DIEM = -(maxV - minV) * (exp_center / vard)
    max_DIEM = (maxV - minV) * (np.sqrt(N) * (maxV - minV) - exp_center) / vard

    if fig_flag == 1:
        width = 10
        x = np.arange(1, width + 1)
        plt.figure(figsize=(6, 6))
        plt.fill_between(x, -std_one, std_one, color='r', alpha=0.2)
        plt.fill_between(x, -2 * std_one, 2 * std_one, color='r', alpha=0.2)
        plt.fill_between(x, -3 * std_one, 3 * std_one, color='r', alpha=0.2)
        plt.plot(x, np.zeros(width), 'k--', linewidth=1)
        plt.plot(x, np.full(width, orth_med), 'k-.', linewidth=1)
        plt.plot(x, np.full(width, min_DIEM), 'k-.', linewidth=1)
        plt.plot(x, np.full(width, max_DIEM), 'k-.', linewidth=1)
        plt.ylabel('DIEM')
        plt.xticks([])
        plt.box(False)
        plt.show()

    return exp_center, vard, std_one, orth_med, min_DIEM, max_DIEM

# --- Utility Functions ---
def cos_sim(a, b):
    return (a.T @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

def randu_sphere(n, N, max_val=1.0, min_val=-1.0):
    x = np.random.randn(n, N)
    x /= np.linalg.norm(x, axis=0)
    radius = np.random.uniform(min_val, max_val, size=(1, N))
    return x * radius

# --- Class for Vector Distance Simulations ---
class DistanceAnalysis:
    def __init__(self, N_range, vmax=1, vmin=0, dist_type=1):
        self.N = N_range
        self.vmax = vmax
        self.vmin = vmin
        self.dist_type = dist_type
        self.results = {}

    def simulate(self):
        for i, n_dim in enumerate(self.N):
            print(f"Simulating for N = {n_dim}...")
            self.results[n_dim] = {
                'cos_p': [], 'cos_n': [], 'cos_t': [],
                'd_p': [], 'd_n': [], 'd_t': [],
                'dnorm_p': [], 'dnorm_n': [], 'dnorm_t': [],
                'man_p': [], 'man_n': [], 'man_t': []
            }
            for _ in range(10000):
                a_p, a_n, a_t = self._generate_vectors(n_dim)
                b_p, b_n, b_t = self._generate_vectors(n_dim)

                self._append_metrics(n_dim, a_p, b_p, 'p')
                self._append_metrics(n_dim, a_n, b_n, 'n')
                self._append_metrics(n_dim, a_t, b_t, 't')

    def _generate_vectors(self, dim):
        if self.dist_type == 1:  # Uniform
            return (
                self.vmax * np.random.rand(dim, 1),
                -self.vmax * np.random.rand(dim, 1),
                (2 * self.vmax) * np.random.rand(dim, 1) - self.vmax
            )
        elif self.dist_type == 2:  # Gaussian
            return (
                0.3 * np.random.randn(dim, 1) + self.vmax / 2,
                0.3 * np.random.randn(dim, 1) - self.vmax / 2,
                0.6 * np.random.randn(dim, 1)
            )
        elif self.dist_type == 3:  # Uniform on Sphere
            return (
                randu_sphere(dim, 1, self.vmax, self.vmin),
                randu_sphere(dim, 1, self.vmin, -self.vmax),
                randu_sphere(dim, 1, self.vmax, -self.vmax)
            )
        else:
            raise ValueError("Invalid distribution type")

    def _append_metrics(self, dim, a, b, key):
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        self.results[dim][f'cos_{key}'].append(cos_sim(a, b))
        self.results[dim][f'd_{key}'].append(cdist(a.T, b.T)[0][0])
        self.results[dim][f'dnorm_{key}'].append(cdist((a / np.linalg.norm(a)).T, (b / np.linalg.norm(b)).T)[0][0])
        self.results[dim][f'man_{key}'].append(cdist(a.T, b.T, metric='cityblock')[0][0])

    def plot_results(self):
        keys = ['cos', 'dnorm', 'd', 'man']
        types = ['p', 'n', 't']
        for metric in keys:
            plt.figure(figsize=(15, 5))
            for i, t in enumerate(types):
                data = [self.results[n][f'{metric}_{t}'] for n in self.N]
                ax = plt.subplot(1, 3, i + 1)
                sns.boxplot(data=data)
                ax.set_title(f"{metric.upper()} - Type {t.upper()}")
                ax.set_xticklabels(self.N)
                ax.set_xlabel("Dimensions")
                ax.set_ylabel(metric)
            plt.tight_layout()
            plt.show()

# --- Class for Text Embedding Similarity ---
class TextEmbeddingSimilarity:
    def __init__(self, embedding_file1, embedding_file2):
        self.sent1 = np.loadtxt(embedding_file1, delimiter=',', skiprows=1)
        self.sent2 = np.loadtxt(embedding_file2, delimiter=',', skiprows=1)

    def compute_cosine_similarity(self):
        return cdist(self.sent1, self.sent2, metric='cosine')

    def compute_diem_similarity(self, maxV, minV, exp_center, vard):
        from diem_functions import getDIEM  # Use previously defined DIEM
        DIEM, _ = getDIEM(self.sent1.T, self.sent2.T, maxV, minV, exp_center, vard)
        return DIEM

    def plot_comparison(self, cosine_sim, diem_sim, min_DIEM, max_DIEM):
        diag_cos = np.diag(cosine_sim)
        diag_diem = np.diag(diem_sim)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 2, 1)
        plt.hist(diag_cos, bins=50, alpha=0.7)
        plt.title("Cosine Similarity (Rated)")

        plt.subplot(2, 2, 2)
        plt.hist(diag_diem, bins=50, alpha=0.7)
        plt.title("DIEM Similarity (Rated)")
        plt.axvline(min_DIEM, color='k', linestyle='--')
        plt.axvline(max_DIEM, color='k', linestyle='--')

        plt.subplot(2, 2, 3)
        plt.hist(cosine_sim.flatten(), bins=50, alpha=0.7)
        plt.title("Cosine Similarity (All)")

        plt.subplot(2, 2, 4)
        plt.hist(diem_sim.flatten(), bins=50, alpha=0.7)
        plt.title("DIEM Similarity (All)")
        plt.axvline(min_DIEM, color='k', linestyle='--')
        plt.axvline(max_DIEM, color='k', linestyle='--')

        plt.tight_layout()
        plt.show()
