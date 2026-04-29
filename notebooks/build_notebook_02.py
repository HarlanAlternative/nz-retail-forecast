"""Build notebooks/02_recommender_svd_vs_als.ipynb programmatically.

Run:
    python notebooks/build_notebook_02.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_PATH = Path(__file__).parent / "02_recommender_svd_vs_als.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text.strip())


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src.strip())


cells = [
    # ── Title ─────────────────────────────────────────────────────────────
    md("""# Recommender System: SVD vs ALS vs Content-Based

**Dataset:** Online Retail II (UCI) — ~500k UK e-commerce transactions
**Scenario:** Implicit-feedback collaborative filtering
**Evaluation:** Leave-one-out Recall@K, NDCG@K, MAP@K

This notebook has two purposes:
1. Compare recommender model families on a real implicit-feedback dataset.
2. Provide the **cross-scenario SVD comparison** — the same matrix factorisation
   mathematics yields *very different* latent factors depending on what matrix
   you factorise (time-series lag features vs user-item interactions).
"""),

    # ── Setup ──────────────────────────────────────────────────────────────
    md("## 1. Setup and Data Loading"),
    code("""
import sys
sys.path.insert(0, '../src')

import logging
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp

from recommender.data_loader import load_data
from recommender.evaluate import evaluate_all
from recommender.popularity_baseline import PopularityRecommender
from recommender.svd_model import SVDRecommender
from recommender.als_model import ALSRecommender
from recommender.content_based import ContentBasedRecommender, HybridRecommender

# Load data (downloads and caches Online Retail II ~50 MB on first run)
train_matrix, full_matrix, test_items, users, items, item_meta = load_data()
n_users, n_items = train_matrix.shape
print(f"Train matrix: {n_users} users × {n_items} items")
print(f"Density: {100 * train_matrix.nnz / (n_users * n_items):.3f}%")
print(f"Test users: {len(test_items)}")
"""),

    # ── Sparsity visualisation ─────────────────────────────────────────────
    md("## 2. Interaction Matrix — Sparsity and Distribution"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Items-per-user distribution
items_per_user = np.diff(train_matrix.indptr)
axes[0].hist(items_per_user, bins=50, color='#2196F3', alpha=0.8, edgecolor='white')
axes[0].set_xlabel('Items per user')
axes[0].set_ylabel('Count')
axes[0].set_title('User Activity Distribution')
axes[0].set_yscale('log')

# Users-per-item distribution
users_per_item = np.diff(train_matrix.tocsc().indptr)
axes[1].hist(users_per_item, bins=50, color='#FF9800', alpha=0.8, edgecolor='white')
axes[1].set_xlabel('Users per item')
axes[1].set_ylabel('Count')
axes[1].set_title('Item Popularity Distribution')
axes[1].set_yscale('log')

fig.suptitle('Online Retail II — Interaction Sparsity', fontsize=13)
fig.tight_layout()
plt.show()

print(f"Median items per user: {np.median(items_per_user):.0f}")
print(f"90th pctile items per user: {np.percentile(items_per_user, 90):.0f}")
print(f"Median users per item: {np.median(users_per_item):.0f}")
"""),

    # ── Train all models ────────────────────────────────────────────────────
    md("## 3. Model Training"),
    code("""
print("Fitting Popularity baseline ...")
pop = PopularityRecommender().fit(train_matrix)

print("Fitting SVD (k=50) ...")
svd = SVDRecommender(n_components=50).fit(train_matrix)

print("Fitting ALS (BM25, 64 factors) ...")
als = ALSRecommender(factors=64, iterations=30).fit(train_matrix)

print("Fitting Content-based (TF-IDF + NMF, 20 topics) ...")
cb = ContentBasedRecommender(n_topics=20).fit(train_matrix, items, item_meta)

print("Fitting Hybrid (SVD 60% + Content-based 40%) ...")
hybrid = HybridRecommender(svd, cb, pop, svd_weight=0.6).fit(train_matrix)

print("All models fitted.")
"""),

    # ── Evaluation ─────────────────────────────────────────────────────────
    md("## 4. Evaluation — Recall@K, NDCG@K, MAP@K"),
    code("""
recommend_fns = {
    'Popularity':     pop.recommend,
    'SVD (k=50)':    svd.recommend,
    'ALS':           als.recommend,
    'Content-based': cb.recommend,
    'Hybrid':        hybrid.recommend,
}

results = evaluate_all(recommend_fns, test_items, k_values=[10, 20, 50])

# Display as DataFrame
rows = []
for name, metrics in results.items():
    rows.append({'Model': name, **metrics})
df_results = pd.DataFrame(rows).set_index('Model')

# Highlight best per metric
display_cols = ['recall@10', 'ndcg@10', 'map@10', 'recall@20', 'recall@50']
df_display = df_results[display_cols].copy()

print("\\n=== Recommender Evaluation Results ===")
print(df_display.to_string(float_format='{:.4f}'.format))

# Flag popularity beat?
pop_r10 = results['Popularity']['recall@10']
for name, m in results.items():
    if name == 'Popularity':
        continue
    delta = m['recall@10'] - pop_r10
    symbol = '✓ beats' if delta > 0.005 else ('= ties' if abs(delta) <= 0.005 else '✗ loses')
    print(f"  {name}: {symbol} popularity by {delta:+.4f} Recall@10")
"""),

    # ── Comparison chart ────────────────────────────────────────────────────
    md("## 5. Visual Comparison"),
    code("""
models_list = list(results.keys())
recall10 = [results[m]['recall@10'] for m in models_list]
ndcg10   = [results[m]['ndcg@10']   for m in models_list]
map10    = [results[m]['map@10']    for m in models_list]

x = np.arange(len(models_list))
width = 0.26

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - width, recall10, width, label='Recall@10', color='#2196F3', alpha=0.85)
ax.bar(x,         ndcg10,   width, label='NDCG@10',   color='#FF9800', alpha=0.85)
ax.bar(x + width, map10,    width, label='MAP@10',    color='#4CAF50', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=20, ha='right')
ax.set_ylabel('Score')
ax.set_title('Recommender Comparison — Online Retail II\\nLeave-one-out, K=10')
ax.legend()

# Mark popularity baseline
ax.axvline(x=0, color='red', linestyle='--', alpha=0.35, linewidth=1.5)
ax.text(0.1, ax.get_ylim()[1] * 0.92, '← beat this', fontsize=8, color='red', alpha=0.7)

fig.tight_layout()
plt.show()
"""),

    # ── Recall@K curve ──────────────────────────────────────────────────────
    md("## 6. Recall@K Curve"),
    code("""
fig, ax = plt.subplots(figsize=(9, 5))
k_vals = [10, 20, 50]
colors = ['#9E9E9E', '#2196F3', '#E91E63', '#FF9800', '#4CAF50']
for (name, metrics), color in zip(results.items(), colors):
    recalls = [metrics[f'recall@{k}'] for k in k_vals]
    ax.plot(k_vals, recalls, 'o-', linewidth=2.5, color=color,
            label=name, markersize=7)

ax.set_xlabel('K')
ax.set_ylabel('Recall@K')
ax.set_title('Recall@K Curve — Online Retail II')
ax.legend(fontsize=9)
ax.set_xticks(k_vals)
fig.tight_layout()
plt.show()
"""),

    # ── SVD spectrum ─────────────────────────────────────────────────────────
    md("""## 7. SVD Analysis — Recommender Latent Space

SVD factorises the user-item interaction matrix $X \\approx U \\Sigma V^T$.

- $U$ rows = user latent vectors (purchase preference profiles)
- $V^T$ columns = item latent vectors (item characteristic profiles)
- $\\Sigma$ diagonal = importance of each latent factor

This is the same mathematics as in `src/dimensionality/svd_analysis.py`,
but applied to a *user-item* matrix instead of a *time×features* matrix.
The resulting singular vectors mean something completely different.
"""),
    code("""
from sklearn.utils.extmath import randomized_svd

# Full SVD on train matrix to inspect spectrum
U, S, Vt = randomized_svd(train_matrix.astype(np.float32), n_components=50, random_state=42)
ev = S**2 / (S**2).sum()
cumvar = np.cumsum(ev)

k90 = int(np.searchsorted(cumvar, 0.90)) + 1
k95 = int(np.searchsorted(cumvar, 0.95)) + 1

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].semilogy(range(1, len(S)+1), S, 'o-', color='#2196F3', markersize=4)
axes[0].set_xlabel('Component index')
axes[0].set_ylabel('Singular value (log)')
axes[0].set_title('SVD Spectrum — User-Item Matrix')
axes[0].axvline(x=k90, color='gray', linestyle='--', alpha=0.6,
                label=f'k={k90} (90% var)')
axes[0].legend()

axes[1].plot(range(1, len(S)+1), cumvar*100, 's-', color='#FF9800', markersize=4)
axes[1].axhline(90, color='gray', linestyle='--')
axes[1].axhline(95, color='red', linestyle='--')
axes[1].set_xlabel('Components k')
axes[1].set_ylabel('Cumulative explained variance (%)')
axes[1].set_title(f'Cumulative Variance  (90%→k={k90}, 95%→k={k95})')

fig.suptitle('SVD of Online Retail II User-Item Matrix', fontsize=13)
fig.tight_layout()
plt.show()

print(f"Intrinsic dimensionality: {k90} components explain 90% variance")
print(f"(Compare: retail lag-feature SVD needed k=4 for 90%)")
"""),

    # ── Cross-scenario comparison ────────────────────────────────────────────
    md("""## 8. Cross-Scenario SVD Comparison

**This is the headline of the project.**

The same SVD/PCA mathematics is deployed in two completely different domains.
The intrinsic dimensionality and the meaning of singular vectors differ because
the *input matrix* differs:

| | NZ Retail Forecasting | Online Retail II |
|---|---|---|
| **Matrix** | Time × Lag-features (quarterly) | Users × Items (e-commerce) |
| **Data type** | Continuous sales values | Implicit feedback (log qty) |
| **SVD k for 90% variance** | **4** | **many more** |
| **SV1 means** | Level (baseline demand) | Dominant purchase cluster |
| **SV2 means** | Trend / momentum | Secondary preference axis |
| **SV3 means** | Seasonal oscillation | Category grouping |
| **Production use** | Reconstruction ablation (rank-3 ≈ full) | Latent factors for ranking |

**Why k differs:** The retail lag-feature matrix is nearly rank-4 because
NZ retail sales has a strong linear structure (trend + quarterly seasonality).
The user-item matrix is high-rank because purchase behaviour is heterogeneous
— thousands of users have partially overlapping but individually distinct
preferences.
"""),
    code("""
# Load pre-computed cross-scenario summary if available, else compute inline
try:
    cross = pd.read_csv('../models/cross_scenario_svd_comparison.csv')
    print("Cross-scenario SVD summary:")
    print(cross.to_string(index=False))
except FileNotFoundError:
    print("Run python -m recommender.train_recommender first to generate summary CSV.")

# Visualise the key comparison
fig, ax = plt.subplots(figsize=(9, 4))
scenarios = ['NZ Retail\\n(time-series\\nlag features)', 'Online Retail II\\n(user-item\\ninteractions)']
k90_vals = [4, k90]  # from cells above

bars = ax.bar(scenarios, k90_vals, color=['#2196F3', '#FF9800'], alpha=0.85, width=0.4)
ax.set_ylabel('SVD components for 90% variance')
ax.set_title('Intrinsic Dimensionality Comparison\\n'
             'Same Mathematics, Different Matrices → Different Rank')
for bar, val in zip(bars, k90_vals, strict=True):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'k = {val}', ha='center', fontsize=13, fontweight='bold')

ax.set_ylim(0, max(k90_vals) * 1.3)
fig.tight_layout()
plt.show()

print("\\nKey insight: NZ retail sales ≈ 4D (trend + seasonality).")
print(f"User-item preferences ≈ {k90}D (heterogeneous, no dominant structure).")
"""),

    # ── Content-based topics ─────────────────────────────────────────────────
    md("## 9. Content-Based: NMF Topic Inspection"),
    code("""
# Show top words per NMF topic
feature_names = cb.feature_names
H = cb.topic_word_matrix  # (n_topics, vocab)

print("NMF Topic Inspection (top-5 words per topic):")
print("-" * 60)
for i, row in enumerate(H):
    top_idx = row.argsort()[::-1][:5]
    top_words = [feature_names[j] for j in top_idx]
    print(f"Topic {i+1:2d}: {', '.join(top_words)}")
"""),

    # ── Why do deep learning methods fail here too? ──────────────────────────
    md("""## 10. Structural Parallels — Small Data Across Both Scenarios

Both scenarios share a critical constraint: **small effective sample size**.

| Scenario | Training size | Effect |
|----------|--------------|--------|
| NZ Retail forecasting | ~102 quarters | LSTM/GRU overfit; Prophet + Ridge win |
| Online Retail II | ~3-4k users, sparse | ALS/SVD may only marginally beat popularity |

In both cases, the simpler, better-regularised model family wins:
- Forecasting: Prophet (additive decomposition) > LSTM (nonlinear RNN)
- Recommender: ALS/SVD (bilinear MF) > deep collaborative filtering

**The recommendation** in both domains: verify against the simplest possible
baseline (Holt-Winters or Ridge in forecasting; Popularity in recommenders)
before investing in deep learning. Scientific maturity means reporting the
baseline prominently even when it is embarrassingly competitive.
"""),

    # ── Leakage audit recap ──────────────────────────────────────────────────
    md("## 11. Leakage Audit Recap"),
    code("""
# Read the audit document
audit_path = '../diagnostics/recommender_audit.md'
try:
    with open(audit_path) as f:
        print(f.read())
except FileNotFoundError:
    print("Audit file not found — see diagnostics/recommender_audit.md")
"""),
]

nb = new_notebook(cells=cells)
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

NB_PATH.parent.mkdir(exist_ok=True)
with NB_PATH.open("w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written → {NB_PATH}")
