# Machine-Learning-Basic-Projects

This repo contains the projects I completed where each folder includes a runnable notebook, brief notes, and outputs (plots/metrics). Below I’ve summarized what I did and what I learned from each task.

## Task 2 — Customer Segmentation (Mall Customers)

### What I built

- Loaded the Mall_Customers.csv dataset with kagglehub (no manual download).
- Cleaned/selected features and standardized them.
- Used K-Means with elbow + silhouette analysis; chose k = 6 based on metrics.
- Visualized clusters in 2D (Income vs Spending) and 3D / PCA.
- Computed per-cluster summaries and wrote customer personas (e.g., young high-income/high-spend VIPs).
- Tried DBSCAN (various eps) and documented why it underperformed (dense, spherical-ish clusters).

### What I learned

- Why feature scaling matters for distance-based methods.
- How to balance inertia vs. silhouette to pick k.
- Translating clusters into business personas and actionable marketing ideas.
- Recognizing when DBSCAN isn’t a good fit (noisy output, weak density structure).

## Task 3 — Forest Cover Type Classification

### What I built

- Pulled Covertype via fetch_covtype (no external files).
- Preprocessing with a ColumnTransformer pipeline: scale numeric, passthrough binary one-hots.
- Trained Random Forest (baseline) and XGBoost (bonus).
- Produced confusion matrices, classification reports, and feature importance plots.

### What I learned

- End-to-end modeling with scikit-learn pipelines (cleaner, reproducible).
- Fixing label issues for XGBoost (mapping 1..7 → 0..6).
- Reading feature importances (e.g., elevation, horizontal/vertical distances, hillshade).
- Why tree ensembles often outperform simple baselines on tabular data.

## Task 5 — Movie Recommendation System (MovieLens 100K)

### What I built

- Loaded MovieLens 100K (KaggleHub or GroupLens URL fallback).
- Built a user–item rating matrix and a per-user holdout split (hide 20% per user).
- Implemented User-based Collaborative Filtering:
- mean-center ratings, cosine similarity, top-K neighbors, predict unseen items.
- Evaluated with Precision@10 on hidden interactions (≥4 considered relevant).
- Generated top-N recommendations for any user.

### Extras:

- Item-based CF and SVD (low-rank reconstruction) for comparison.
- Popularity baseline, MAP@K / nDCG@K, neighbor-sweep, cold-start fallback from item similarity.
- Saved predictions for fast reuse.

### What I learned

- How to avoid data leakage with per-user train/test splits.
- Implementing ranking metrics (Precision@K, nDCG@K, MAP@K).
- Trade-offs between User-CF / Item-CF / Matrix Factorization.
- Practicalities: similarity sparsity, neighbor count tuning, coverage vs. accuracy.

## Task 6 — Music Genre Classification (GTZAN)

### What I built

- Loaded GTZAN with robust I/O (skipped a known corrupted file: jazz.00054.wav).
- Converted audio to mel-spectrograms, normalized to [0,1].
- Trained a CNN (Keras) and achieved around 76% test accuracy.
- Reported per-class precision/recall/F1; noted weak classes (rock/disco/reggae).

### What I learned

- Audio ML workflow: feature extraction → model → evaluation.
- Handling real-world dataset issues (corrupted files, backend fallbacks).
- Why augmentation (SpecAugment, time-crops) and transfer learning (e.g., EfficientNet on 3-channel mel/Δ/ΔΔ) can boost accuracy.


## Key Takeaways Across Projects

- Pipelines & Reproducibility: using Pipeline, fixed seeds, and clear data splits keeps work repeatable and clean.
- Right Tool for the Data: K-Means for compact clusters; ensembles for tabular; CF for recommendations; CNNs for spectrograms.

## Personal Reflection

It had been a while since I last did end-to-end ML projects, so I’d honestly forgotten a bunch of the day-to-day process—clean splits, scaling, leakage checks, metric choices, and the little things like fixing label encodings or skipping bad files. Working through these tasks refreshed that muscle memory: building pipelines, comparing baselines, reading confusion matrices the right way, and stress-testing evaluation. I leaned on docs and small checklists to get back up to speed, and by the later tasks I was moving much faster with cleaner, more reproducible notebooks. This sprint was a good reset and I’m keeping these templates for future projects.
