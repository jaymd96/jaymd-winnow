"""Stability selection implementation.

Bootstrap subsampling with L1-penalised models to identify features
with statistically reliable target associations. A feature is "stable"
if it is selected by a majority of (bootstrap, regularisation) combinations.

Reference: Meinshausen & Bühlmann (2010), "Stability selection",
Journal of the Royal Statistical Society B.
"""

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler


def _fit_subsample(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    alpha: float,
    sample_fraction: float,
    rng_seed: int,
) -> np.ndarray:
    """Fit an L1 model on a bootstrap subsample, return boolean feature mask."""
    rng = np.random.RandomState(rng_seed)
    n = X.shape[0]
    n_sub = max(1, int(n * sample_fraction))
    indices = rng.choice(n, size=n_sub, replace=False)
    X_sub, y_sub = X[indices], y[indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "regression":
            model = Lasso(alpha=alpha, max_iter=5000)
            model.fit(X_scaled, y_sub)
            coefs = model.coef_
        else:
            model = LogisticRegression(
                penalty="l1", solver="saga", C=1.0 / max(alpha, 1e-10),
                max_iter=5000,
            )
            model.fit(X_scaled, y_sub)
            coefs = model.coef_.ravel()

    return np.abs(coefs) > 1e-10


def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    n_bootstraps: int = 200,
    threshold: float = 0.6,
    alphas: np.ndarray | None = None,
    sample_fraction: float = 0.5,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run stability selection to identify reliably selected features.

    For each regularisation strength in `alphas`, draws `n_bootstraps` random
    half-samples, fits an L1 model, and records which features have nonzero
    coefficients. Features selected in >= `threshold` fraction of all
    (alpha, bootstrap) combinations are returned.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Target vector, shape (n_samples,).
        task: One of "regression", "binary", "multiclass".
        n_bootstraps: Number of bootstrap iterations per regularisation value.
        threshold: Selection frequency threshold in [0, 1].
        alphas: Regularisation strengths to sweep. If None, uses logspace(-3, 1, 25).
        sample_fraction: Fraction of data to subsample per bootstrap.
        n_jobs: Parallelism (-1 = all cores).

    Returns:
        selected_indices: Indices of features exceeding the threshold.
        stability_scores: Selection frequency per feature, shape (n_features,).
    """
    if alphas is None:
        alphas = np.logspace(-3, 1, 25)

    n_features = X.shape[1]
    total_fits = n_bootstraps * len(alphas)

    jobs = []
    for alpha in alphas:
        for b in range(n_bootstraps):
            seed = int((b * len(alphas) + hash(str(alpha))) % (2**31))
            jobs.append((alpha, seed))

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_subsample)(X, y, task, alpha, sample_fraction, seed)
        for alpha, seed in jobs
    )

    selection_counts = np.zeros(n_features)
    for mask in results:
        selection_counts += mask.astype(float)

    stability_scores = selection_counts / total_fits
    selected_indices = np.where(stability_scores >= threshold)[0]

    return selected_indices, stability_scores
