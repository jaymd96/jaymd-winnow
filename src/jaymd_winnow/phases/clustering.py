"""Phase 1: Feature clustering to reduce correlated features to independent representatives."""

import logging
import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from jaymd_winnow.config import ClusteringConfig

logger = logging.getLogger(__name__)


def cluster_features(
    X: np.ndarray,
    config: ClusteringConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster correlated features and select one representative per cluster.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        config: Clustering configuration.

    Returns:
        cluster_labels: Cluster assignment per feature, shape (n_features,).
        representative_indices: Column indices of representatives, shape (n_clusters,).
    """
    n_features = X.shape[1]

    # Drop zero-variance features from clustering (they cause NaN correlations)
    variances = np.var(X, axis=0)
    valid_mask = variances > 0
    n_valid = valid_mask.sum()

    if n_valid == 0:
        raise ValueError("All features have zero variance")

    if n_valid < config.max_clusters:
        logger.info(
            "n_valid_features (%d) < max_clusters (%d), skipping clustering",
            n_valid, config.max_clusters,
        )
        labels = np.arange(n_features)
        labels[~valid_mask] = -1
        representatives = np.where(valid_mask)[0]
        return labels, representatives

    X_valid = X[:, valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Pairwise absolute correlation → distance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(X_valid.T)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=config.method)
    cluster_ids = fcluster(Z, t=config.max_clusters, criterion="maxclust")

    # Map back to full feature space
    labels = np.full(n_features, -1, dtype=int)
    labels[valid_mask] = cluster_ids

    # Select representative per cluster: highest variance within cluster
    representatives = []
    for cid in range(1, cluster_ids.max() + 1):
        member_mask = cluster_ids == cid
        member_indices = valid_indices[member_mask]
        member_vars = variances[member_indices]
        best = member_indices[np.argmax(member_vars)]
        representatives.append(best)

    return labels, np.array(representatives, dtype=int)
