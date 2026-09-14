"""
Rough neighborhood edge partition for Paper 2 (RNG-GCN).

Classify structural edges into Lower (mutual class-pure granule) vs Boundary
using NRS safety radii δ_B and granules G_B — static batch on full graph.
"""
import numpy as np

from core.incremental_nrs import calc_deltas


def pairwise_dist(X):
    """Full (n,n) Euclidean distances; use rowwise if too large."""
    n, d = X.shape
    if n * n > 500_000:
        dist = np.zeros((n, n))
        for i in range(n):
            diff = X - X[i]
            dist[i] = np.sqrt(np.sum(diff * diff, axis=1))
        return dist
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def in_granule(dist_matrix, deltas, i, j):
    """j in G_B(i): dist(i,j) < δ(i), same class implied by granule def."""
    return dist_matrix[i, j] < deltas[i]


def calc_deltas_trusted(X, y_hat, trusted, dist=None):
    """
    Safety radii using only trusted nodes as potential enemies.
    Untrusted nodes receive +inf and never enter the Lower test as endpoints.
    """
    n = len(y_hat)
    if dist is None:
        dist = pairwise_dist(X)
    deltas = np.full(n, np.inf)
    trusted = np.asarray(trusted, dtype=bool)
    for i in range(n):
        if not trusted[i]:
            continue
        enemy = trusted & (y_hat != y_hat[i])
        if enemy.any():
            deltas[i] = dist[i, enemy].min()
    return deltas


def partition_edges_label_safe(X, y_hat, trusted, adj, dist=None, mode="relaxed"):
    """
    Label-safe Lower/Boundary partition.

    Only nodes in `trusted` may contribute labels. An edge is Lower iff both
    endpoints are trusted, share y_hat, and satisfy the distance test.
    `y_hat` on untrusted nodes is ignored (may be dummy).
    """
    n = len(y_hat)
    trusted = np.asarray(trusted, dtype=bool)
    if dist is None:
        dist = pairwise_dist(X)
    deltas = calc_deltas_trusted(X, y_hat, trusted, dist=dist)
    lower = np.zeros((n, n), dtype=bool)
    boundary = np.zeros((n, n), dtype=bool)

    for v in range(n):
        neighbors = np.where(adj[v])[0]
        for u in neighbors:
            if u <= v:
                continue
            both_trusted = bool(trusted[v] and trusted[u])
            same_class = both_trusted and (y_hat[v] == y_hat[u])
            d_vu = dist[v, u]
            if mode == "cross":
                is_lower = same_class
            elif mode == "relaxed":
                is_lower = same_class and d_vu < min(deltas[v], deltas[u])
            else:
                is_lower = (
                    same_class
                    and in_granule(dist, deltas, v, u)
                    and in_granule(dist, deltas, u, v)
                )
            if is_lower:
                lower[v, u] = lower[u, v] = True
            else:
                boundary[v, u] = boundary[u, v] = True
    return lower, boundary, deltas, dist


def build_trusted_labels(y_train_only, train_mask, pred, conf, tau):
    """
    Train nodes keep ground-truth labels. Other nodes enter the trusted set
    only if teacher confidence >= tau, using the teacher prediction.
    """
    n = len(y_train_only)
    trusted = train_mask.copy()
    y_hat = np.full(n, -1, dtype=int)
    y_hat[train_mask] = y_train_only[train_mask]
    extra = (~train_mask) & (conf >= tau)
    trusted[extra] = True
    y_hat[extra] = pred[extra]
    return y_hat, trusted


def partition_edges(X, y, adj, deltas=None, mode="mutual"):
    """
    Partition edges of adj into Lower and Boundary masks (symmetric).

    mode:
      mutual — Lower if same class AND mutual granule (strict; D0).
      relaxed — Lower if same class AND dist < min(δ_v, δ_u) (D1).
      cross — Lower if same class only on structural edge (baseline ablation).
    Boundary: connected but not Lower.
    """
    n = len(y)
    if deltas is None:
        deltas = calc_deltas(X, y)

    dist = pairwise_dist(X)
    lower = np.zeros((n, n), dtype=bool)
    boundary = np.zeros((n, n), dtype=bool)

    for v in range(n):
        neighbors = np.where(adj[v])[0]
        for u in neighbors:
            if u <= v:
                continue
            d_vu = dist[v, u]
            same_class = y[v] == y[u]

            if mode == "cross":
                is_lower = same_class
            elif mode == "relaxed":
                is_lower = same_class and d_vu < min(deltas[v], deltas[u])
            else:  # mutual
                is_lower = (
                    same_class
                    and in_granule(dist, deltas, v, u)
                    and in_granule(dist, deltas, u, v)
                )

            if is_lower:
                lower[v, u] = lower[u, v] = True
            else:
                boundary[v, u] = boundary[u, v] = True

    return lower, boundary, deltas, dist


def edge_region_stats(adj, lower, boundary, edge_mask=None):
    """
    Fraction of edges (in edge_mask if given) that are Lower vs Boundary.
    edge_mask: boolean (n,n) — e.g. fake edges only.
    """
    if edge_mask is None:
        edge_mask = adj.copy()
    # count undirected edges once
    triu = np.triu(edge_mask & (adj | lower | boundary), k=1)
    n_edges = int(triu.sum())
    if n_edges == 0:
        return {"n_edges": 0, "pct_lower": None, "pct_boundary": None}
    low = int(np.triu(lower & edge_mask, k=1).sum())
    bnd = int(np.triu(boundary & edge_mask, k=1).sum())
    return {
        "n_edges": n_edges,
        "n_lower": low,
        "n_boundary": bnd,
        "pct_lower": low / n_edges,
        "pct_boundary": bnd / n_edges,
    }


def build_boundary_aware_adjacency(
    adj, lower, boundary, w_lower=1.0, w_boundary=0.1
):
    """Weighted adjacency: suppress boundary edges (no structural backbone)."""
    W = np.zeros_like(adj, dtype=np.float64)
    W[lower] = w_lower
    W[boundary] = w_boundary
    return W


def build_hybrid_rng_adjacency(adj, lower, boundary, lambda_boundary=0.3):
    """
    Structural backbone A=1 on all edges; scale boundary region by lambda_boundary.
    Preserves connectivity on clean graphs while damping ambiguous/noisy edges.
    """
    W = adj.astype(np.float64).copy()
    W[boundary] *= lambda_boundary
    # lower edges stay at 1.0
    return W


def inject_random_edges(adj, y, n_add, rng, heterophilous_bias=0.8):
    """
    Add n_add random non-edges. Prefer opposite-class pairs (fake / noisy links).
    Returns noisy_adj, fake_edge_mask (symmetric, only new edges True).
    """
    n = adj.shape[0]
    fake = np.zeros((n, n), dtype=bool)
    added = 0
    max_tries = n_add * 50
    tries = 0
    while added < n_add and tries < max_tries:
        tries += 1
        v = rng.integers(0, n)
        u = rng.integers(0, n)
        if v == u or adj[v, u]:
            continue
        if rng.random() < heterophilous_bias and y[v] == y[u]:
            continue
        adj = adj.copy()
        adj[v, u] = adj[u, v] = True
        fake[v, u] = fake[u, v] = True
        added += 1
    return adj, fake, added
