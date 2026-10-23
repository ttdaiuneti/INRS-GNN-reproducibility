"""
Incremental rough adjacency — Theorem 3 (adjacency locality).
"""
import numpy as np


def nrs_psi(rho, delta_v, delta_u, eps=1e-9):
    """ψ(ρ, δ_v, δ_u) = exp(-ρ / (δ_v + δ_u + ε))."""
    denom = delta_v + delta_u + eps
    if np.isinf(delta_v) or np.isinf(delta_u):
        return 0.0
    return float(np.exp(-rho / denom))


def paired_rg_weights(x_v, x_u, B_strict, B_loose, lam=0.5):
    """RG-GCN style paired min/max on attribute subsets."""
    d_strict = np.abs(x_v[B_strict] - x_u[B_strict])
    d_loose = np.abs(x_v[B_loose] - x_u[B_loose])
    r_min = float(d_strict.min()) if len(B_strict) else 0.0
    r_max = float(d_loose.max()) if len(B_loose) else 0.0
    return lam * r_max + (1.0 - lam) * r_min


def rough_weight_nrs(v, u, x, y, deltas, adj, B_idx, eps=1e-9):
    """w(v,u) for v != u; returns 0 if no edge."""
    if not adj[v, u]:
        return 0.0
    rho = float(np.linalg.norm(x[v, B_idx] - x[u, B_idx]))
    return nrs_psi(rho, deltas[v], deltas[u], eps)


def batch_rg_adjacency(x, adj, lam=0.5, normalize=True):
    """RG-GCN style paired min/max on half/half feature split."""
    d = x.shape[1]
    mid = d // 2
    B_strict = np.arange(0, mid)
    B_loose = np.arange(mid, d)
    n = x.shape[0]
    W = np.zeros((n, n))
    for v in range(n):
        for u in range(v + 1, n):
            if adj[v, u]:
                w = paired_rg_weights(x[v], x[u], B_strict, B_loose, lam)
                W[v, u] = W[u, v] = w
    if normalize and W.max() > 0:
        W = W / W.max()
    return W


def batch_rough_adjacency(x, y, deltas, adj, B_idx, eps=1e-9):
    """Full symmetric rough adjacency (small graphs only).

    Pure-Python double loop over all n^2/2 pairs with an `adj[v,u]` guard
    inside; used throughout the pilot/theory correctness checks. See
    batch_rough_adjacency_vectorized below for a bit-identical, edge-list
    vectorized comparator used to give an honest (rather than
    reference-implementation-dependent) speedup baseline in the timing
    benchmark -- review_opus_round1.md MAJOR-E2.
    """
    n = len(y)
    W = np.zeros((n, n))
    for v in range(n):
        for u in range(v + 1, n):
            if adj[v, u]:
                w = nrs_psi(
                    float(np.linalg.norm(x[v, B_idx] - x[u, B_idx])),
                    deltas[v],
                    deltas[u],
                    eps,
                )
                W[v, u] = W[u, v] = w
    return W


def batch_rough_adjacency_vectorized(x, y, deltas, adj, B_idx, eps=1e-9):
    """Edge-list, fully vectorized rough adjacency -- bit-identical output to
    batch_rough_adjacency (verified in
    experiments/test_batch_rough_adjacency_vectorized_matches.py: max abs
    difference exactly 0 on Cora/CiteSeer, including delta=+inf entries) but
    O(|E| * |B|) instead of O(|V|^2 + |E| * |B|): only pairs with adj[v,u]
    are ever visited, and rho/psi are computed for all edges in one NumPy
    call rather than per-pair Python-level nrs_psi calls.
    """
    n = len(y)
    vs, us = np.triu_indices(n, k=1)
    edge_mask = adj[vs, us]
    ev, eu = vs[edge_mask], us[edge_mask]
    if ev.size == 0:
        return np.zeros((n, n))
    diff = x[ev][:, B_idx] - x[eu][:, B_idx]
    rho = np.sqrt(np.sum(diff * diff, axis=1))
    delta_v, delta_u = deltas[ev], deltas[eu]
    denom = delta_v + delta_u + eps
    w = np.exp(-rho / denom)
    inf_mask = np.isinf(delta_v) | np.isinf(delta_u)
    w[inf_mask] = 0.0
    W = np.zeros((n, n))
    W[ev, eu] = w
    W[eu, ev] = w
    return W


def incremental_rough_adjacency_update(
    x, y, deltas, adj, B_idx, v_new_idx, touch_str, S_delta, eps=1e-9
):
    """
    Recompute rows/cols for T_adj = touch_str ∪ S_delta.
    Returns (W_partial_updates dict (v,u)->w, T_adj).
    """
    n = len(y)
    T_adj = set(touch_str) | set(S_delta)
    updates = {}
    for v in T_adj:
        for u in range(n):
            if v == u:
                continue
            if adj[v, u]:
                rho = float(np.linalg.norm(x[v, B_idx] - x[u, B_idx]))
                w = nrs_psi(rho, deltas[v], deltas[u], eps)
                updates[(v, u)] = w
                updates[(u, v)] = w
    return updates, T_adj


def structural_touch_set(n_old, neighbors_of_new):
    """T_str = {v_new} ∪ neighbors in V_old."""
    return {n_old} | set(neighbors_of_new)


# ---------------------------------------------------------------------------
# Sparse-adjacency variants (Proposition 4's adjacency-list representation:
# O(|T_adj| * dbar) instead of O(|T_adj| * |V|)). A dense |V| x |V| adjacency
# or W is infeasible past ~3e4 nodes; these take a scipy CSR adjacency and
# never materialise a dense matrix, so the same maintenance runs at OGB scale
# (e2e_ogb_scale.py). Bit-identical to the dense versions on Cora/CiteSeer
# (test_sparse_rough_adjacency_matches.py).
# ---------------------------------------------------------------------------

def incremental_rough_adjacency_update_sparse(
    x, y, deltas, adj_csr, B_idx, touch_str, S_delta, eps=1e-9, active_mask=None
):
    """Recompute rough weights for edges incident to T_adj = touch_str u S_delta,
    visiting only the true neighbours of each v in T_adj via the CSR row slice
    (O(|T_adj| * dbar)). ``adj_csr`` is a symmetric scipy.sparse CSR matrix.

    ``active_mask`` (optional bool array over adj_csr's node set): when given,
    ``adj_csr`` is the *full* graph and only edges to active nodes are updated,
    so no per-insertion CSR growth is needed (e2e_ogb_scale.py). When None the
    matrix is taken to be exactly the active subgraph.

    Returns (updates dict (v,u)->w, T_adj); v == u self-pairs skipped.
    """
    T_adj = set(touch_str) | set(S_delta)
    indptr, indices = adj_csr.indptr, adj_csr.indices
    updates = {}
    for v in T_adj:
        xv = x[v, B_idx]
        dv = deltas[v]
        for u in indices[indptr[v]:indptr[v + 1]]:
            u = int(u)
            if u == v or (active_mask is not None and not active_mask[u]):
                continue
            rho = float(np.linalg.norm(xv - x[u, B_idx]))
            w = nrs_psi(rho, dv, deltas[u], eps)
            updates[(v, u)] = w
            updates[(u, v)] = w
    return updates, T_adj


def batch_rough_adjacency_edgelist(x, y, deltas, adj_csr, B_idx, eps=1e-9):
    """Full rough adjacency as a COO triple (rows, cols, vals) over the upper
    triangle of a CSR adjacency -- O(|E| * |B|), no dense |V| x |V| matrix.
    Bit-identical values to batch_rough_adjacency on the shared edge set.
    """
    coo = adj_csr.tocoo()
    m = coo.row < coo.col
    ev, eu = coo.row[m], coo.col[m]
    if ev.size == 0:
        return np.array([], int), np.array([], int), np.array([])
    diff = x[ev][:, B_idx] - x[eu][:, B_idx]
    rho = np.sqrt(np.sum(diff * diff, axis=1))
    dv, du = deltas[ev], deltas[eu]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w = np.exp(-rho / (dv + du + eps))
    w[np.isinf(dv) | np.isinf(du)] = 0.0
    return ev, eu, w
