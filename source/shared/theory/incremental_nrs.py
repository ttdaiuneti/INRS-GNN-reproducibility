"""
Incremental NRS core — safety-radius (δ_B) maintenance, Theorem 1 (MHMG lift).
Granules G_B are a derived quantity (recomputed from δ_B on demand); Theorem 2
characterises how they change but they are not maintained as state here.
"""
import numpy as np


def calc_deltas(X, y):
    """Batch δ_B for all rows. X: (n, d), y: (n,)."""
    n = len(y)
    if n == 0:
        return np.array([])
    if X.shape[1] * len(y) > 2_000_000 or len(y) * len(y) > 500_000:
        return calc_deltas_rowwise(X, y)
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    deltas = np.full(n, np.inf)
    for i in range(n):
        enemy = y != y[i]
        ed = dist[i][enemy]
        if ed.size > 0:
            deltas[i] = ed.min()
    return deltas


def calc_deltas_rowwise(X, y):
    """O(n²d) time, O(nd) memory — for high-dimensional features."""
    n = len(y)
    deltas = np.full(n, np.inf)
    for i in range(n):
        diff = X - X[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        enemy = y != y[i]
        if enemy.any():
            deltas[i] = dist[enemy].min()
    return deltas


def calc_deltas_vectorized(X, y):
    """Batch δ_B for all rows via one Gram-matrix pairwise-distance evaluation.

    ‖x_i − x_j‖² = ‖x_i‖² + ‖x_j‖² − 2·x_iᵀx_j is formed with a single BLAS
    GEMM (``X @ X.T``); same-class entries are masked to +∞ and a row-min
    gives δ_B(x_i) = distance to the nearest enemy. Identical O(|V|²·|B|)
    arithmetic to :func:`calc_deltas_rowwise`, but the |B| reduction runs
    inside BLAS rather than a Python ``for``-loop over rows — this is the
    fair batch-δ comparator for the end-to-end timing benchmark
    (``review_opus_round2.md`` MAJOR-E2-R2: the previous benchmark timed the
    incremental path against a batch path whose δ term was an unvectorized
    row loop, 94–99 % of batch cost).

    Bit-compatible with :func:`calc_deltas_rowwise` up to floating-point
    rounding of the Gram identity (``test_batch_delta_vectorized_matches.py``
    checks max abs difference < 1e-6 and an identical +∞ pattern on Cora and
    CiteSeer). Not used on any correctness path — Table 3 / Corollary 1 keep
    the direct-difference :func:`calc_deltas`.
    """
    n = len(y)
    if n == 0:
        return np.array([])
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y)
    sq = np.einsum("ij,ij->i", X, X)
    # NumPy 2 + Apple Accelerate emits spurious FP RuntimeWarnings from the
    # BLAS matmul path on some builds; the Gram result is finite (verified
    # bit-for-bit against calc_deltas_rowwise in
    # test_batch_delta_vectorized_matches.py), so the warnings are silenced
    # locally rather than left to mislead.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        gram = X @ X.T
    d2 = sq[:, None] + sq[None, :] - 2.0 * gram
    np.maximum(d2, 0.0, out=d2)
    dist = np.sqrt(d2, out=d2)
    dist[y[:, None] == y[None, :]] = np.inf
    return dist.min(axis=1)


def calc_deltas_blocked(X, y, block=2048):
    """Batch delta_B via the Gram identity, computed in row-blocks so the
    working set is O(block * |V|) rather than O(|V|^2). Lets one full batch-delta
    call run at OGB scale (|V| ~ 1.7e5), where a dense |V| x |V| distance
    matrix (2.3e11 doubles) is infeasible -- used to anchor the "batch is not
    a practical per-insertion baseline at scale" comparison in e2e_ogb_scale.py.
    Same O(|V|^2 * |B|) arithmetic as calc_deltas_rowwise / calc_deltas_vectorized.
    """
    n = len(y)
    if n == 0:
        return np.array([])
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y)
    sq = np.einsum("ij,ij->i", X, X)
    out = np.full(n, np.inf)
    for s in range(0, n, block):
        e = min(s + block, n)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            g = X[s:e] @ X.T
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * g
        np.maximum(d2, 0.0, out=d2)
        dist = np.sqrt(d2, out=d2)
        dist[y[s:e, None] == y[None, :]] = np.inf
        out[s:e] = dist.min(axis=1)
    return out


def calc_granule_sizes(X, y, deltas):
    n = len(y)
    sizes = np.zeros(n, dtype=int)
    for i in range(n):
        diff = X - X[i]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        sizes[i] = int(np.sum(d < deltas[i]))
    return sizes


def incremental_insert(X, y, deltas, x_new, y_new):
    """
    Insert one object; return updated (X, y, deltas, gain, shrink).
    Implements Theorem 1 (δ_B friend-invariance / enemy-shrink). `gain` and
    `shrink` are the index sets Theorem 2 would act on if G_B were
    materialised; nothing downstream consumes them (granules are derived).
    """
    n = len(y)
    if n == 0:
        d_new = np.inf
        return (
            x_new[None, :],
            np.array([y_new]),
            np.array([d_new]),
            set(),
            set(),
        )

    diff = X - x_new
    dist_to = np.sqrt(np.sum(diff * diff, axis=1))
    enemy_mask = y != y_new

    # Vectorized equivalent of the per-node loop over Theorem 1(2) / Theorem 2
    # cases (friend within radius -> granule gain; enemy within radius ->
    # radius shrink). Element-for-element identical to the previous Python
    # for-loop. The distance kernel is the same direct-difference form used
    # by calc_deltas / calc_deltas_rowwise, so incremental vs. batch delta
    # agree bit-for-bit for any feature values (Corollary 1); the Gram-based
    # calc_deltas_vectorized is a timing-only comparator, never on this path.
    new_deltas = deltas.copy()
    closer = dist_to < deltas
    gain_mask = (~enemy_mask) & closer
    shrink_mask = enemy_mask & closer
    new_deltas[shrink_mask] = dist_to[shrink_mask]
    gain = set(np.flatnonzero(gain_mask).tolist())
    shrink = set(np.flatnonzero(shrink_mask).tolist())

    enemy_dist = dist_to[enemy_mask]
    delta_new = float(enemy_dist.min()) if enemy_dist.size > 0 else np.inf

    X_out = np.vstack([X, x_new[None, :]])
    y_out = np.append(y, y_new)
    deltas_out = np.append(new_deltas, delta_new)

    return X_out, y_out, deltas_out, gain, shrink


def batch_granule_sizes_after_insert(X_old, y_old, deltas_old, x_new, y_new, deltas_new):
    """Granule sizes on V_old after insert (for shrink verification)."""
    n = len(y_old)
    X_all = np.vstack([X_old, x_new[None, :]])
    y_all = np.append(y_old, y_new)
    sizes_old = calc_granule_sizes(X_old, y_old, deltas_old)
    sizes_new = calc_granule_sizes(X_all, y_all, deltas_new[: n + 1])
    return sizes_old, sizes_new[:n]
