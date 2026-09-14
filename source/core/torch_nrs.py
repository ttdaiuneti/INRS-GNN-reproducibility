"""Torch/MPS helpers for δ and rough adjacency (Apple Metal GPU on M4)."""
import numpy as np
import torch


def get_device(prefer_mps=True):
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _use_rowwise(n, d):
    return n * n > 500_000 or n * d > 2_000_000


def calc_deltas_torch(X, y, device=None):
    """
    δ_B per node — vectorized on GPU when small; rowwise CPU otherwise.

    WARNING (label-safe): this uses `y` for every row, enemy included. Valid only
    as an oracle/theory check (Theorem 1-2 parity, D0 locality) where seeing every
    label is legitimate. NEVER feed the output into a model that predicts labels
    for any node whose `y` was used here — use calc_deltas_torch_trusted instead
    for GNN experiments (train/val/test split), or ground-truth test/val labels
    leak into the model's input structure. See test_label_safe_leakage_paper1.py.
    """
    n, d = X.shape[0], X.shape[1]
    if _use_rowwise(n, d):
        from core.incremental_nrs import calc_deltas_rowwise
        return torch.as_tensor(calc_deltas_rowwise(X, y), dtype=torch.float32)

    device = device or get_device()
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    dist = torch.sqrt(torch.sum(diff * diff, dim=2))
    same = y_t.unsqueeze(0) == y_t.unsqueeze(1)
    dist_enemy = dist.masked_fill(same, float("inf"))
    deltas = dist_enemy.min(dim=1).values
    return deltas


def calc_deltas_torch_trusted(X, y_hat, trusted, device=None):
    """
    Label-safe δ_B: enemy candidates restricted to `trusted` nodes only (train
    labels + optional high-confidence pseudo-labels — see
    graph_data.build_trusted_labels_via_gcn). Untrusted nodes get δ=+inf (no
    claimed enemy), which batch_rough_adjacency_torch already maps to ψ=0 for
    any edge touching them — i.e. unlabeled/unsure nodes contribute no rough
    signal instead of leaking their held-out ground truth. Mirrors
    core.rough_partition.calc_deltas_trusted (Paper 2), ported to
    torch for GPU parity with calc_deltas_torch.
    """
    n, d = X.shape[0], X.shape[1]
    if _use_rowwise(n, d):
        from core.rough_partition import calc_deltas_trusted
        deltas_np = calc_deltas_trusted(X, np.asarray(y_hat), np.asarray(trusted))
        return torch.as_tensor(deltas_np, dtype=torch.float32)

    device = device or get_device()
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_hat, dtype=torch.long, device=device)
    trust_t = torch.as_tensor(np.asarray(trusted, dtype=bool), device=device)

    diff = x.unsqueeze(0) - x.unsqueeze(1)
    dist = torch.sqrt(torch.sum(diff * diff, dim=2))
    same = y_t.unsqueeze(0) == y_t.unsqueeze(1)
    # candidate enemy (row i, col j) must be: trusted, different label
    enemy_ok = trust_t.unsqueeze(0) & (~same)
    dist_enemy = dist.masked_fill(~enemy_ok, float("inf"))
    deltas = dist_enemy.min(dim=1).values
    # untrusted rows never had a valid "own label" to compare against -> inf
    deltas = torch.where(trust_t, deltas, torch.full_like(deltas, float("inf")))
    return deltas


def batch_rough_adjacency_torch(X, y, deltas, adj, device=None, eps=1e-9):
    """
    W[v,u] = 1[(v,u) in E] * exp(-ρ/(δ_v+δ_u+ε)). adj: bool (n,n).
    Falls back to NumPy edge loop when n*d is too large for GPU pairwise tensor.
    """
    n, d = X.shape[0], X.shape[1]
    if _use_rowwise(n, d):
        from core.incremental_nrs import calc_deltas_rowwise
        from core.rough_adjacency import batch_rough_adjacency

        if torch.is_tensor(deltas):
            deltas_np = deltas.detach().cpu().numpy()
        else:
            deltas_np = np.asarray(deltas, dtype=np.float64)
        B_cols = np.arange(d)
        return batch_rough_adjacency(X, y, deltas_np, adj, B_cols, eps)

    device = device or get_device()
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    adj_t = torch.as_tensor(adj, dtype=torch.bool, device=device)
    if not torch.is_tensor(deltas):
        d = torch.as_tensor(deltas, dtype=torch.float32, device=device)
    else:
        d = deltas.to(device=device, dtype=torch.float32)

    n = x.shape[0]
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    rho = torch.sqrt(torch.sum(diff * diff, dim=2))
    denom = d.unsqueeze(1) + d.unsqueeze(0) + eps
    psi = torch.exp(-rho / denom)
    psi = torch.where(torch.isinf(d).unsqueeze(1) | torch.isinf(d).unsqueeze(0), 
                      torch.zeros_like(psi), psi)
    W = torch.where(adj_t, psi, torch.zeros_like(psi))
    W.fill_diagonal_(0.0)
    return W.cpu().numpy()


def time_cpu_vs_mps_build(X, y, adj, repeats=3):
    """Benchmark numpy vs MPS rough matrix build."""
    import time
    from core.incremental_nrs import calc_deltas
    from core.rough_adjacency import batch_rough_adjacency

    B_cols = np.arange(X.shape[1])
    times_cpu = []
    for _ in range(repeats):
        t0 = time.time()
        d = calc_deltas(X, y)
        batch_rough_adjacency(X, y, d, adj, B_cols)
        times_cpu.append(time.time() - t0)

    mps = get_device()
    times_mps = []
    if mps.type == "mps":
        for _ in range(repeats):
            t0 = time.time()
            d_t = calc_deltas_torch(X, y, mps)
            batch_rough_adjacency_torch(X, y, d_t, adj, mps)
            if mps.type == "mps":
                torch.mps.synchronize()
            times_mps.append(time.time() - t0)

    return {
        "device": str(mps),
        "cpu_mean_s": float(np.mean(times_cpu)),
        "mps_mean_s": float(np.mean(times_mps)) if times_mps else None,
        "speedup_x": (
            float(np.mean(times_cpu) / np.mean(times_mps)) if times_mps else None
        ),
    }
