#!/usr/bin/env python3
"""
One-step dataset preparation (review 2026-09-10, finding m6).

The README previously told readers to run `bash shared/link_data.sh`, which is
not part of the pinned submodule, and the ogbn-arxiv scripts referred to a
"download step in this file's __main__" that did not exist. This script is that
step, and it lives in this repository so a fresh clone needs nothing else.

  * Planetoid (Cora, CiteSeer, PubMed) are fetched on demand by PyTorch
    Geometric; this script only creates the cache directory.
  * ogbn-arxiv is downloaded through the `ogb` package (pinned in
    requirements.txt) and cached as three .npy arrays, whose SHA-256 digests are
    printed and checked against the values used for the reported results.

Usage:
    python3 experiments/fetch_data.py            # fetch what is missing, verify
    python3 experiments/fetch_data.py --verify   # verify only, never download
"""
import argparse
import hashlib
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(_ROOT, "core", "data")
OGB_DIR = os.path.join(DATA, "ogb")

# Digests of the arrays behind every ogbn-arxiv number reported in the paper.
EXPECTED = {
    "arxiv_X.npy": "fdb507bf68c08e17ac1d5760f27cba520029ce7ee74845c38211f347d38e0a49",
    "arxiv_y.npy": "c8dd0692e9d38bb138be13691546d3953422d93c763ff7a46c3ee7895e40fbb6",
    "arxiv_edge_index.npy": "aab407d7d75cbf0a296a7ed7007bf75f0466e3f6c90d76e945aa03a6c09d6997",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_arxiv():
    from ogb.nodeproppred import NodePropPredDataset  # pinned: ogb==1.3.6

    os.makedirs(OGB_DIR, exist_ok=True)
    ds = NodePropPredDataset(name="ogbn-arxiv", root=OGB_DIR)
    graph, labels = ds[0]
    np.save(os.path.join(OGB_DIR, "arxiv_X.npy"), graph["node_feat"])
    np.save(os.path.join(OGB_DIR, "arxiv_y.npy"), labels.reshape(-1))
    np.save(os.path.join(OGB_DIR, "arxiv_edge_index.npy"), graph["edge_index"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check existing caches only; do not download")
    args = ap.parse_args()

    os.makedirs(os.path.join(DATA, "pyg"), exist_ok=True)
    print(f"Planetoid cache directory: {os.path.join(DATA, 'pyg')} "
          "(PyTorch Geometric downloads Cora/CiteSeer/PubMed on first use)")

    missing = [f for f in EXPECTED if not os.path.exists(os.path.join(OGB_DIR, f))]
    if missing:
        if args.verify:
            print(f"[FAIL] missing ogbn-arxiv caches: {missing}")
            return 1
        print(f"Downloading ogbn-arxiv (missing: {missing}) ...", flush=True)
        download_arxiv()

    ok = True
    for fname, expect in EXPECTED.items():
        path = os.path.join(OGB_DIR, fname)
        got = sha256(path)
        mark = "OK " if got == expect else "MISMATCH"
        if got != expect:
            ok = False
        print(f"  {mark} {fname}  {got}")
    if not ok:
        print("[WARN] a digest differs from the arrays used for the reported "
              "results; re-download or report the difference.")
        return 1
    print("[PASS] all dataset caches present and matching")
    return 0


if __name__ == "__main__":
    sys.exit(main())
