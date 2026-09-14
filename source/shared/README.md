# nrs-shared

Lõi NRS dùng chung cho [INRS-GNN](https://github.com/ttdaiuneti/INRS-GNN) (Paper 1) và [RNG-GCN](https://github.com/ttdaiuneti/RNG-GCN) (Paper 2).

## Modules

| File | Role |
|------|------|
| `graph_data.py` | Planetoid / PyG loaders |
| `theory/incremental_nrs.py` | δ, granule, incremental insert |
| `theory/rough_neighborhood_partition.py` | Lower/Boundary edge partition |
| `theory/torch_nrs_mps.py` | Torch/MPS δ and rough adjacency |

## Dùng trong paper repo

Mỗi paper repo mount repo này tại `shared/` (git submodule):

```bash
git submodule update --init --recursive
export PYTHONPATH="$(pwd):$(pwd)/shared:$PYTHONPATH"
```

## Data

Dataset cache (Planetoid, PyG, OGB) tải vào `data/` — không commit (gitignored).

```bash
pip install -r requirements.txt
```
