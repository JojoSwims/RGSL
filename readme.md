# Regularized Graph Structure Learning (RGSL)

This repository is based on the original implementation available at
<https://github.com/alipay/RGSL> and accompanies the paper:

> Hongyuan Yu, Ting Li, Weichen Yu, Jianguo Li, Yan Huang, Liang Wang and Alex Liu,
> "Regularized Graph Structure Learning with Explicit and Implicit Knowledge for
> Multi-variates Time-Series Forecasting," IJCAI 2022.

The goal of this fork is to make RGSL easier to apply to your own
time-series datasets.

## Preparing Data

RGSL expects two files for a dataset:

1. **Node features** – a `numpy` archive (`.npz`) containing one key named
   `data`. Its value should be an array of shape `(T, N, F)` where `T` is the
   sequence length, `N` the number of nodes and `F` the number of features per
   node.
2. **Adjacency matrix** – a CSV file describing graph connections with three
   columns `from`, `to` and `cost`. Each row denotes a directed edge from node
   `from` to node `to` with weight `cost`.

Example adjacency file:

```csv
from,to,cost
0,1,1.0
1,2,0.5
```

A helper function `save_triplets_csv` is available in
`lib/utils.py` for converting a dense `N x N` adjacency matrix into this format.

## Adapting the Code

1. **Load your dataset** – edit `load_st_dataset` in `lib/load_dataset.py`
   to point to your `.npz` file and return the loaded array.
2. **Create a config** – copy one of the `model/*.conf` files and update:
   - `num_nodes`: number of graph nodes
   - `adj_filename`: path to your adjacency CSV
   - other training parameters if needed
3. **Run** – execute `python model/Run.py --dataset YOUR_DATASET`.
   The dataset name should match the case handled inside `load_st_dataset`.

## Important Parameters

The configuration files contain the following commonly modified options:

- `lag` – length of input sequence (history window)
- `horizon` – prediction horizon
- `batch_size`, `epochs`, `lr_init` – training hyper parameters
- `cheb_order` – order of Chebyshev polynomials for graph convolution

Each dataset has its own `.conf` file. The filename should follow the pattern
`<DATASET>_<MODEL>.conf` as in `METRLA_RGSL.conf`.

## Helper Utilities

To convert an existing dense adjacency matrix `A` (numpy array) to the required
CSV format:

```python
from lib.utils import save_triplets_csv
save_triplets_csv(A, 'distance.csv')
```

This produces a file compatible with `get_adjacency_matrix` used by the model.

## Citation

If you use this code, please cite the IJCAI paper mentioned above.
