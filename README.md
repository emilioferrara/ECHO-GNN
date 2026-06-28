# ECHO: Encoding Communities via High-Order Operators

**Research implementation.** Author: Emilio Ferrara.

ECHO is a scalable, self-supervised framework for **attributed community detection** in large
networks. It treats communities not as static partitions but as regions of *adaptive diffusion* on
semantic manifolds, combining attention-modulated high-order diffusion, a memory-sharded contrastive
objective, and an `O(|E| + N·k_max)` clustering extraction that scales to million-node graphs on a
single GPU.

The canonical, paper-reproducing model is:

```
echo_gnn.py        # ECHO v1.4 — full-batch research edition (recommended)
```

> **Note on versions.** `echo_gnn.py` (v1.4) is the implementation that reproduces the results
> reported in the paper. An earlier experimental refactor with an automatic encoder router,
> `legacy/echo_gnn_v3.py`, is retained for provenance only; its router can mis-route dense assortative
> graphs to a non-aggregating path and does **not** reproduce the paper's numbers. Use `echo_gnn.py`.

## Results (NMI on real attributed graphs)

| Method | Computers | Photo | Coauthor-CS | CoraFull |
|---|---|---|---|---|
| DGI | 0.232 | 0.442 | 0.528 | 0.352 |
| GDC | 0.327 | 0.362 | 0.638 | 0.487 |
| H2GCN | 0.025 | 0.047 | 0.092 | 0.058 |
| NodeFormer | 0.017 | 0.050 | 0.213 | 0.039 |
| **ECHO** | **0.564** | **0.637** | **0.664** | **0.502** |

ECHO attains the best NMI on every assortative benchmark, surpassing recent self-supervised,
heterophily-specific, and graph-transformer baselines.

## Usage

```python
import igraph as ig
from echo_gnn import ECHO   # alias of SelfSupervisedCommunityGNN_Alpha

# G: an igraph.Graph (or networkx.Graph);  X: optional [N, d] node features (numpy)
model = ECHO(feat_dim=X.shape[1], sage_hidden=128, diff_steps=1,
             epochs=200, temp=0.1, lr=5e-4, sparsity_penalty=1e-4, seed=42)
model.fit(G, X, use_amp=False)        # use_amp=False recommended on recent PyTorch
labels = model.predict()              # community assignment per node
```

Key arguments: `diff_steps` (number of high-order diffusion steps `K`), `temp` (contrastive
temperature `τ`), `sparsity_penalty` (`λ`, ℓ1 on attention), `cluster_threshold`. For benchmarking we
select `K ∈ {0,1,2}`, `τ`, and `λ` per dataset with a small grid.

## Requirements

```
python >= 3.9
torch  >= 2.0   (CUDA strongly recommended; the model is GPU-oriented)
numpy, python-igraph
```

## Running the LFR benchmark

```bash
python examples/run_lfr_benchmark.py --n 1000 --mu 0.1 --feat-dim 16 --epochs 50 --seed 42
```

Note: on synthetic LFR with degree-only features, ECHO is competitive but not dominant — its
advantage is a topology–feature synergy that materializes on real attributed graphs where node
features carry community signal (see the results table above).

## Citation

```bibtex
@article{ferrara2026echo,
  title={ECHO: Encoding Communities via High-order Operators},
  author={Ferrara, Emilio},
  journal={Machine Learning with Applications},
  year={2026}
}
```

## License
Apache 2.0.
