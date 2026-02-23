# ECHO: Encoding Communities via High-Order Operators

**Research Implementation --- v3.0.0**\
Author: Emilio Ferrara\
Year: 2026

This repository contains the original research implementation of
**ECHO**, a scalable, self-supervised framework for attributed community
detection in large-scale networks.

The core implementation is preserved in its original research form:

    echo_gnn_v3.py

This file is intentionally kept unchanged to reflect the version
described in the paper.

------------------------------------------------------------------------

# Abstract

Community detection in attributed networks faces a fundamental dual
barrier:

-   **Semantic Wall** --- Over-smoothing and heterophilic poisoning in
    graph neural networks.
-   **Systems Wall** --- Quadratic memory bottlenecks (O(N²)) in
    contrastive learning and similarity extraction.

ECHO dismantles both by introducing:

-   A **Topology-Aware Router** that dynamically selects inductive bias.
-   **Attention-Guided Multi-Scale Diffusion** to prevent semantic
    collapse.
-   A **Memory-Sharded Full-Batch Contrastive Objective**.
-   A **Chunked O(N·K) Similarity Extraction** method that eliminates
    O(N²) clustering bottlenecks.

The architecture is designed to scale to **million-node attributed
graphs** on a single GPU while preserving exact gradient fidelity.

------------------------------------------------------------------------

# Core Architectural Contributions

## 1. Topology-Aware Routing

Before training begins, ECHO evaluates:

-   Feature sparsity\
-   Structural density\
-   Semantic assortativity

Based on these unsupervised structural heuristics, the model
automatically routes the graph through:

-   **Isolating Encoder (MLP)** for dense or heterophilic graphs\
-   **Densifying Encoder (GraphSAGE)** for sparse, homophilic graphs

This prevents heterophilic poisoning and semantic starvation.

------------------------------------------------------------------------

## 2. Attention-Guided Multi-Scale Diffusion

Rather than isotropic message passing, ECHO:

-   Learns edge-level attention weights\
-   Dynamically prunes cross-community noise\
-   Applies K-step high-order diffusion

This halts feature homogenization while reinforcing intra-community
structure.

------------------------------------------------------------------------

## 3. Memory-Sharded Full-Batch Contrastive Learning

ECHO maintains exact full-batch InfoNCE gradients while:

-   Dynamically chunking negative sampling tensors\
-   Enforcing GPU memory thresholds\
-   Preserving mathematical equivalence to dense objectives

This bypasses the classical O(N²) systems barrier.

------------------------------------------------------------------------

## 4. Sub-Quadratic Clustering Extraction

Instead of constructing a dense similarity matrix, ECHO:

-   Performs chunked similarity evaluation\
-   Retains only top-k degree-adaptive neighbors\
-   Constructs a sparse similarity graph\
-   Applies modularity maximization (e.g., Leiden via igraph)

Overall space complexity:

    O(|E| + N · k_max)

------------------------------------------------------------------------

# Hardware Requirements

⚠ **Important**

This research implementation is designed for **CUDA-enabled GPU
environments**.

The model uses:

-   PyTorch Automatic Mixed Precision (AMP)
-   `torch.amp.GradScaler`
-   GPU memory sharding

Requirements:

-   NVIDIA GPU
-   CUDA-compatible PyTorch build
-   Python ≥ 3.9
-   PyTorch ≥ 2.0

CPU-only execution is **not officially supported** in this research
snapshot.

------------------------------------------------------------------------

# Installation

Minimal GPU setup:

``` bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch numpy python-igraph networkx
```

Verify CUDA:

``` python
import torch
print(torch.cuda.is_available())  # should return True
```

------------------------------------------------------------------------

# Running the LFR Benchmark

Benchmark runner:

    examples/run_lfr_benchmark.py

Run:

``` bash
python examples/run_lfr_benchmark.py     --n 1000     --mu 0.1     --feat-dim 16     --epochs 50     --seed 42
```

The script:

1.  Generates an LFR graph
2.  Converts to igraph
3.  Creates synthetic node features
4.  Runs ECHO training
5.  Outputs runtime and community count

------------------------------------------------------------------------

# Repository Structure

    ECHO-GNN/
    ├── echo_gnn_v3.py
    ├── examples/
    │   └── run_lfr_benchmark.py
    ├── docs/
    │   └── ECHO.pdf
    ├── README.md
    ├── CITATION.cff
    ├── LICENSE
    └── CHANGELOG.md

------------------------------------------------------------------------

# Citation

``` bibtex
@article{ferrara2026echo,
  title={ECHO: Encoding Communities via High-order Operators},
  author={Ferrara, Emilio},
  year={2026}
}
```

------------------------------------------------------------------------

# Version

**v3.0.0 --- Research Snapshot**

------------------------------------------------------------------------

# License

Apache 2.0 License.
