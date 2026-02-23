# ECHO: Encoding Communities via High-Order Operators

**Research Implementation (v3.0.0)**\
Author: Emilio Ferrara\
Year: 2026

This repository contains the original research implementation of
**ECHO**, a scalable, self-supervised framework for attributed community
detection in large networks.

The implementation is preserved in its original research form as:

    echo_gnn_v3.py

------------------------------------------------------------------------

## Abstract

Community detection in attributed networks faces a dual challenge:

1.  **Semantic Wall** --- Feature over-smoothing and heterophilic
    poisoning in Graph Neural Networks.
2.  **Systems Wall** --- Quadratic memory constraints (O(N²)) in
    contrastive and clustering pipelines.

ECHO resolves both by introducing:

-   A **Topology-Aware Router** that dynamically selects inductive bias.
-   **Attention-Guided Multi-Scale Diffusion** to prevent semantic
    collapse.
-   A **Memory-Sharded Full-Batch Contrastive Objective**.
-   A **Chunked O(N·K) Similarity Extraction** method that eliminates
    O(N²) clustering bottlenecks.

The result is a framework capable of scaling to **million-node
attributed networks** on a single GPU while preserving full-batch
gradient fidelity.

------------------------------------------------------------------------

## Core Contributions

### 1. Topology-Aware Routing

Before training begins, ECHO evaluates:

-   Feature sparsity\
-   Structural density\
-   Semantic assortativity

Based on these unsupervised heuristics, the model automatically routes
the graph through:

-   **Isolating Encoder (MLP)** for dense / heterophilic graphs\
-   **Densifying Encoder (GraphSAGE)** for sparse / homophilic graphs

This prevents heterophilic poisoning and semantic starvation.

------------------------------------------------------------------------

### 2. Attention-Guided Multi-Scale Diffusion

Rather than isotropic message passing, ECHO:

-   Learns edge-level attention\
-   Dynamically prunes noisy cross-community edges\
-   Applies K-step high-order diffusion

This halts over-smoothing while allowing intra-community reinforcement.

------------------------------------------------------------------------

### 3. Memory-Sharded Full-Batch Contrastive Learning

ECHO preserves exact full-batch InfoNCE gradients while:

-   Dynamically chunking negative sampling tensors\
-   Enforcing VRAM-safe memory thresholds\
-   Maintaining mathematical equivalence to full dense objectives

This bypasses the traditional O(N²) systems barrier.

------------------------------------------------------------------------

### 4. Sub-Quadratic Clustering Extraction

Instead of constructing a dense similarity matrix, ECHO:

-   Performs chunked similarity evaluation\
-   Retains only top-k degree-adaptive neighbors\
-   Constructs a sparse similarity graph\
-   Applies modularity maximization (e.g., Leiden / igraph)

Overall memory complexity becomes:

    O(|E| + N · k_max)

------------------------------------------------------------------------

## Scaling Characteristics

ECHO was designed to operate on:

-   Synthetic LFR benchmarks up to 1M nodes\
-   Real-world social networks \>1.6M nodes\
-   Graphs with tens of millions of edges

While traditional self-supervised GNNs fail beyond \~50k nodes due to
dense similarity matrices, ECHO maintains sub-quadratic growth.

------------------------------------------------------------------------

## Repository Structure

    ECHO-GNN/
    ├── echo_gnn_v3.py
    ├── docs/
    │   └── ECHO.pdf
    ├── README.md
    ├── CITATION.cff
    ├── LICENSE
    └── CHANGELOG.md

------------------------------------------------------------------------

## Citation

``` bibtex
@article{ferrara2026echo,
  title={ECHO: Encoding Communities via High-order Operators},
  author={Ferrara, Emilio},
  year={2026}
}
```

------------------------------------------------------------------------

## License

Apache 2.0 License.
