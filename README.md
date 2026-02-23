# ECHO-GNN — Encoding Communities via High-order Operators

**Research snapshot v3.0.0** — Original implementation file: `echo_gnn_v3.py` (kept unchanged).

![CI](https://github.com/emilioferrara/ECHO-GNN/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Release](https://img.shields.io/badge/release-v3.0.0-brightgreen)

## Summary

ECHO is a contrastive, topology-aware method for scalable attributed community detection. This repository contains the original research implementation file `echo_gnn_v3.py` (research snapshot). The implementation and paper are intended for research use and reproducibility.

## Quick start

1. Review the model file: `echo_gnn_v3.py` — **do not edit** if you want the pristine research snapshot.
2. To reproduce experiments, add `docs/paper.pdf` (the paper) and create an `examples/` script using the API in the file.
3. Use the included `setup_repo.sh` (or your own workflow) to create tags/releases and push to GitHub.

## Citation

Please cite this work as:

```bibtex
@article{ferrara2026echo,
  title={ECHO: Encoding Communities via High-order Operators},
  author={Ferrara, Emilio},
  year={2026}
}
