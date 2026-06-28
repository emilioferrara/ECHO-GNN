# Changelog

## [1.4] - 2026 (canonical)
- `echo_gnn.py` (ECHO v1.4, full-batch research edition) is now the recommended, paper-reproducing
  implementation. Exposes `ECHO` (alias of `SelfSupervisedCommunityGNN_Alpha`).
- Added a real-world results table to the README; updated the LFR example to use `echo_gnn.py`.
- Moved the earlier experimental refactor to `legacy/echo_gnn_v3.py` (retained for provenance; its
  automatic router can mis-route dense graphs and does not reproduce the paper's numbers).
