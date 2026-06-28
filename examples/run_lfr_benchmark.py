#!/usr/bin/env python3
"""
Run a simple LFR benchmark for the research snapshot model.

Usage:
    python examples/run_lfr_benchmark.py --n 1000 --mu 0.1 --feat-dim 16 --seed 42

The script will:
 - generate an LFR graph (networkx)
 - convert to igraph
 - create random node features
 - attempt to import and run ECHO (first tries `from echo import ECHO`, then falls back
   to loading `echo_gnn_v3.py` dynamically and finding an ECHO class or Trainer)
 - measure runtime and print a short report
"""
import argparse
import time
import sys
import os
import importlib
import importlib.util
import types
import numpy as np

try:
    import igraph as ig
except Exception as e:
    print("ERROR: python-igraph is required (pip install python-igraph).", e)
    raise SystemExit(1)

try:
    import networkx as nx
except Exception as e:
    print("ERROR: networkx is required (pip install networkx).", e)
    raise SystemExit(1)


def gen_lfr(n=1000, tau1=3.0, tau2=1.5, mu=0.1, average_degree=10, min_community=20, seed=42):
    """
    Generate an LFR benchmark graph via networkx (fallback if igraph has no generator).
    Returns: networkx Graph with node attribute 'community' (set of nodes) if available.
    """
    # networkx LFR can be a bit fragile for small graphs; adjust parameters sensibly
    print(f"Generating LFR-like graph: n={n}, mu={mu}, seed={seed}")
    G_nx = nx.generators.community.LFR_benchmark_graph(
        n, tau1, tau2, mu,
        average_degree=average_degree,
        min_community=min_community,
        seed=seed
    )
    # convert to simple undirected Graph with integer node labels 0..n-1
    mapping = {node: i for i, node in enumerate(G_nx.nodes())}
    G_nx = nx.relabel_nodes(G_nx, mapping)
    return G_nx


def nx_to_igraph(G_nx):
    """Convert a NetworkX graph to python-igraph Graph (undirected)."""
    edges = list(G_nx.edges())
    G_ig = ig.Graph(n=G_nx.number_of_nodes(), edges=edges, directed=False)
    return G_ig


def load_echo_class(repo_root):
    """
    Attempts to obtain a class named ECHO (or Trainer) from available code paths.
    1) Try standard import: from echo import ECHO
    2) Fallback: dynamically load echo_gnn_v3.py and introspect for ECHO/Trainer.
    Returns: (echo_class, source_desc)
    """
    # 1) Try installed package
    try:
        spec = importlib.util.find_spec("echo")
        if spec is not None:
            mod = importlib.import_module("echo")
            if hasattr(mod, "ECHO"):
                return mod.ECHO, "installed package 'echo'"
            # maybe api module
            try:
                ap = importlib.import_module("echo.api")
                if hasattr(ap, "ECHO"):
                    return ap.ECHO, "installed package 'echo.api'"
            except Exception:
                pass
    except Exception:
        pass

    # 2) Try to import echo_gnn_v3.py from repo root
    candidate = os.path.join(repo_root, "echo_gnn_v3.py")
    if os.path.exists(candidate):
        try:
            name = "echo_gnn_v3_local"
            spec = importlib.util.spec_from_file_location(name, candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            # prefer ECHO
            if hasattr(module, "ECHO"):
                return getattr(module, "ECHO"), f"local file {candidate} (ECHO)"
            # check for Trainer class
            if hasattr(module, "Trainer"):
                return getattr(module, "Trainer"), f"local file {candidate} (Trainer)"
            # else if the module defines a fit(G, X) function, we will return module itself
            return module, f"local file {candidate} (module fallback)"
        except Exception as e:
            print("Failed to load echo_gnn_v3.py dynamically:", e)
            raise
    else:
        raise FileNotFoundError("Could not find 'echo_gnn_v3.py' in repo root.")


def run_benchmark(n=1000, mu=0.1, feat_dim=16, seed=42, epochs=10, verbose=True):
    repo_root = os.getcwd()
    # 1) generate LFR graph
    G_nx = gen_lfr(n=n, mu=mu, seed=seed)
    G_ig = nx_to_igraph(G_nx)

    # 2) build random features (user may replace with real features)
    rng = np.random.RandomState(seed)
    X = rng.randn(G_ig.vcount(), feat_dim).astype(float)

    # 3) load ECHO implementation
    EchoClassOrModule, source = load_echo_class(repo_root)
    print(f"Loaded model from: {source}")

    # 4) instantiate / run depending on available API
    start_t = time.time()
    model_instance = None
    labels = None

    # Case A: a class named ECHO with fit/predict API
    if isinstance(EchoClassOrModule, type):
        try:
            # try to instantiate with reasonable defaults
            model_instance = EchoClassOrModule(feat_dim=feat_dim, hidden_dim=64, diff_steps=1, epochs=epochs)
            print("Instantiated ECHO class; calling fit(G, X)...")
            model_instance.fit(G_ig, X, verbose=verbose)
            # prefer predict() if available
            if hasattr(model_instance, "predict"):
                labels = model_instance.predict()
        except Exception as e:
            print("Error when instantiating or running class:", e)
            raise

    # Case B: module fallback (module may expose a function like fit_model or run)
    elif isinstance(EchoClassOrModule, types.ModuleType):
        mod = EchoClassOrModule
        # common patterns
        if hasattr(mod, "fit") and hasattr(mod, "predict"):
            try:
                print("Using module-level fit/predict")
                mod.fit(G_ig, X)
                labels = mod.predict()
            except Exception as e:
                print("Module-level fit/predict failed:", e)
                raise
        elif hasattr(mod, "main"):
            print("Calling module.main()")
            mod.main()
            labels = None
        else:
            raise RuntimeError("Loaded module does not provide a usable fit/predict API.")
    else:
        raise RuntimeError("Unsupported object returned from loader.")

    duration = time.time() - start_t

    # 5) report
    if labels is None:
        print("Model ran but did not return labels via predict(); check model API.")
    else:
        labels = np.asarray(labels)
        print(f"Labels type: {labels.dtype}, length: {labels.shape}")
        unique = np.unique(labels)
        print(f"Found {len(unique)} communities (unique labels).")

    print(f"Benchmark runtime: {duration:.2f}s (n={n}, feat_dim={feat_dim})")

    return {"n": n, "mu": mu, "feat_dim": feat_dim, "time_s": duration, "n_labels": int(len(unique)) if labels is not None else None}


def parse_args():
    p = argparse.ArgumentParser(description="Run a simple LFR benchmark for ECHO model")
    p.add_argument("--n", type=int, default=1000, help="number of nodes")
    p.add_argument("--mu", type=float, default=0.1, help="mixing parameter for LFR")
    p.add_argument("--feat-dim", type=int, default=16, help="feature dimensionality")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--epochs", type=int, default=50, help="epochs passed to model (if supported)")
    p.add_argument("--verbose", action="store_true", help="print verbose training output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Running LFR benchmark with args:", args)
    res = run_benchmark(n=args.n, mu=args.mu, feat_dim=args.feat_dim, seed=args.seed, epochs=args.epochs, verbose=args.verbose)
    print("Result summary:", res)