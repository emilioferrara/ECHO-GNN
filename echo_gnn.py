"""
ECHO: Encoding Communities via High-order Operators
Version: 1.4.0 (Full-Batch Research Edition)
===================================================
A high-performance implementation of the ECHO framework optimized for 
rigorous hyperparameter grid searches on research-grade datasets.

Core Architecture:
- Full-Batch Contrastive Learning (InfoNCE)
- Attention-Guided Diffusion (High-Order Operators)
- igraph C-Core backend for native topological speed
- Memory-Sharded Negative Sampling to prevent VRAM overflow

Key engineering choice: This version utilizes the igraph C-core to handle 
symmetrization and Louvain partitioning, removing the Python/NetworkX 
overhead entirely.
"""

import numpy as np
import igraph as ig
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import warnings
from typing import Optional, Union, Tuple

# Enable A100/H100 Tensor Core optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def set_seed(seed: int):
    """Sets global seeds for deterministic research results."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SimpleGraphSAGE(nn.Module):
    r"""
    Inductive mean-pooling GraphSAGE layer.
    
    Performs neighbor aggregation and feature concatenation.
    $$h_v = \tanh(W_{self} \cdot x_v \, \Vert \, W_{neigh} \cdot \text{mean}(x_u, u \in N(v)))$$
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim, hidden_dim)
        self.W_neigh = nn.Linear(in_dim, hidden_dim)

    def forward(self, X: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        n = X.size(0)
        src, dst = indices[0], indices[1]

        # Scalable Degree Normalization
        deg = torch.zeros(n, 1, device=X.device, dtype=X.dtype).index_add_(
            0, src, torch.ones(indices.shape[1], 1, device=X.device, dtype=X.dtype)
        ) + 1e-9

        # Neighbor aggregation via index_add_ (scatter-add)
        neigh_sum = torch.zeros_like(X).index_add_(0, src, X[dst])
        h_self = self.W_self(X)
        h_neigh = self.W_neigh(neigh_sum / deg)

        return torch.tanh(torch.cat([h_self, h_neigh], dim=1))

class AttentionDiffusionGPU(nn.Module):
    """
    The 'High-Order Operator' of ECHO.
    
    Learns attention weights for edges to selectively diffuse features,
    crucial for mitigating noise in heterophilic graphs.
    """
    def __init__(self, in_dim: int, attn_hidden: int = 32):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, 1)
        )
        self.node_lin = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, S0: torch.Tensor, indices: torch.Tensor, n_steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        S = S0
        n = S.size(0)
        src, dst = indices[0], indices[1]
        raw_scores_final = None

        for _ in range(n_steps):
            if indices.shape[1] == 0: break

            # Edge attention: (src || dst) -> weight
            pair = torch.cat([S[src], S[dst]], dim=1)
            scores = self.attn_mlp(pair).squeeze(-1)
            raw_scores_final = scores

            # Stable Softmax-normalization across neighbors
            scores = torch.clamp(scores, min=-20.0, max=20.0)
            exp_scores = torch.exp(scores)
            sum_exp = torch.zeros(n, device=S.device).index_add_(0, src, exp_scores)
            attn_weights = exp_scores / (sum_exp[src] + 1e-9)

            # Selective Diffusion
            agg = torch.zeros_like(S).index_add_(0, dst, self.node_lin(S)[src] * attn_weights.unsqueeze(-1))
            S = torch.tanh(S + agg)

        # Return both embeddings and edge weights for sparsity regularization
        return S, torch.sigmoid(raw_scores_final) if raw_scores_final is not None else torch.tensor([], device=S.device)

class FullGraphPipelineEncoder(nn.Module):
    """Integrates GraphSAGE and High-Order Diffusion into a unified pipeline."""
    def __init__(self, in_dim: int, hidden_dim: int = 32, attn_hidden: int = 32, diff_steps: int = 3):
        super().__init__()
        self.sage = SimpleGraphSAGE(in_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.diffusion = AttentionDiffusionGPU(hidden_dim, attn_hidden)
        self.diff_steps = diff_steps

    def forward(self, X: torch.Tensor, adj_sparse: torch.sparse_coo_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = adj_sparse.indices().to(X.device)
        H = self.sage(X, indices)
        H_proj = torch.tanh(self.proj(H))
        return self.diffusion(H_proj, indices, self.diff_steps)

class SelfSupervisedCommunityGNN_Alpha:
    """
    The ECHO API (v1.4.0 Full-Batch).
    
    Optimized for hyperparameter search on standard GPU (24GB - 80GB VRAM).
    """
    DENSE_POS_THRESHOLD = 5000 # threshold for building O(N^2) masks

    def __init__(self, feat_dim: int = 16, sage_hidden: int = 128, 
                 attn_hidden: int = 32, diff_steps: int = 1,
                 epochs: int = 200, lr: float = 0.01, temp: float = 0.1, 
                 neg_per_node: int = 256, sparsity_penalty: float = 0.05, 
                 cluster_threshold: float = 0.55, seed: int = 42, 
                 device: Optional[str] = None, **kwargs):
        
        # Mapping legacy kwargs for grid-search compatibility
        self.feat_dim = kwargs.get('in_feats', kwargs.get('in_dim', feat_dim))
        self.sage_hidden = kwargs.get('hidden_dim', sage_hidden)
        self.epochs = kwargs.get('n_epochs', epochs)
        self.neg_per_node = kwargs.get('negatives_per_node', neg_per_node)
        
        self.attn_hidden = attn_hidden
        self.diff_steps = diff_steps
        self.lr = lr
        self.temp = temp
        self.sparsity_penalty = sparsity_penalty
        self.cluster_threshold = cluster_threshold

        set_seed(seed)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = FullGraphPipelineEncoder(self.feat_dim, self.sage_hidden, self.attn_hidden, self.diff_steps).to(self.device)

        self.labels_ = None
        self._original_G = None

    def _prepare_tensors(self, G, n, device):
        """Native igraph edge extraction and conversion to torch sparse tensors."""
        if not hasattr(G, 'ecount'): # Legacy NetworkX compatibility
            import networkx as nx
            node_list = sorted(G.nodes())
            G_ig = ig.Graph(n=len(node_list), edges=list(G.edges()))
            edges = np.array(G_ig.get_edgelist())
        else:
            edges = np.array(G.get_edgelist())

        # Symmetrize and deduplicate edges for undirected support
        edges_rev = edges[:, [1, 0]]
        edges_all = np.unique(np.vstack([edges, edges_rev]), axis=0)
        indices = torch.from_numpy(edges_all.T).long().to(device)
        adj_sparse = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1], device=device), (n, n), device=device).coalesce()
        return adj_sparse, indices

    def fit(self, G: Union[ig.Graph, 'nx.Graph'], X_ext: Optional[np.ndarray] = None, 
            verbose: bool = True, use_amp: bool = True):
        """Trains the encoder using full-batch InfoNCE contrastive loss."""
        
        # 1. Compatibility & Setup
        if not isinstance(G, ig.Graph):
            nodes = list(G.nodes())
            node_to_ig = {n: i for i, n in enumerate(nodes)}
            self._original_G = ig.Graph(n=len(nodes), edges=[(node_to_ig[u], node_to_ig[v]) for u, v in G.edges()])
        else:
            self._original_G = G

        device = torch.device(self.device)
        n = self._original_G.vcount()
        adj_sparse, indices = self._prepare_tensors(self._original_G, n, device)
        pos_neighbors = [set(neighbors) for neighbors in self._original_G.get_adjlist()]

        # 2. Embedding Initialization
        X = X_ext.astype(np.float32) if X_ext is not None else np.random.randn(n, self.feat_dim).astype(np.float32) * 0.05
        self.node_emb = nn.Embedding(n, X.shape[1]).to(device)
        self.node_emb.weight.data.copy_(torch.from_numpy(X))
        self.node_emb.weight.requires_grad = False

        # 3. Training Loop with Sharded Negative Sampling
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=1e-4)
        self._scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        SHARD_ELEM_THRESH = 200_000_000 # Prevent VRAM spikes during similarity calculation
        shard_needed = (n * self.neg_per_node * self.sage_hidden) > SHARD_ELEM_THRESH

        

        self.encoder.train()
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                S_raw, attn = self.encoder(self.node_emb.weight, adj_sparse)
                S = F.normalize(S_raw, p=2, dim=1)

                # Positive Similarity (Edges)
                src, dst = indices[0], indices[1]
                edge_sim = torch.clamp((S[src] * S[dst]).sum(dim=1) / (self.temp + 1e-12), -50.0, 50.0)
                edge_weight = (attn.squeeze() + 0.1) if attn.numel() > 0 else 1.0
                pos_sum = torch.zeros(n, device=device).index_add_(0, dst, torch.exp(edge_sim) * edge_weight)

                # Negative Similarity (Sharded for memory safety)
                neg_idx = torch.randint(0, n, (n, self.neg_per_node), device=device)
                neg_sum = torch.zeros(n, device=device)
                
                if shard_needed:
                    chunk = max(1, int(SHARD_ELEM_THRESH / (self.neg_per_node * self.sage_hidden)))
                    for start in range(0, n, chunk):
                        end = min(n, start + chunk)
                        n_sim = torch.clamp((S[start:end].unsqueeze(1) * S[neg_idx[start:end]]).sum(dim=2) / self.temp, -50, 50)
                        neg_sum[start:end] = torch.exp(n_sim).sum(dim=1)
                else:
                    n_sim = torch.clamp((S.unsqueeze(1) * S[neg_idx]).sum(dim=2) / self.temp, -50, 50)
                    neg_sum = torch.exp(n_sim).sum(dim=1)

                valid = pos_sum > 0
                loss = -torch.log(pos_sum[valid] / (pos_sum[valid] + neg_sum[valid] + 1e-12)).mean()
                if attn.numel() > 0: loss += self.sparsity_penalty * torch.mean(attn)

            self._scaler.scale(loss).backward()
            self._scaler.step(optimizer)
            self._scaler.update()

            if verbose and epoch % 10 == 0: print(f"Epoch {epoch:4d} | Loss {loss.item():.6f}")

        # 4. Clustering (igraph backend)
        self.encoder.eval()
        with torch.no_grad():
            S_final, _ = self.encoder(self.node_emb.weight, adj_sparse)
            self.labels_ = self._cluster(S_final.cpu().numpy())
        return self

    def _cluster(self, S_cpu):
        """Chunked similarity extraction and Louvain partitioning via igraph."""
        n = S_cpu.shape[0]
        S = F.normalize(torch.tensor(S_cpu, device=self.device), p=2, dim=1)
        k_values = np.clip((np.array(self._original_G.degree()) * 1.0).astype(int), 5, 30)
        max_k = int(k_values.max())
        rows, cols, vals = [], [], []

        for start in range(0, n, 2048):
            end = min(start + 2048, n)
            sim = (torch.matmul(S[start:end], S.T) + 1.0) / 2.0
            tv, ti = torch.topk(sim, k=max_k + 1, dim=1)
            tv_np, ti_np = tv.cpu().numpy(), ti.cpu().numpy()
            
            for i_local in range(end - start):
                i_global = start + i_local
                mask = tv_np[i_local, :k_values[i_global]+1] > self.cluster_threshold
                rows.extend([i_global] * mask.sum())
                cols.extend(ti_np[i_local, :k_values[i_global]+1][mask].tolist())
                vals.extend(tv_np[i_local, :k_values[i_global]+1][mask].tolist())

        G_res = ig.Graph(n=n, edges=list(zip(rows, cols)), directed=False)
        G_res.es['weight'] = vals
        # Native C-Core Louvain
        G_res.simplify(combine_edges='max')
        return np.array(G_res.community_multilevel(weights='weight').membership)

    def predict(self): return self.labels_

# Public API alias: the canonical ECHO class that reproduces the paper results.
ECHO = SelfSupervisedCommunityGNN_Alpha
