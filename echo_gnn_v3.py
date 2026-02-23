"""
ECHO: Encoding Communities via High-order Operators
Version: 3.0.0 (Topology-Aware AutoECHO Edition)
===================================================
Features built-in unsupervised topology analysis to automatically route
graphs through either a Densifying Encoder (GraphSAGE) or an Isolating 
Encoder (Pure Attention MLP) to prevent heterophilic poisoning.
"""
import numpy as np
import igraph as ig
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional, Union, Tuple

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. AUTO-ROUTER
# ==========================================
class TopologyAnalyzer:
    """Computes unsupervised heuristics to automatically select the optimal encoder."""
    def __init__(self, sparsity_thresh=0.85, assortativity_thresh=0.1, density_thresh=20.0):
        self.sparsity_thresh = sparsity_thresh
        self.assortativity_thresh = assortativity_thresh
        self.density_thresh = density_thresh

    def analyze(self, X: torch.Tensor, indices: torch.Tensor, num_nodes: int) -> str:
        print("\n--- AutoECHO: Analyzing Network Topology ---")
        
        # 1. Feature Sparsity
        sparsity = (X == 0).float().mean().item()
        
        # 2. Structural Density
        num_edges = indices.shape[1]
        avg_degree = num_edges / max(1, num_nodes)
        
        # 3. Semantic Assortativity (Cosine Sim across edges)
        max_samples = min(num_edges, 50000)
        if max_samples > 0:
            idx = torch.randperm(num_edges)[:max_samples]
            src, dst = indices[0][idx], indices[1][idx]
            sim = F.cosine_similarity(X[src], X[dst], dim=1)
            assortativity = sim.mean().item()
        else:
            assortativity = 0.0

        print(f"  -> Feature Sparsity:       {sparsity:.2%} (Threshold: {self.sparsity_thresh:.2%})")
        print(f"  -> Semantic Assortativity: {assortativity:.4f} (Threshold: {self.assortativity_thresh:.4f})")
        print(f"  -> Average Degree:         {avg_degree:.1f} (Threshold: {self.density_thresh:.1f})")

        # RULE 1: Over-smoothing Prevention (Dense or Heterophilic)
        if avg_degree > self.density_thresh or assortativity < self.assortativity_thresh:
            print(">>> Auto-Decision: ISOLATING ENCODER (Pure Attention MLP) <<<")
            print("    Reason: High density or low semantic assortativity detected. Preventing Heterophilic Poisoning.\n")
            return "MLP"
            
        # RULE 2: Signal Densification (Sparse Bag-of-Words)
        elif sparsity > self.sparsity_thresh:
            print(">>> Auto-Decision: DENSIFYING ENCODER (GraphSAGE) <<<")
            print("    Reason: High feature sparsity and stable homophily detected. Densifying signal prior to diffusion.\n")
            return "SAGE"
            
        # DEFAULT: Well-behaved continuous features
        else:
            print(">>> Auto-Decision: ISOLATING ENCODER (Pure Attention MLP) <<<")
            print("    Reason: Well-behaved continuous features. Relying on native High-Order Diffusion.\n")
            return "MLP"

# ==========================================
# 2. NEURAL ARCHITECTURE
# ==========================================
class SimpleGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim, hidden_dim)
        self.W_neigh = nn.Linear(in_dim, hidden_dim)

    def forward(self, X: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        n = X.size(0)
        src, dst = indices[0], indices[1]
        deg = torch.zeros(n, 1, device=X.device, dtype=X.dtype).index_add_(
            0, src, torch.ones(indices.shape[1], 1, device=X.device, dtype=X.dtype)
        ) + 1e-9
        neigh_sum = torch.zeros_like(X).index_add_(0, src, X[dst])
        h_self = self.W_self(X)
        h_neigh = self.W_neigh(neigh_sum / deg)
        return torch.tanh(torch.cat([h_self, h_neigh], dim=1))

class AttentionDiffusionGPU(nn.Module):
    def __init__(self, in_dim: int, attn_hidden: int = 32):
        super().__init__()
        self.attn_mlp = nn.Sequential(nn.Linear(2 * in_dim, attn_hidden), nn.LeakyReLU(0.2), nn.Linear(attn_hidden, 1))
        self.node_lin = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, S0: torch.Tensor, indices: torch.Tensor, n_steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        S = S0; n = S.size(0); src, dst = indices[0], indices[1]; raw_scores = None
        if n_steps == 0 or indices.shape[1] == 0:
            return S, torch.tensor([], device=S.device, dtype=S.dtype)
            
        for _ in range(n_steps):
            pair = torch.cat([S[src], S[dst]], dim=1)
            raw_scores = self.attn_mlp(pair).squeeze(-1)
            scores_fp32 = torch.clamp(raw_scores.float(), min=-20.0, max=10.0)
            exp_scores = torch.exp(scores_fp32)
            sum_exp = torch.zeros(n, device=S.device, dtype=torch.float32).index_add_(0, src, exp_scores)
            attn_weights = (exp_scores / (sum_exp[src] + 1e-9)).to(S.dtype)
            agg = torch.zeros_like(S).index_add_(0, dst, self.node_lin(S)[src] * attn_weights.unsqueeze(-1))
            S = torch.tanh(S + agg)
        
        attn_out = attn_weights if raw_scores is not None else torch.tensor([], device=S.device, dtype=S.dtype)
        return S, attn_out

class ECHOEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, diff_steps: int = 1, encoder_type: str = 'MLP'):
        super().__init__()
        self.encoder_type = encoder_type
        
        # Dual-Path Modularity
        if self.encoder_type == 'SAGE':
            self.sage = SimpleGraphSAGE(in_dim, hidden_dim)
            self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.lin = nn.Linear(in_dim, hidden_dim)
            self.out_lin = nn.Linear(hidden_dim, hidden_dim)
            
        self.conv = AttentionDiffusionGPU(hidden_dim, 32)
        self.diff_steps = diff_steps

    def forward(self, X: torch.Tensor, adj_sparse: torch.sparse_coo_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = adj_sparse.indices().to(X.device)
        
        if self.encoder_type == 'SAGE':
            H = self.sage(X, indices)
            h = torch.tanh(self.proj(H))
            z, alpha = self.conv(h, indices, self.diff_steps)
            return z, alpha
        else:
            h = F.relu(self.lin(X))
            z, alpha = self.conv(h, indices, self.diff_steps)
            return self.out_lin(z), alpha

# ==========================================
# 3. MAIN API
# ==========================================
class ECHO:
    def __init__(self, feat_dim: int = 16, hidden_dim: int = 128, 
                 attn_hidden: int = 32, diff_steps: int = 1,
                 epochs: int = 300, lr: float = 0.001, temp: float = 0.1, 
                 neg_per_node: int = 256, sparsity_penalty: float = 0.001, 
                 cluster_threshold: float = 0.55, batch_size: Optional[int] = None, 
                 patience: int = 20, seed: int = 42, device: Optional[str] = None):
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.diff_steps = diff_steps
        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.neg_per_node = neg_per_node
        self.sparsity_penalty = sparsity_penalty
        self.cluster_threshold = cluster_threshold
        self.batch_size = batch_size
        self.patience = patience

        set_seed(seed)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialized dynamically in fit()
        self.encoder = None 
        self.optimizer = None
        self.encoder_type_ = None 
        
        self.labels_ = None
        self._original_G = None

    def _prepare_data(self, G, X_ext):
        self._original_G = G
        n = self._original_G.vcount()
        edges = np.array(self._original_G.get_edgelist())
        edges_all = np.unique(np.vstack([edges, edges[:, [1, 0]]]), axis=0)
        indices = torch.from_numpy(edges_all.T).long().to(self.device)
        adj_sparse = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1], device=self.device), (n, n), device=self.device).coalesce()
        
        X = X_ext.astype(np.float32) if X_ext is not None else np.random.randn(n, self.feat_dim).astype(np.float32) * 0.05
        self.node_emb = nn.Embedding(n, X.shape[1]).to(self.device)
        self.node_emb.weight.data.copy_(torch.from_numpy(X))
        self.node_emb.weight.requires_grad = False
        return adj_sparse, indices, n

    def fit(self, G, X_ext: Optional[np.ndarray] = None, verbose: bool = True, use_amp: bool = True):
        adj_sparse, indices, n = self._prepare_data(G, X_ext)
        
        # --- TOPOLOGY-AWARE ROUTING ---
        analyzer = TopologyAnalyzer()
        self.encoder_type_ = analyzer.analyze(self.node_emb.weight.data, indices, n)
        
        # Instantiate correct encoder based on analysis
        self.encoder = ECHOEncoder(self.feat_dim, self.hidden_dim, self.diff_steps, self.encoder_type_).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=5e-4)
        self._scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        self._train_full_batch(adj_sparse, indices, n, use_amp, verbose)

        self.encoder.eval()
        with torch.no_grad():
            S_final, _ = self.encoder(self.node_emb.weight, adj_sparse)
            self.labels_ = self._cluster(S_final.cpu().numpy())
        return self

    def _train_full_batch(self, adj_sparse, indices, n, use_amp, verbose):
        SHARD_ELEM_THRESH = 200_000_000
        shard_needed = (n * self.neg_per_node * self.hidden_dim) > SHARD_ELEM_THRESH
        self.encoder.train()
        best_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                z, alpha = self.encoder(self.node_emb.weight, adj_sparse)
                z = F.normalize(z, p=2, dim=1)

                src, dst = indices[0], indices[1]
                edge_sim = torch.clamp((z[src] * z[dst]).sum(dim=1) / self.temp, -50.0, 50.0).float()
                edge_weight = alpha.squeeze().float() if alpha.numel() > 0 else 1.0
                
                pos_signal = torch.exp(edge_sim) * edge_weight
                pos_sum = torch.zeros(n, device=self.device, dtype=torch.float32).index_add_(0, dst, pos_signal)

                neg_idx = torch.randint(0, n, (n, self.neg_per_node), device=self.device)
                neg_sum = torch.zeros(n, device=self.device, dtype=torch.float32)
                
                if shard_needed:
                    chunk = max(1, int(SHARD_ELEM_THRESH / (self.neg_per_node * self.hidden_dim)))
                    for start in range(0, n, chunk):
                        end = min(n, start + chunk)
                        n_sim = torch.clamp((z[start:end].unsqueeze(1) * z[neg_idx[start:end]]).sum(dim=2) / self.temp, -50.0, 50.0).float()
                        neg_sum[start:end] = torch.exp(n_sim).sum(dim=1)
                else:
                    n_sim = torch.clamp((z.unsqueeze(1) * z[neg_idx]).sum(dim=2) / self.temp, -50.0, 50.0).float()
                    neg_sum = torch.exp(n_sim).sum(dim=1)

                valid = pos_sum > 0
                loss = -torch.log(pos_sum[valid] / (pos_sum[valid] + neg_sum[valid] + 1e-12)).mean()
                if alpha.numel() > 0:
                    loss += self.sparsity_penalty * torch.mean(torch.abs(alpha))

            self._scaler.scale(loss).backward()
            self._scaler.step(self.optimizer)
            self._scaler.update()

            current_loss = loss.item()
            if current_loss < best_loss - 1e-4:
                best_loss = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= self.patience:
                if verbose: print(f"  -> Early stopping triggered at epoch {epoch} (Loss: {best_loss:.4f})")
                break

    def _cluster(self, S_cpu: np.ndarray) -> np.ndarray:
        n = S_cpu.shape[0]
        S_full = F.normalize(torch.tensor(S_cpu, device=self.device, dtype=torch.float32), p=2, dim=1)
        k_values = np.clip((np.array(self._original_G.degree()) * 1.0).astype(int), 5, 30)
        max_k, chunk_size = int(k_values.max()), 2048
        rows, cols, vals = [], [], []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            S_chunk = S_full[start:end]
            sim_chunk = (torch.matmul(S_chunk, S_full.T) + 1.0) / 2.0
            tv, ti = torch.topk(sim_chunk, k=max_k + 1, dim=1)
            tv_np, ti_np = tv.cpu().numpy(), ti.cpu().numpy()
            del sim_chunk, tv, ti
            torch.cuda.empty_cache()
            
            for i_local in range(end - start):
                i_global = start + i_local
                mask = tv_np[i_local, :k_values[i_global]+1] > self.cluster_threshold
                rows.extend([i_global] * mask.sum())
                cols.extend(ti_np[i_local, :k_values[i_global]+1][mask].tolist())
                vals.extend(tv_np[i_local, :k_values[i_global]+1][mask].tolist())

        G_res = ig.Graph(n=n, edges=list(zip(rows, cols)), directed=False)
        G_res.es['weight'] = vals
        G_res.simplify(combine_edges='max')
        return np.array(G_res.community_multilevel(weights='weight').membership)

    def predict(self) -> np.ndarray:
        return self.labels_
