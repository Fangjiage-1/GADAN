"""
Geometry-Aware Spatial Bias Injection (GSBI) Cross-Attention Module.

Decouples semantic matching from geometric spatial logic inside Cross-Attention.

Standard cross-attention:  Attention = Softmax(Q K^T / sqrt(d)) V

GSBI extends this by adding a learnable geometric bias B_geo:
    Attention_new = Softmax(Q K^T / sqrt(d) + lambda * B_geo) V

where:
    E_vis_geo  = MLP_geo(C)            ,  C in R^{B x N x 4}
    E_text_geo = H_text W_geo          ,  H_text in R^{B x M x d_model}
    B_geo      = E_text_geo E_vis_geo^T,  B_geo in R^{B x M x N}

The effective gate is factorised as:
    lambda = lambda_base * lambda_adaptive

where lambda_base is a ZERO-INITIALIZED per-head learnable scalar (safe residual
fallback), and lambda_adaptive = sigmoid(MLP_gate([pool(H_txt); pool(H_vis)]))
is an input-dependent modulation that dynamically adjusts geometric reliance
based on the query's spatial complexity.

At initialisation lambda_base = 0, so the module is an exact residual
safe-fallback to standard cross-attention regardless of lambda_adaptive.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GSBICrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention with Geometry-Aware Spatial Bias Injection
    and Adaptive Gating.

    Inputs:
        query      : text features from LSTM           (B, M, d_model)
        key / value: visual features                   (B, N, d_model)
        vis_coords : normalized bbox coords per patch  (B, N, 4)  [xmin, ymin, xmax, ymax]
        text_mask  : optional bool mask for text pad    (B, M)      True = padded

    The geometric bias B_geo is computed from vis_coords and projected text
    features, then added to the pre-softmax attention logits scaled by an
    effective gate lambda = lambda_base * lambda_adaptive, where:
      - lambda_base    : zero-initialised per-head scalar (safe fallback)
      - lambda_adaptive: input-dependent modulation in (0,1) predicted from
                         pooled text & visual features
    """

    def __init__(self, d_model, nhead, d_geo=64, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_geo = d_geo

        # ---------- standard Q / K / V projections ----------
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # ---------- geometry encoding ----------
        # 2-layer MLP:  4 (bbox coords)  -> d_geo  -> d_geo
        self.vis_geo_mlp = nn.Sequential(
            nn.Linear(4, d_geo),
            nn.GELU(),
            nn.Linear(d_geo, d_geo),
        )

        # Linear map from text hidden space to geometry space
        self.text_geo_proj = nn.Linear(d_model, d_geo)

        # ---------- learnable base gate (ZERO-INITIALIZED) ----------
        # per-head scalar, broadcasts over (M, N)
        self.lambda_base = nn.Parameter(torch.zeros(1, nhead, 1, 1))

        # ---------- adaptive gate (input-dependent modulation) ----------
        # pools text & visual features → predicts per-sample per-head gating
        # final bias init = 2.0 → sigmoid(2.0) ≈ 0.88 (near-identity modulation
        # at start, i.e. the adaptive gate does not suppress the base gate by default)
        self.adaptive_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, nhead),
        )

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Match nn.MultiheadAttention's effective initialization:
        # nn.MHA uses a fused in_proj_weight of shape (3*d_model, d_model).
        # xavier_uniform_ on that fused matrix sees fan_in=d_model, fan_out=3*d_model.
        # Since we split into three separate Linear(d_model, d_model) layers,
        # we must compute the gain w.r.t. the fused fan dimensions so the
        # initial weight magnitudes are identical to nn.MHA.
        fused_gain = math.sqrt(2.0 / (3.0 * self.d_model + self.d_model))
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, std=fused_gain)
            nn.init.constant_(proj.bias, 0.0)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

        # geometry MLP
        for module in self.vis_geo_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

        nn.init.xavier_uniform_(self.text_geo_proj.weight)
        nn.init.constant_(self.text_geo_proj.bias, 0.0)

        # adaptive gate MLP
        nn.init.xavier_uniform_(self.adaptive_gate[0].weight)
        nn.init.constant_(self.adaptive_gate[0].bias, 0.0)
        nn.init.xavier_uniform_(self.adaptive_gate[2].weight)
        # bias = 2.0 → sigmoid(2.0) ≈ 0.88, near-identity modulation at start
        nn.init.constant_(self.adaptive_gate[2].bias, 2.0)

        # lambda_base is already zero from __init__ parameter creation

    def forward(self, query, key, value, vis_coords, text_mask=None, key_padding_mask=None):
        """
        Args:
            query:             (B, M, d_model)  — text features
            key:               (B, N, d_model)  — visual features
            value:             (B, N, d_model)  — visual features
            vis_coords:        (B, N, 4)        — normalized bbox coords [xmin, ymin, xmax, ymax]
            text_mask:         (B, M) | None    — True = padded text position
            key_padding_mask:  (B, N) | None    — True = padded visual position

        Returns:
            output: (B, M, d_model)
        """
        B, M, _d = query.shape
        B_q, N, _d_k = key.shape
        assert _d == _d_k == self.d_model
        assert vis_coords.shape == (B_q, N, 4)

        # ---- 1. linear projections ----
        Q = self.q_proj(query)  # (B, M, d_model)
        K = self.k_proj(key)    # (B, N, d_model)
        V = self.v_proj(value)  # (B, N, d_model)

        # ---- 2. reshape into multi-head  (B, nhead, seq, head_dim) ----
        Q = Q.view(B, M, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, M,  hd)
        K = K.view(B, N, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N,  hd)
        V = V.view(B, N, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N,  hd)

        # ---- 3. semantic attention scores ----
        scale = self.head_dim ** 0.5
        scores = torch.einsum('bhmd,bhnd->bhmn', Q, K) / scale  # (B, nhead, M, N)

        # ---- 4. geometric spatial bias B_geo ----
        # E_vis_geo  = MLP_geo(vis_coords)            (B, N, d_geo)
        # E_text_geo = text_geo_proj(query)            (B, M, d_geo)
        # B_geo      = E_text_geo @ E_vis_geo^T        (B, M, N) -> unsqueeze heads
        E_vis_geo = self.vis_geo_mlp(vis_coords)                       # (B, N, d_geo)
        E_text_geo = self.text_geo_proj(query)                          # (B, M, d_geo)
        B_geo = torch.einsum('bmd,bnd->bmn', E_text_geo, E_vis_geo)    # (B, M, N)
        B_geo = B_geo.unsqueeze(1)                                      # (B, 1, M, N)

        # ---- 5. adaptive gate: lambda = lambda_base * lambda_adaptive ----
        # (a) pool text features: masked mean over non-padded tokens
        if text_mask is not None:
            # invert: True=padded → False=valid
            valid_txt = ~text_mask  # (B, M)
            txt_pooled = (query * valid_txt.unsqueeze(-1).float()).sum(dim=1) \
                         / valid_txt.sum(dim=1, keepdim=True).clamp(min=1).float()  # (B, d_model)
        else:
            txt_pooled = query.mean(dim=1)  # (B, d_model)

        # (b) pool visual features: masked mean over non-padded regions
        if key_padding_mask is not None:
            valid_vis = ~key_padding_mask  # (B, N)
            vis_pooled = (key * valid_vis.unsqueeze(-1).float()).sum(dim=1) \
                         / valid_vis.sum(dim=1, keepdim=True).clamp(min=1).float()  # (B, d_model)
        else:
            vis_pooled = key.mean(dim=1)  # (B, d_model)

        # (c) predict per-sample, per-head modulation
        gate_input = torch.cat([txt_pooled, vis_pooled], dim=-1)       # (B, 2*d_model)
        lambda_adaptive = torch.sigmoid(self.adaptive_gate(gate_input))  # (B, nhead)
        lambda_adaptive = lambda_adaptive.view(B, self.nhead, 1, 1)      # (B, nhead, 1, 1)

        # (d) effective gate = base (zero-init) × adaptive (data-dependent)
        lambda_effective = self.lambda_base * lambda_adaptive           # (B, nhead, 1, 1)

        # ---- 6. inject bias with effective gate ----
        scores = scores + lambda_effective * B_geo

        # ---- 7. mask padded positions (set to -inf before softmax) ----
        if text_mask is not None:
            # (B, M) → (B, 1, M, 1) broadcast over heads & key dim
            scores = scores.masked_fill(
                text_mask.unsqueeze(1).unsqueeze(-1),
                float('-inf'),
            )
        if key_padding_mask is not None:
            # (B, N) → (B, 1, 1, N) broadcast over heads & query dim
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )

        # ---- 8. softmax + dropout  (NaN-safe: all -inf rows → 0) ----
        attn_weights = F.softmax(scores, dim=-1)  # (B, nhead, M, N)
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)
        attn_weights = self.dropout(attn_weights)

        # ---- 9. weighted aggregation ----
        output = torch.einsum('bhmn,bhnd->bhmd', attn_weights, V)  # (B, nhead, M, head_dim)

        # ---- 10. merge heads + output projection ----
        output = output.transpose(1, 2).contiguous().view(B, M, self.d_model)  # (B, M, d_model)
        output = self.out_proj(output)

        return output


def generate_vis_coords(h, w, t, b, device):
    """
    Create normalised bounding-box coordinates for a feature-map grid.

    Each spatial cell (row, col) in an  h x w  feature map is mapped to:
        [col/w,  row/h,  (col+1)/w,  (row+1)/h]
    These are repeated across *t* temporal frames and *b* batch entries.

    Args:
        h, w   : spatial resolution of one feature-map frame
        t      : number of temporal frames
        b      : batch size (excluding temporal)
        device : torch device

    Returns:
        coords : (b, t*h*w, 4)   float32, values in [0, 1]
    """
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
    )
    # (h, w, 4)
    coords_1f = torch.stack([
        grid_x / w,
        grid_y / h,
        (grid_x + 1.0) / w,
        (grid_y + 1.0) / h,
    ], dim=-1)
    coords_1f = coords_1f.reshape(-1, 4)  # (h*w, 4)

    # repeat over t frames  →  (t*h*w, 4)
    coords = coords_1f.unsqueeze(0).expand(t, -1, -1).reshape(t * h * w, 4)

    # expand over batch  →  (b, t*h*w, 4)
    coords = coords.unsqueeze(0).expand(b, -1, -1)

    return coords
