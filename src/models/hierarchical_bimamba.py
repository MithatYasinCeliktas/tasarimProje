"""
FAZ 4 & FAZ 5: 57M Hücre Ön-Eğitimli SC-MAMBA2 + Çift Yönlü Gated BiMamba +
4 Aşamalı Hiyerarşik Karar Ağacı & Örüntü Prototip Bankası (`hierarchical_bimamba.py`)
--------------------------------------------------------------------------------------
Bu modül:
1. 57 milyon tek hücre verisiyle eğitilmiş `SC-MAMBA2` (`best_model.pt`) ağırlıklarını yükler.
2. Standart paylaşımlı `mambaformer` katmanlarını, yönlü asimetriyi (5'->3' ve 3'->5' kromozomal
   gen komşuluğu farklarını) yakalayan `GatedBiMambaAdapter` ve Yama-Tabanlı (Patch-Based)
   `BiMambaPatchMixer` ile güçlendirir.
3. FAZ 2'de çıkarılan 16 bilimsel biyolojik imzayı (`CSC`, `CytoTRACE Entropisi`, `Kromozomal CNV`,
   `EMT`, `Proliferasyon`, `Viral İnterferon`, `MS Nöro-İnflamasyon`, `TAM SPP1/C1QC`) ve
   Modality bilgisini (`Single-Cell` vs `Bulk RNA-seq`) hücre temsiliyle birleştirir.
4. `PatternPrototypeBank` ile her hastalık ve kanser varyantı için öğrenilebilir prototip merkezleri
   ($\mu_k$) oluşturur ve kosinüs benzerliğiyle Örüntü Uyum Matrisini üretir.
5. Birbirini koşullandıran 6 Hiyerarşik Teşhis Başlığı (`Hücre Tipi` -> `Sağlıklı/Hastalıklı` ->
   `Hastalık Örüntüsü` -> `Kanser mi Değil mi?` -> `Kanser Varyantı` -> `Kanser Kök Hücresi / CSC`)
   ile tek ileri yayılımda (forward pass) tam klinik teşhis ve XAI gradyan atfı üretir.
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2
    HAS_MAMBA_CUDA = True
except ImportError:
    HAS_MAMBA_CUDA = False


class FallbackBiSSMBlock(nn.Module):
    """
    Tensor-Core accelerated Selective State-Space & Conv1D Mixer block
    for machines without `mamba_ssm` CUDA kernels.
    """
    def __init__(self, d_model: int, d_conv: int = 3, expand: int = 1):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=1)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        x = self.norm(x)
        x_in, z = self.in_proj(x).chunk(2, dim=-1)
        gate = F.silu(z)
        xc = F.silu(self.conv(x_in.transpose(1, 2))[:, :, : x.size(1)]).transpose(1, 2)
        return residual + self.out_proj(gate * xc)


class GatedBiMambaLayer(nn.Module):
    """
    True Bidirectional Mamba-2 Block with Asymmetric Directional Gating:
    Processes sequence forward (5'->3' chromosomal order) and backward (3'->5'),
    capturing both symmetric context and directional regulatory / CNV boundary diffs (H_fwd - H_bwd).
    """
    def __init__(self, d_model: int = 256, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA_CUDA:
            self.mamba_fwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_bwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba_fwd = FallbackBiSSMBlock(d_model=d_model, d_conv=3, expand=1)
            self.mamba_bwd = FallbackBiSSMBlock(d_model=d_model, d_conv=3, expand=1)

        # Directional Gate + Asymmetric Interaction Projection
        self.dir_gate = nn.Linear(d_model * 2, d_model)
        self.asym_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )
        # Zero-init last linear of asym_proj for stable residual start
        nn.init.zeros_(self.asym_proj[-1].weight)
        nn.init.zeros_(self.asym_proj[-1].bias)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        h = self.norm(x)
        h_fwd = self.mamba_fwd(h)
        h_bwd = self.mamba_bwd(h.flip([1])).flip([1])

        gate = torch.sigmoid(self.dir_gate(torch.cat([h_fwd, h_bwd], dim=-1)))
        h_asym = self.asym_proj(h_fwd - h_bwd)
        h_bi = gate * h_fwd + (1.0 - gate) * h_bwd + h_asym

        x = x + h_bi
        x = x + self.mlp(self.norm2(x))
        # Per-token directional importance score for XAI Heatmap attribution
        token_salience = (h_fwd.norm(dim=-1) + h_bwd.norm(dim=-1)) * 0.5
        return x, token_salience


class PatternPrototypeBank(nn.Module):
    """
    FAZ 5 (Yöntem 2): Learnable Archetype Prototype Bank.
    Maintains L2-normalized prototype centroids for:
    - 14 Canonical Cell Types
    - 4 Disease Categories (Healthy, COVID-19, MS, Cancer)
    - 12 Cancer Variants (Non-Cancer + 11 Cancer Variants)
    Returns cosine similarity matrices in [-1, 1] and temperature-scaled logits for Pattern Heatmaps.
    """
    def __init__(self, d_emb: int = 256, n_celltypes: int = 14, n_diseases: int = 4, n_cancers: int = 12):
        super().__init__()
        self.cell_prototypes = nn.Parameter(torch.randn(n_celltypes, d_emb) * 0.05)
        self.disease_prototypes = nn.Parameter(torch.randn(n_diseases, d_emb) * 0.05)
        self.cancer_prototypes = nn.Parameter(torch.randn(n_cancers, d_emb) * 0.05)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(12.0)))

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_norm = F.normalize(z, p=2, dim=-1)
        c_proto = F.normalize(self.cell_prototypes, p=2, dim=-1)
        d_proto = F.normalize(self.disease_prototypes, p=2, dim=-1)
        v_proto = F.normalize(self.cancer_prototypes, p=2, dim=-1)

        scale = self.logit_scale.exp().clamp(1.0, 50.0)
        sim_cell = torch.matmul(z_norm, c_proto.t())       # (B, 14) in [-1, 1]
        sim_disease = torch.matmul(z_norm, d_proto.t())    # (B, 4)  in [-1, 1]
        sim_cancer = torch.matmul(z_norm, v_proto.t())     # (B, 12) in [-1, 1]

        return {
            "sim_cell": sim_cell,
            "sim_disease": sim_disease,
            "sim_cancer": sim_cancer,
            "proto_logits_cell": sim_cell * scale,
            "proto_logits_disease": sim_disease * scale,
            "proto_logits_cancer": sim_cancer * scale,
        }


class HierarchicalScMambaBiMambaModel(nn.Module):
    """
    Complete Hybrid Architecture:
    1. Pretrained 57M-Cell SC-MAMBA2 Gene Token Embedding (`pretrained_gene_emb`, 60,697 genes x 512)
       + Continuous Binned Value Encoder + Chromosomal Sequence Gated BiMamba Backbone
    2. Patch-Based Core Diagnostic Panel BiMamba Encoder (256 chromo-ordered marker genes)
    3. 16-Pathway Biological Signature & Modality (Single-Cell vs Bulk) Cross-Gated Encoder
    4. Pattern Prototype Bank + 6 Hierarchical Conditional Diagnostic Task Heads
    5. Built-in Temperature Parameter `T` for Post-Hoc Confidence Calibration
    """
    def __init__(
        self,
        vocab_size: int = 60700,
        d_pretrained: int = 512,
        d_model: int = 256,
        n_bimamba_layers: int = 4,
        panel_size: int = 512,
        n_sigs: int = 16,
        n_celltypes: int = 14,
        n_diseases: int = 4,
        n_cancers: int = 11,
        use_bimamba: bool = True,
        pad_token_id: int = 60694,
    ):
        super().__init__()
        self.use_bimamba = use_bimamba
        self.pad_token_id = pad_token_id

        # 1. Pretrained 57M-cell Gene Vocabulary Embedding (initialized from best_model.pt & frozen to preserve foundation geometry!)
        self.gene_emb = nn.Embedding(vocab_size, d_pretrained, padding_idx=pad_token_id)
        self.gene_emb.weight.requires_grad_(False)
        self.val_proj = nn.Sequential(
            nn.Linear(1, d_pretrained // 4),
            nn.SiLU(),
            nn.Linear(d_pretrained // 4, d_pretrained),
        )
        self.token_down_proj = nn.Sequential(
            nn.LayerNorm(d_pretrained),
            nn.Linear(d_pretrained, d_model),
        )

        # 2. Sequence BiMamba Layers scanning chromosomally-ordered gene tokens
        if use_bimamba:
            self.seq_layers = nn.ModuleList(
                [GatedBiMambaLayer(d_model=d_model, d_state=64, d_conv=4, expand=2) for _ in range(n_bimamba_layers)]
            )
        else:
            # Baseline ablation: simple unidirectional / shared-MLP mixer without BiMamba gating
            self.seq_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, d_model),
                        nn.GELU(),
                        nn.Linear(d_model, d_model),
                    )
                    for _ in range(n_bimamba_layers)
                ]
            )

        self.seq_attn_pool = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # 3. Core Diagnostic Panel Patch-BiMamba Branch (256 chromo-ordered genes -> 16 patches of 16 genes)
        self.patch_size = 16
        self.n_patches = panel_size // self.patch_size
        self.patch_embed = nn.Sequential(
            nn.Linear(self.patch_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.patch_pos_emb = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        if use_bimamba:
            self.patch_bimamba = GatedBiMambaLayer(d_model=d_model, d_state=32, d_conv=4, expand=2)
        else:
            self.patch_bimamba = None

        self.panel_direct_mlp = nn.Sequential(
            nn.LayerNorm(panel_size),
            nn.Linear(panel_size, d_model),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model, d_model),
        )

        # 4. Biological Signature (16 pathways) + Modality (Single-Cell vs Bulk) Encoder
        self.modality_emb = nn.Embedding(2, 32)  # 0=Single-Cell, 1=Bulk
        self.sig_encoder = nn.Sequential(
            nn.Linear(n_sigs + 32, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # 5. Multi-Modal Gated Fusion (Combining Token-BiMamba + Patch-BiMamba + BioSignatures)
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # 6. Pattern Prototype Bank
        self.prototype_bank = PatternPrototypeBank(
            d_emb=d_model, n_celltypes=n_celltypes, n_diseases=n_diseases, n_cancers=n_cancers
        )

        # 7. Hierarchical Conditional Task Heads
        # Level 1: Cell Type (14 classes)
        self.head_celltype = nn.Sequential(
            nn.Linear(d_model + n_sigs, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_celltypes),
        )
        # Level 2 (4a): Healthy (0) vs Diseased (1), conditioned on Cell Type probabilities
        self.head_health = nn.Sequential(
            nn.Linear(d_model + n_sigs + n_celltypes, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 2),
        )
        # Level 3 (4b): Disease Archetype (0: Healthy, 1: COVID-19, 2: MS, 3: Cancer), conditioned on CellType + Health
        self.head_disease = nn.Sequential(
            nn.Linear(d_model + n_sigs + n_celltypes + 2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_diseases),
        )
        # Level 4 (4c): Is Cancer? (0: Non-Cancer, 1: Cancer), conditioned on CellType + Disease Archetype
        self.head_cancer_bin = nn.Sequential(
            nn.Linear(d_model + n_sigs + n_celltypes + n_diseases, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 2),
        )
        # Level 5 (4d): Which Cancer Variant? (12 classes), conditioned on CellType + Disease + Cancer Binary
        self.head_cancer_var = nn.Sequential(
            nn.Linear(d_model + n_sigs + n_celltypes + n_diseases + 2, d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_cancers),
        )
        # Level 6: Cancer Stem Cell (CSC- vs CSC+) conditioned on z + BioSignatures + Cancer Variant
        self.head_csc = nn.Sequential(
            nn.Linear(d_model + n_sigs + n_cancers, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 2),
        )

        # Post-hoc Confidence Calibration Temperatures (learned on Validation Split in FAZ 7)
        self.register_buffer("temp_celltype", torch.ones(1))
        self.register_buffer("temp_health", torch.ones(1))
        self.register_buffer("temp_disease", torch.ones(1))
        self.register_buffer("temp_cancer_bin", torch.ones(1))
        self.register_buffer("temp_cancer_var", torch.ones(1))
        self.register_buffer("temp_csc", torch.ones(1))

    def load_pretrained_scmamba2_embeddings(self, ckpt_path: str):
        """
        Loads the 57-million-cell pretrained GeneEncoder embeddings from SC-MAMBA2 `best_model.pt`.
        """
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        for key in ("encoder.embedding.weight", "module.encoder.embedding.weight"):
            if key in sd:
                w = sd[key]
                n_copy = min(w.size(0), self.gene_emb.weight.size(0))
                with torch.no_grad():
                    self.gene_emb.weight[:n_copy].copy_(w[:n_copy])
                print(f"[FAZ 4] Loaded 57M-cell pretrained GeneEncoder weights for {n_copy} genes from {ckpt_path}!")
                return True
        print("[WARNING] Could not find encoder.embedding.weight in checkpoint.")
        return False

    def forward(
        self,
        tokens_genes: torch.Tensor,
        tokens_vals: torch.Tensor,
        panel_expr: torch.Tensor,
        bio_sigs_norm: torch.Tensor,
        modality: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # 1. Tokenized Chromosomal Gene Sequence Branch
        mask_valid = (tokens_genes != self.pad_token_id).float()  # (B, L)
        g_emb = self.gene_emb(tokens_genes)                       # (B, L, 512)
        v_emb = self.val_proj(tokens_vals.unsqueeze(-1))          # (B, L, 512)
        seq_x = self.token_down_proj(g_emb + v_emb)               # (B, L, d_model)

        token_saliences = []
        for layer in self.seq_layers:
            if self.use_bimamba:
                seq_x, sal = layer(seq_x)
                token_saliences.append(sal)
            else:
                seq_x = seq_x + layer(seq_x)

        # Attentional pooling over valid non-pad tokens + <cls> token
        attn_scores = self.seq_attn_pool(seq_x).squeeze(-1)  # (B, L)
        attn_scores = attn_scores.masked_fill(mask_valid == 0, -1e4)
        attn_weights = F.softmax(attn_scores, dim=-1)        # (B, L)
        z_seq = 0.5 * seq_x[:, 0, :] + 0.5 * torch.bmm(attn_weights.unsqueeze(1), seq_x).squeeze(1)

        # 2. Core Diagnostic Panel Patch-BiMamba Branch
        B = panel_expr.size(0)
        patches = panel_expr.view(B, self.n_patches, self.patch_size)
        p_x = self.patch_embed(patches) + self.patch_pos_emb
        if self.patch_bimamba is not None:
            p_x, _ = self.patch_bimamba(p_x)
            z_patch = p_x.mean(dim=1)
            z_panel_direct = self.panel_direct_mlp(panel_expr)
        else:
            # Unidirectional baseline ablation without Patch-BiMamba mixer
            z_patch = p_x[:, 0, :]
            z_panel_direct = torch.zeros_like(z_patch)

        # 3. Biological Signatures + Modality Branch
        mod_e = self.modality_emb(modality.long())
        z_sig = self.sig_encoder(torch.cat([bio_sigs_norm, mod_e], dim=-1))

        # 4. Unified Cell/Tissue Representation z in R^d_model
        z = self.fusion_proj(torch.cat([z_seq, z_patch, z_panel_direct, z_sig], dim=-1))

        # 5. Pattern Prototype Similarities & Logits
        proto_out = self.prototype_bank(z)

        # 6. Hierarchical Conditional Classification Heads (fused with Prototype Logits!)
        # 6.1 Cell Type
        logits_celltype = 0.7 * self.head_celltype(torch.cat([z, bio_sigs_norm], dim=-1)) + 0.3 * proto_out["proto_logits_cell"]
        prob_celltype = F.softmax(logits_celltype / self.temp_celltype, dim=-1)

        # 6.2 Health Status (4a: Healthy vs Diseased)
        logits_health = self.head_health(torch.cat([z, bio_sigs_norm, prob_celltype], dim=-1))
        prob_health = F.softmax(logits_health / self.temp_health, dim=-1)

        # 6.3 Disease Archetype (4b: Healthy, COVID-19, MS, Cancer)
        logits_disease = (
            0.7 * self.head_disease(torch.cat([z, bio_sigs_norm, prob_celltype, prob_health], dim=-1))
            + 0.3 * proto_out["proto_logits_disease"]
        )
        prob_disease = F.softmax(logits_disease / self.temp_disease, dim=-1)

        # 6.4 Cancer Binary (4c: Non-Cancer vs Cancer)
        logits_cancer_bin = self.head_cancer_bin(torch.cat([z, bio_sigs_norm, prob_celltype, prob_disease], dim=-1))
        prob_cancer_bin = F.softmax(logits_cancer_bin / self.temp_cancer_bin, dim=-1)

        # 6.5 Cancer Variant (4d: Non-Cancer + 11 Cancer Variants)
        logits_cancer_var = (
            0.7 * self.head_cancer_var(torch.cat([z, bio_sigs_norm, prob_celltype, prob_disease, prob_cancer_bin], dim=-1))
            + 0.3 * proto_out["proto_logits_cancer"]
        )
        prob_cancer_var = F.softmax(logits_cancer_var / self.temp_cancer_var, dim=-1)

        # 6.6 Cancer Stem Cell (CSC- vs CSC+)
        logits_csc = self.head_csc(torch.cat([z, bio_sigs_norm, prob_cancer_var], dim=-1))
        prob_csc = F.softmax(logits_csc / self.temp_csc, dim=-1)

        return {
            "z": z,
            "logits_celltype": logits_celltype,
            "logits_health": logits_health,
            "logits_disease": logits_disease,
            "logits_cancer_bin": logits_cancer_bin,
            "logits_cancer_var": logits_cancer_var,
            "logits_csc": logits_csc,
            "prob_celltype": prob_celltype,
            "prob_health": prob_health,
            "prob_disease": prob_disease,
            "prob_cancer_bin": prob_cancer_bin,
            "prob_cancer_var": prob_cancer_var,
            "prob_csc": prob_csc,
            "sim_cell": proto_out["sim_cell"],
            "sim_disease": proto_out["sim_disease"],
            "sim_cancer": proto_out["sim_cancer"],
            "attn_weights": attn_weights,
        }
