"""
FAZ 5: Hiyerarşik Odaklı Kayıp (Focal Loss), Prototip Kontrastif Kayıp ve
Ağaç Tutarlılık Kısıtı (`losses.py`)
-------------------------------------------------------------------------
Referans Makaleler:
- Lin et al., ICCV (2017) - Focal Loss for Dense Object Detection (Nadir CSC ve alt-tipler için)
- Khosla et al., NeurIPS (2020) - Supervised Contrastive Learning (Örüntü merkezleri için)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.03):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", weight=self.weight, label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        focal_term = (1.0 - pt) ** self.gamma
        return (focal_term * ce_loss).mean()


class HierarchicalDiagnosticLoss(nn.Module):
    """
    Combines:
    1. Class-balanced Focal Losses for all 6 hierarchical diagnostic heads
    2. Prototype Contrastive Alignment Loss (aligning cell embeddings z with prototype centroids mu_k)
    3. Tree Logical Consistency Loss:
       - P(Diseased=0) <-> P(DiseaseCat=Healthy)
       - P(Cancer=1)   <-> P(DiseaseCat=Cancer)
       - P(Cancer=0)   <-> P(CancerVariant=Non_Cancer)
       - P(CSC=1)      <=  P(Cancer=1)
    """
    def __init__(
        self,
        w_celltype: Optional[torch.Tensor] = None,
        w_disease: Optional[torch.Tensor] = None,
        w_cancer_var: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.loss_cell = FocalLoss(gamma=1.5, weight=w_celltype, label_smoothing=0.02)
        self.loss_health = FocalLoss(gamma=1.5, label_smoothing=0.01)
        self.loss_disease = FocalLoss(gamma=1.5, weight=w_disease, label_smoothing=0.02)
        self.loss_cancer_bin = FocalLoss(gamma=1.5, label_smoothing=0.01)
        self.loss_cancer_var = FocalLoss(gamma=2.0, weight=w_cancer_var, label_smoothing=0.02)
        self.loss_csc = FocalLoss(gamma=2.0, label_smoothing=0.01)

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l_cell = self.loss_cell(out["logits_celltype"], batch["cell_type"])
        l_health = self.loss_health(out["logits_health"], batch["is_diseased"])
        l_dis = self.loss_disease(out["logits_disease"], batch["disease_cat"])
        l_cbin = self.loss_cancer_bin(out["logits_cancer_bin"], batch["is_cancer"])
        l_cvar = self.loss_cancer_var(out["logits_cancer_var"], batch["cancer_var"])
        l_csc = self.loss_csc(out["logits_csc"], batch["is_csc"])

        # Prototype contrastive alignment loss
        l_proto = (
            0.33 * F.cross_entropy(out["sim_cell"] * 12.0, batch["cell_type"])
            + 0.33 * F.cross_entropy(out["sim_disease"] * 12.0, batch["disease_cat"])
            + 0.34 * F.cross_entropy(out["sim_cancer"] * 12.0, batch["cancer_var"])
        )

        # Tree logical consistency loss
        p_health = F.softmax(out["logits_health"], dim=-1)          # [P(Healthy), P(Diseased)]
        p_dis = F.softmax(out["logits_disease"], dim=-1)            # [P(Healthy), P(COVID), P(MS), P(Cancer)]
        p_cbin = F.softmax(out["logits_cancer_bin"], dim=-1)        # [P(NonCancer), P(Cancer)]
        p_cvar = F.softmax(out["logits_cancer_var"], dim=-1)        # [P(NonCancer), P(Var1)..P(Var10)]
        p_csc = F.softmax(out["logits_csc"], dim=-1)                # [P(CSC-), P(CSC+)]

        l_consistency = (
            F.mse_loss(p_health[:, 0], p_dis[:, 0])
            + F.mse_loss(p_cbin[:, 1], p_dis[:, 3])
            + F.mse_loss(p_cbin[:, 0], p_cvar[:, 0])
            + F.relu(p_csc[:, 1] - p_cbin[:, 1]).mean()
        )

        total = (
            1.2 * l_cell
            + 1.0 * l_health
            + 1.2 * l_dis
            + 1.0 * l_cbin
            + 1.5 * l_cvar
            + 1.0 * l_csc
            + 0.4 * l_proto
            + 0.5 * l_consistency
        )

        return {
            "total": total,
            "l_cell": l_cell.detach(),
            "l_health": l_health.detach(),
            "l_dis": l_dis.detach(),
            "l_cbin": l_cbin.detach(),
            "l_cvar": l_cvar.detach(),
            "l_csc": l_csc.detach(),
            "l_proto": l_proto.detach(),
            "l_consistency": l_consistency.detach(),
        }
