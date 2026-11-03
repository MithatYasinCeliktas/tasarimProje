"""
FAZ 7 & FAZ 8: Açıklanabilir Yapay Zeka (XAI) 4-Panelli Klinik Teşhis Heatmap ve
A/B Karşılaştırma Görselleştirme Motoru (`diagnostic_heatmap.py`)
--------------------------------------------------------------------------------
Bu modül, eğitilmiş `HierarchicalScMambaBiMambaModel` çıktılarından:
1. Tek-Hücre (scRNA-seq) ve Bulk (Bulk RNA-seq) klinik test vakaları için 4-Panelli
   Yüksek Çözünürlüklü Teşhis ve XAI Heatmap Raporları (`diagnostic_report_<case>.png`),
2. Çoklu-Örnek Biyolojik Yolak & Örüntü Uyum Master Heatmap'i (`master_xai_pattern_heatmap.png`),
3. Faz 8 A/B Mimarisi Karşılaştırma ve Hata Matrisi (Confusion Matrix) Heatmap'ini
   (`faz8_ab_benchmark_and_confusion.png`) üretir.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Consistent high-contrast scientific styling
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.8


def build_single_case_figure(
    case_idx: int,
    case_name: str,
    xai_data: Dict[str, np.ndarray],
    metadata: Dict,
) -> plt.Figure:
    """
    Builds and returns a 4-Panel Clinical Diagnostic & XAI Heatmap Figure for a
    Single-Cell (scRNA-seq) or Bulk (Bulk RNA-seq) patient sample.
    """
    panel_genes: List[str] = metadata.get("CORE_PANEL_GENES", metadata.get("panel_genes", []))
    sig_names: List[str] = metadata.get("SIGNATURE_NAMES", metadata.get("biological_signature_names", []))
    cell_names: List[str] = metadata.get("CELL_TYPES", metadata.get("cell_type_names", []))
    disease_names: List[str] = metadata.get("DISEASE_CATEGORIES", metadata.get("disease_names", []))
    cancer_names: List[str] = metadata.get("CANCER_VARIANTS", metadata.get("cancer_variant_names", []))

    mod_val = int(xai_data["modality"][case_idx])
    mod_str = "BULK RNA-seq (Doku Biyopsisi)" if mod_val == 1 else "TEK-HÜCRE (scRNA-seq)"
    mod_badge_color = "#7c3aed" if mod_val == 1 else "#0284c7"

    # Extract true & predicted classes + calibrated confidences
    t_ct = int(xai_data["true_celltype"][case_idx])
    t_hl = int(xai_data["true_health"][case_idx])
    t_ds = int(xai_data["true_disease"][case_idx])
    t_cb = int(xai_data["true_cancer_bin"][case_idx])
    t_cv = int(xai_data["true_cancer_var"][case_idx])
    t_csc = int(xai_data["true_csc"][case_idx])

    p_ct_vec = xai_data["prob_celltype"][case_idx]
    p_hl_vec = xai_data["prob_health"][case_idx]
    p_ds_vec = xai_data["prob_disease"][case_idx]
    p_cb_vec = xai_data["prob_cancer_bin"][case_idx]
    p_cv_vec = xai_data["prob_cancer_var"][case_idx]
    p_csc_vec = xai_data["prob_csc"][case_idx]

    pred_ct, conf_ct = int(p_ct_vec.argmax()), float(p_ct_vec.max()) * 100.0
    pred_hl, conf_hl = int(p_hl_vec.argmax()), float(p_hl_vec.max()) * 100.0
    pred_ds, conf_ds = int(p_ds_vec.argmax()), float(p_ds_vec.max()) * 100.0
    pred_cb, conf_cb = int(p_cb_vec.argmax()), float(p_cb_vec.max()) * 100.0
    pred_cv, conf_cv = int(p_cv_vec.argmax()), float(p_cv_vec.max()) * 100.0
    pred_csc, conf_csc = int(p_csc_vec.argmax()), float(p_csc_vec.max()) * 100.0

    fig = plt.figure(figsize=(20, 13.5), dpi=180, facecolor="#f8fafc")
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.05, 1.35], height_ratios=[1.0, 1.05], wspace=0.22, hspace=0.28)

    # Top Title Banner
    fig.suptitle(
        f"scMamba2 + BiMamba Hiyerarşik Klinik Teşhis & XAI Örüntü Raporu  |  Örnek: {case_name} [{mod_str}]",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
        y=0.975,
    )

    # =========================================================================
    # PANEL A (Top-Left): Hiyerarşik Klinik Karar Ağacı & Kalibre Güven Kartı
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#ffffff")
    ax_a.set_title(
        "PANEL A: Hiyerarşik Klinik Teşhis Sonuçları ve Kalibre Güven Yüzdeleri",
        fontsize=11.5,
        fontweight="bold",
        color="#1e293b",
        pad=10,
    )

    health_labels = ["SAĞLIKLI (Healthy)", "HASTALIKLI (Diseased)"]
    cancer_bin_labels = ["KANSER DEĞİL (Non-Cancer)", "KANSER (Malignant)"]
    csc_labels = ["Negatif (Non-CSC)", "POZİTİF (Kanser Kök Hücresi / CSC+)"]

    rows_info = [
        ("1. Hücre Tipi Tahmini", cell_names[pred_ct], cell_names[t_ct], conf_ct, pred_ct == t_ct),
        ("4a. Sağlıklı mı? Hastalıklı mı?", health_labels[pred_hl], health_labels[t_hl], conf_hl, pred_hl == t_hl),
        ("4b. Hastalık Örüntüsü (Arketipi)", disease_names[pred_ds], disease_names[t_ds], conf_ds, pred_ds == t_ds),
        ("4c. Kanser mi? Değil mi?", cancer_bin_labels[pred_cb], cancer_bin_labels[t_cb], conf_cb, pred_cb == t_cb),
        ("4d. Kanser Varyantı", cancer_names[pred_cv], cancer_names[t_cv], conf_cv, pred_cv == t_cv),
        ("4d-Ek. Kanser Kök Hücre (CSC)", csc_labels[pred_csc], csc_labels[t_csc], conf_csc, pred_csc == t_csc),
    ]

    y_pos = np.arange(len(rows_info))[::-1]
    confs = [r[3] for r in rows_info]
    bar_colors = ["#16a34a" if r[4] else "#dc2626" for r in rows_info]

    ax_a.barh(y_pos, confs, height=0.56, color=bar_colors, alpha=0.85, edgecolor="#0f172a", linewidth=0.8)
    ax_a.set_xlim(0, 118)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([r[0] for r in rows_info], fontsize=9.5, fontweight="bold", color="#1e293b")
    ax_a.set_xlabel("Kalibre Edilmiş Güven / Doğruluk Yüzdesi (%)", fontsize=9.5, fontweight="bold")
    ax_a.axvline(95.0, color="#0284c7", linestyle="--", linewidth=1.2, label="%95 Hedef Eşiği")
    ax_a.grid(axis="x", linestyle=":", alpha=0.5)

    for idx_r, (stage_lbl, pred_str, true_str, cf, is_ok) in enumerate(rows_info):
        y_v = y_pos[idx_r]
        mark = "DOĞRU" if is_ok else f"Gerçek: {true_str}"
        ax_a.text(
            2.5,
            y_v,
            f"Tahmin: {pred_str}  [{mark}]",
            va="center",
            ha="left",
            fontsize=9.0,
            fontweight="bold",
            color="#ffffff" if cf > 45 else "#0f172a",
        )
        ax_a.text(
            cf + 1.5,
            y_v,
            f"%{cf:.1f}",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#0f172a",
        )

    # Input modality badge inside Panel A
    ax_a.text(
        0.02,
        1.02,
        f"Girdi Modu: {mod_str}",
        transform=ax_a.transAxes,
        fontsize=9.0,
        fontweight="bold",
        color="#ffffff",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=mod_badge_color, edgecolor="none"),
    )

    # =========================================================================
    # PANEL B (Top-Right): Top-20 Karar Verici Gen Feature & İfade Heatmap'i
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    expr_vec = xai_data["panel_expr"][case_idx]  # (512,)
    attr_vec = xai_data["panel_attr"][case_idx]  # (512,)
    # Combine gradient salience with expression magnitude so top active diagnostic markers stand out
    combined_importance = attr_vec / (attr_vec.max() + 1e-8) + 0.35 * (expr_vec / (expr_vec.max() + 1e-8))
    top_k = 18
    top_gene_idx = np.argsort(combined_importance)[::-1][:top_k]

    healthy_ref = xai_data["disease_ref_panel_expr"][0][top_gene_idx]
    if pred_cv > 0:
        matched_ref = xai_data["cancer_ref_panel_expr"][pred_cv][top_gene_idx]
    else:
        matched_ref = xai_data["disease_ref_panel_expr"][pred_ds][top_gene_idx]

    sample_top_expr = expr_vec[top_gene_idx]
    norm_salience = (combined_importance[top_gene_idx] / (combined_importance[top_gene_idx].max() + 1e-8)) * max(
        float(sample_top_expr.max()), 2.0
    )

    heatmap_mat_b = np.vstack([sample_top_expr, matched_ref, healthy_ref, norm_salience])
    im_b = ax_b.imshow(heatmap_mat_b, aspect="auto", cmap="magma")

    gene_col_labels = [f"{panel_genes[i]}\n(Rank #{r+1})" for r, i in enumerate(top_gene_idx)]
    ax_b.set_xticks(np.arange(top_k))
    ax_b.set_xticklabels(gene_col_labels, rotation=45, ha="right", fontsize=8.0, fontweight="bold")
    ax_b.set_yticks([0, 1, 2, 3])
    ax_b.set_yticklabels(
        [
            "Hasta Örneği İfadesi (log1p)",
            "Eşleşen Hastalık/Kanser Örüntüsü",
            "Sağlıklı Kontrol Referansı",
            "BiMamba XAI Karar Ağırlığı",
        ],
        fontsize=8.8,
        fontweight="bold",
    )
    ax_b.set_title(
        "PANEL B: Kararı Belirleyen Top-18 Gen Feature İfadesi ve BiMamba XAI Önem Heatmap'i",
        fontsize=11.5,
        fontweight="bold",
        color="#1e293b",
        pad=10,
    )

    for r_i in range(heatmap_mat_b.shape[0]):
        for c_i in range(heatmap_mat_b.shape[1]):
            val = heatmap_mat_b[r_i, c_i]
            txt_color = "white" if val < (heatmap_mat_b.max() * 0.55) else "black"
            ax_b.text(c_i, r_i, f"{val:.1f}", ha="center", va="center", fontsize=7.5, color=txt_color, fontweight="bold")

    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.025, pad=0.02)
    cbar_b.ax.tick_params(labelsize=8)

    # =========================================================================
    # PANEL C (Bottom-Left): Hastalık & Kanser Varyantı Örüntü Uyum Heatmap'i
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    sim_c = xai_data["sim_cancer"][case_idx]  # (11,) cosine similarity in [-1, 1]
    prob_c = p_cv_vec * 100.0  # (11,) %
    # Normalize both rows to [0, 100] scale for clean joint heatmap visualization
    sim_c_scaled = np.clip((sim_c + 1.0) * 50.0, 0.0, 100.0)
    pattern_mat = np.vstack([prob_c, sim_c_scaled])

    im_c = ax_c.imshow(pattern_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    short_cancer_names = [
        "Non-Cancer",
        "Lung LUAD",
        "NSCLC A549",
        "Leukemia CML",
        "Kidney RCC",
        "Endometrial UCEC",
        "Esophageal ESCA",
        "Myeloma MYE",
        "Thyroid THCA",
        "Ovarian OV",
        "Pancreas PAAD",
    ]
    ax_c.set_xticks(np.arange(len(short_cancer_names)))
    ax_c.set_xticklabels(short_cancer_names, rotation=38, ha="right", fontsize=8.5, fontweight="bold")
    ax_c.set_yticks([0, 1])
    ax_c.set_yticklabels(
        ["Kalibre Model Olasılığı (%)", "Prototip Örüntü Uyumu (%)"],
        fontsize=9.0,
        fontweight="bold",
    )
    ax_c.set_title(
        "PANEL C: Hastalık ve 11 Kanser Varyantı Prototip Örüntü Uyum Heatmap'i",
        fontsize=11.5,
        fontweight="bold",
        color="#1e293b",
        pad=10,
    )

    for r_i in range(2):
        for c_i in range(len(short_cancer_names)):
            v = pattern_mat[r_i, c_i]
            tc = "white" if v > 55 else "#0f172a"
            ax_c.text(c_i, r_i, f"{v:.1f}%", ha="center", va="center", fontsize=7.8, fontweight="bold", color=tc)

    # Also annotate the 4 Disease Archetype probabilities above the cancer matrix
    dis_summary = "  |  ".join([f"{disease_names[i]}: %{p_ds_vec[i]*100:.1f}" for i in range(4)])
    ax_c.text(
        0.5,
        -0.42,
        f"4b. Ana Hastalık Arketip Dağılımı ->  {dis_summary}",
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e2e8f0", edgecolor="#94a3b8"),
    )

    # =========================================================================
    # PANEL D (Bottom-Right): 16 Biyolojik Yolak & CSC İmza Heatmap'i
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    sig_z = xai_data["bio_sigs_norm"][case_idx]  # (16,)
    sig_imp = xai_data["sig_attr"][case_idx]  # (16,)
    sig_imp_norm = (sig_imp / (sig_imp.max() + 1e-8)) * 3.0

    sig_mat = np.vstack([sig_z, sig_imp_norm])
    im_d = ax_d.imshow(sig_mat, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.8)

    short_sig_names = [
        "CSC Stemness",
        "Proliferation",
        "EMT Metastasis",
        "Hypoxia/Glyco",
        "P53 DNA Repair",
        "Viral ISG",
        "MS NeuroInflam",
        "TAM Angio SPP1",
        "TAM Comp C1QC",
        "T-Cell Exhaust",
        "Epithelial Lin",
        "Neural/Glial",
        "Hematopoietic",
        "CytoTRACE Ent",
        "Chr CNV Instab",
        "Modality Dens",
    ]
    ax_d.set_xticks(np.arange(len(short_sig_names)))
    ax_d.set_xticklabels(short_sig_names, rotation=38, ha="right", fontsize=8.2, fontweight="bold")
    ax_d.set_yticks([0, 1])
    ax_d.set_yticklabels(["Hasta Z-Skoru (Aktivasyon)", "XAI Yolak Karar Etkisi"], fontsize=9.0, fontweight="bold")
    ax_d.set_title(
        "PANEL D: 16 Biyolojik Yolak, Kök Hücre (CSC) ve Kromozomal CNV İmza Heatmap'i",
        fontsize=11.5,
        fontweight="bold",
        color="#1e293b",
        pad=10,
    )

    for r_i in range(2):
        for c_i in range(len(short_sig_names)):
            v = sig_mat[r_i, c_i]
            tc = "white" if abs(v) > 1.4 else "#0f172a"
            ax_d.text(c_i, r_i, f"{v:+.1f}", ha="center", va="center", fontsize=7.5, fontweight="bold", color=tc)

    cbar_d = fig.colorbar(im_d, ax=ax_d, fraction=0.025, pad=0.02)
    cbar_d.ax.tick_params(labelsize=8)

    return fig


def render_single_case_dashboard(
    case_idx: int,
    case_name: str,
    xai_data: Dict[str, np.ndarray],
    metadata: Dict,
    out_path: Path,
):
    """
    Renders and saves a 4-Panel Clinical Diagnostic Dashboard PNG file.
    """
    fig = build_single_case_figure(case_idx, case_name, xai_data, metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def render_master_xai_overview_heatmap(xai_data: Dict[str, np.ndarray], metadata: Dict, out_path: Path):
    """
    Renders a Master Multi-Sample XAI Heatmap comparing all representative Single-Cell
    and Bulk RNA-seq cases across the 16 Biological Pathways and 11 Cancer Variants.
    """
    case_names = [str(x) for x in xai_data["case_names"]]
    sig_z = xai_data["bio_sigs_norm"]  # (N_cases, 16)
    prob_cancer = xai_data["prob_cancer_var"] * 100.0  # (N_cases, 11)

    short_sig_names = [
        "CSC Stemness",
        "Proliferation",
        "EMT Metastasis",
        "Hypoxia/Glyco",
        "P53 DNA Repair",
        "Viral ISG",
        "MS NeuroInflam",
        "TAM Angio SPP1",
        "TAM Comp C1QC",
        "T-Cell Exhaust",
        "Epithelial Lin",
        "Neural/Glial",
        "Hematopoietic",
        "CytoTRACE Ent",
        "Chr CNV Instab",
        "Modality Dens",
    ]
    short_cancer_names = [
        "Non-Cancer",
        "Lung LUAD",
        "NSCLC A549",
        "Leukemia CML",
        "Kidney RCC",
        "Endometrial UCEC",
        "Esophageal ESCA",
        "Myeloma MYE",
        "Thyroid THCA",
        "Ovarian OV",
        "Pancreas PAAD",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7.5), dpi=180, gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.suptitle(
        "Tek-Hücre (scRNA-seq) ve Bulk RNA-seq Klinik Vakaları: Biyolojik Yolak & Kanser Varyantı Örüntü Heatmap'i",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    im1 = ax1.imshow(sig_z, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.5)
    ax1.set_yticks(np.arange(len(case_names)))
    ax1.set_yticklabels(case_names, fontsize=9, fontweight="bold")
    ax1.set_xticks(np.arange(len(short_sig_names)))
    ax1.set_xticklabels(short_sig_names, rotation=40, ha="right", fontsize=8.5, fontweight="bold")
    ax1.set_title("16 Biyolojik Yolak & CSC İmza Aktivasyonu (Z-Skor Heatmap)", fontsize=11, fontweight="bold")
    for r in range(sig_z.shape[0]):
        for c in range(sig_z.shape[1]):
            v = sig_z[r, c]
            ax1.text(c, r, f"{v:+.1f}", ha="center", va="center", fontsize=7.2, color="white" if abs(v) > 1.3 else "black")
    fig.colorbar(im1, ax=ax1, fraction=0.025, pad=0.02)

    im2 = ax2.imshow(prob_cancer, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax2.set_yticks(np.arange(len(case_names)))
    ax2.set_yticklabels(["" for _ in case_names])
    ax2.set_xticks(np.arange(len(short_cancer_names)))
    ax2.set_xticklabels(short_cancer_names, rotation=40, ha="right", fontsize=8.5, fontweight="bold")
    ax2.set_title("Kanser Varyantı Sınıflandırma Olasılık Örüntüsü (% Güven)", fontsize=11, fontweight="bold")
    for r in range(prob_cancer.shape[0]):
        for c in range(prob_cancer.shape[1]):
            v = prob_cancer[r, c]
            ax2.text(c, r, f"{v:.0f}%", ha="center", va="center", fontsize=7.5, fontweight="bold", color="white" if v > 50 else "black")
    fig.colorbar(im2, ax=ax2, fraction=0.025, pad=0.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def render_faz8_ab_and_confusion_dashboard(report: Dict, metadata: Dict, out_path: Path):
    """
    FAZ 8: Renders the A/B Architecture Comparison (Proposed Gated BiMamba vs Baseline),
    Single-Cell vs Bulk RNA-seq Accuracy Comparison, and Cancer Variant Confusion Matrix Heatmap.
    """
    bim = report["proposed_bimamba_metrics"]
    base = report["baseline_no_bimamba_metrics"]

    tasks = ["celltype", "health", "disease", "cancer_bin", "cancer_var", "csc"]
    task_labels = [
        "Hücre Tipi\n(14 Sınıf)",
        "Sağlıklı vs\nHastalıklı (4a)",
        "Hastalık\nÖrüntüsü (4b)",
        "Kanser mi?\nDeğil mi? (4c)",
        "Kanser\nVaryantı (4d)",
        "Kanser Kök\nHücre (CSC)",
    ]

    bim_acc = [bim["overall"][k]["accuracy"] for k in tasks]
    base_acc = [base["overall"][k]["accuracy"] for k in tasks]
    sc_acc = [bim["single_cell_only"][k]["accuracy"] for k in tasks]
    bulk_acc = [bim["bulk_rnaseq_only"][k]["accuracy"] for k in tasks]

    fig = plt.figure(figsize=(20, 11), dpi=180, facecolor="#f8fafc")
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.1, 1.2], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.32)
    fig.suptitle(
        "FAZ 8: scMamba2 + Gated BiMamba A/B Karşılaştırması, Bulk vs Tek-Hücre Performansı ve Hata Matrisleri",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    # 1. A/B Benchmark Plot
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(tasks))
    w = 0.36
    b1 = ax1.bar(x - w / 2, base_acc, w, label="Baseline (Tek Yönlü / BiMamba Yok)", color="#94a3b8", edgecolor="#1e293b")
    b2 = ax1.bar(x + w / 2, bim_acc, w, label="Önerilen: scMamba2 + Gated BiMamba", color="#0284c7", edgecolor="#1e293b")
    ax1.axhline(95.0, color="#dc2626", linestyle="--", linewidth=1.3, label="%95 Hedef Doğruluk Eşiği")
    ax1.set_ylim(75, 103)
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_labels, fontsize=8.5, fontweight="bold")
    ax1.set_ylabel("Test Seti Doğruluk (Accuracy %)", fontsize=9.5, fontweight="bold")
    ax1.set_title("A/B Mimari Karşılaştırması: Baseline vs Gated BiMamba", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"%{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0369a1")

    # 2. Single-Cell vs Bulk RNA-seq Performance Plot
    ax2 = fig.add_subplot(gs[1, 0])
    b_sc = ax2.bar(x - w / 2, sc_acc, w, label="Tek-Hücre (scRNA-seq Test Seti)", color="#0d9488", edgecolor="#1e293b")
    b_bk = ax2.bar(x + w / 2, bulk_acc, w, label="Bulk RNA-seq (Doku Biyopsisi Test Seti)", color="#7c3aed", edgecolor="#1e293b")
    ax2.axhline(95.0, color="#dc2626", linestyle="--", linewidth=1.3, label="%95 Hedef Doğruluk Eşiği")
    ax2.set_ylim(80, 103)
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_labels, fontsize=8.5, fontweight="bold")
    ax2.set_ylabel("Test Seti Doğruluk (Accuracy %)", fontsize=9.5, fontweight="bold")
    ax2.set_title("Modality Genellemesi: Tek-Hücre (scRNA-seq) vs Bulk RNA-seq Doğruluğu", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    for bar in b_sc:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"%{h:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#0f766e")
    for bar in b_bk:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"%{h:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#6d28d9")

    # 3. Cancer Variant Confusion Matrix Heatmap (11 classes, row-normalized %)
    ax3 = fig.add_subplot(gs[:, 1])
    cm_cancer = np.array(bim["confusion_matrices"]["cancer_var"], dtype=np.float64)
    cm_row_sum = np.maximum(cm_cancer.sum(axis=1, keepdims=True), 1.0)
    cm_pct = (cm_cancer / cm_row_sum) * 100.0

    im3 = ax3.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    short_cancer_names = [
        "Non-Cancer",
        "Lung LUAD",
        "NSCLC A549",
        "Leukemia CML",
        "Kidney RCC",
        "Endometrial UCEC",
        "Esophageal ESCA",
        "Myeloma MYE",
        "Thyroid THCA",
        "Ovarian OV",
        "Pancreas PAAD",
    ]
    ax3.set_xticks(np.arange(len(short_cancer_names)))
    ax3.set_xticklabels(short_cancer_names, rotation=40, ha="right", fontsize=8.5, fontweight="bold")
    ax3.set_yticks(np.arange(len(short_cancer_names)))
    ax3.set_yticklabels(short_cancer_names, fontsize=8.5, fontweight="bold")
    ax3.set_xlabel("Modelin Tahmin Ettiği Kanser Varyantı", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Gerçek Kanser Varyantı (Ground Truth)", fontsize=10, fontweight="bold")
    ax3.set_title("4d. Kanser Varyantı Sınıflandırma Hata Matrisi (Confusion Matrix - % Doğruluk)", fontsize=11.5, fontweight="bold")

    for r in range(cm_pct.shape[0]):
        for c in range(cm_pct.shape[1]):
            v = cm_pct[r, c]
            if v >= 0.5 or r == c:
                ax3.text(
                    c,
                    r,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold" if r == c else "normal",
                    color="white" if v > 50 else "#0f172a",
                )

    fig.colorbar(im3, ax=ax3, fraction=0.03, pad=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main():
    base_dir = Path("/kaggle/working/results") if os.path.exists("/kaggle/working/results") else Path("results")
    meta_candidates = [
        "/kaggle/working/dataset_metadata.json",
        "results/dataset_metadata.json",
    ]
    meta_path = next(p for p in meta_candidates if os.path.exists(p))
    metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))

    report = json.loads((base_dir / "faz6_faz8_evaluation_report.json").read_text(encoding="utf-8"))
    xai_raw = np.load(base_dir / "xai_case_studies.npz", allow_pickle=True)
    xai_data = {k: xai_raw[k] for k in xai_raw.files}

    heatmaps_dir = base_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    case_names = [str(x) for x in xai_data["case_names"]]
    for i, cname in enumerate(case_names):
        safe_name = cname.replace("+", "pos").replace(" ", "_")
        out_p = heatmaps_dir / f"diagnostic_report_{i+1:02d}_{safe_name}.png"
        render_single_case_dashboard(i, cname, xai_data, metadata, out_p)
        print(f"[XAI HEATMAP] Saved {out_p}", flush=True)

    master_p = heatmaps_dir / "master_xai_pattern_heatmap.png"
    render_master_xai_overview_heatmap(xai_data, metadata, master_p)
    print(f"[MASTER HEATMAP] Saved {master_p}", flush=True)

    ab_p = heatmaps_dir / "faz8_ab_benchmark_and_confusion.png"
    render_faz8_ab_and_confusion_dashboard(report, metadata, ab_p)
    print(f"[FAZ 8 DASHBOARD] Saved {ab_p}", flush=True)


if __name__ == "__main__":
    main()
