# -*- coding: utf-8 -*-
"""
scMamba2 + Gated BiMamba Ar-Ge Test ve Kaggle Kontrol Tezgahı (app.py)
----------------------------------------------------------------------
Kullanıcı Dostu İnteraktif Arayüz:
1. Kaggle 2x Tesla T4 GPU Canlı Kontrolü (Kota, Başlat/Durdur, Durum)
2. Eğitim Dışı (Unseen Test Split, N=8,486) Rastgele Tekil Hücre veya Grup Örnekleyici
3. Model Tahminleri vs. Gerçek Etiket Doğruluk Karşılaştırma Kartları
4. "Tahmin Neye Göre Alındı?" (XAI Kanıt Paneli: Top-15 Gen, 16 Yolak, Prototip Uyumu)
5. Dinamik 4-Panelli Klinik Heatmap Raporu & Tek Tıkla PNG İndirme
6. Toplu Stres Testi (Batch Validation Benchmarker: 20/50/100 Hücre)
"""

import base64
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

# Proje dizinini ekle
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hierarchical_bimamba import HierarchicalScMambaBiMambaModel
from src.visualization.diagnostic_heatmap import build_single_case_figure

# Sayfa Yapılandırması
st.set_page_config(
    page_title="scMamba2 + BiMamba Ar-Ge & Test Tezgahı",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Özel CSS Tasarımı
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric { background: #ffffff; padding: 12px 18px; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .badge-ok { background-color: #dcfce7; color: #15803d; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.85rem; border: 1px solid #86efac; }
    .badge-err { background-color: #fee2e2; color: #b91c1c; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.85rem; border: 1px solid #fca5a5; }
    .badge-info { background-color: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.85rem; border: 1px solid #7dd3fc; }
    .card-title { font-size: 0.82rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 4px; }
    .card-pred { font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 2px; }
    .card-true { font-size: 0.88rem; color: #475569; margin-bottom: 6px; }
    .card-conf { font-size: 0.95rem; font-weight: 700; color: #2563eb; }
    .decision-card { background: #ffffff; padding: 14px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
</style>
""", unsafe_allow_html=True)

# Biyolojik Gen Açıklamaları Sözlüğü
GENE_ANNOTATIONS = {
    "CD44": "Kanser Kök Hücresi (CSC) Yüzey Reseptörü, Kemodirenç & Metastaz",
    "PROM1": "(CD133) Kanser Kök Hücre Kök성 & Farklılaşmamışlık Belirteci",
    "ALDH1A1": "Kanser Kök Hücresi Enzimatik Aktivitesi ve Toksin Detoksifikasyonu",
    "SOX2": "Kök Hücre Pluripotens ve Kendi Kendini Yenileme (Self-Renewal) Faktörü",
    "NANOG": "Embriyonik Kök Hücre ve CSC Farklılaşmamışlık Ağı",
    "POU5F1": "(OCT4) Kanser Kök Hücre Ana Transkripsiyonel Aç-Kapa Faktörü",
    "EPCAM": "Karsinom Hücre Adezyon Molekülü ve Kanser Kök Hücre Belirteci",
    "ABCG2": "CSC İlaç Dışa-Atım Pompası (Kemoterapi Direnç Faktörü)",
    "MKI67": "(Ki-67) Malign Proliferasyon ve Agresif Hücre Bölünme Hızı",
    "TOP2A": "DNA Topoizomeraz II, Malign Hücre Döngüsü G2/M Hedefi",
    "PCNA": "Prolifere Olan Hücre Nükleer Antijeni (DNA Replikasyonu)",
    "VIM": "Vimentin - Epitelyal-Mezenkimal Geçiş (EMT) ve Tümör İnvazyonu",
    "SNAI1": "EMT Transkripsiyonel Baskılayıcı (Metastatik Başlatıcı)",
    "CDH2": "N-Cadherin, Mezenkimal Göç ve Metastaz Fenotipi",
    "VEGFA": "Tümör Hipoksi ve Anjiyogenez (Yeni Damar Oluşumu) Faktörü",
    "HIF1A": "Hipoksiyle İndüklenen Faktör 1-Alfa (Warburg Glikoliz Etkisi)",
    "LDHA": "Laktat Dehidrojenaz A (Kanser Glikoliz ve Asidik Mikroçevre)",
    "TP53": "Tümör Baskılayıcı P53 Proteini, DNA Hasar ve Onarım Kapısı",
    "CDKN1A": "(P21) Hücre Döngüsü Kinaz İnhibitörü ve P53 Yanıtı",
    "ISG15": "Tip-I İnterferon Uyarılı Gen 15 (Viral Enfeksiyon / COVID-19 İmzası)",
    "IFIT1": "İnterferon Tetratrikopeptid Yanıtı (Akut Viral İmmün Savunma)",
    "MX1": "Myxovirus Direnç Proteini 1 (Akut Antiviral İnterferon Yanıtı)",
    "HLA-DRA": "MHC Sınıf II Antijen Sunumu (Multipl Skleroz Nöro-İnflamasyon)",
    "CD74": "MHC Sınıf II Şaperonu ve Mikroglia İmmün Aktivasyonu (MS İmzası)",
    "C1QA": "Komplement C1q Alt Birimi A (Nörodejeneratif Sinaps Budama)",
    "APOE": "Apolipoprotein E, Hasarlı Mikroglia ve Lipid Taşınması",
    "GFAP": "Glial Fibriler Asidik Protein (Astrosit Aktivasyonu / Gliozis)",
    "MBP": "Miyelin Temel Proteini (Oligodendrosit ve Demiyelinizasyon Hasarı)",
    "SPP1": "Osteopontin - Pan-Kanser Anjiyojenik Tümör Makrofajı (TAM)",
    "C1QC": "Komplement Doku Makrofajı ve İmmünsüpresif Mikroçevre",
    "SFTPC": "Sürfaktan Protein C, Sağlıklı Alveolar Epitel (Akciğer AT2)",
    "SNAP25": "Sinaptik Nöronal İletim ve Aksonal Vezikül Proteini",
    "SLC17A7": "VGLUT1 Glutamaterjik Eksitatör Nöron Belirteci",
    "PTPRC": "(CD45) Tüm Bağışıklık Hücreleri (Pan-Hematopoietik Soy)",
    "CD14": "Monosit ve Doku Makrofajı LPS Reseptörü",
    "MS4A1": "(CD20) B Lenfosit Yüzey Antijeni",
    "CD3D": "T Lenfosit Reseptör Kompleksi Delta Zinciri",
    "CD8A": "Sitotoksik T Lenfosit Yüzey Koreseptörü",
    "MPO": "Miyeloperoksidaz, Miyeloid Lösemi (AML / CML) Belirteci",
    "INS": "İnsülin Hormonu, Pankreas Adacık Beta Hücresi",
}


@st.cache_resource(show_spinner=False)
def load_resources():
    """Model ağırlıklarını ve test veri setini bir kez belleğe yükler."""
    ckpt_path = PROJECT_ROOT / "results" / "best_hierarchical_bimamba.pt"
    ds_path = PROJECT_ROOT / "results" / "unified_hierarchical_dataset.npz"
    xai_path = PROJECT_ROOT / "results" / "xai_case_studies.npz"

    if not ckpt_path.exists() or not ds_path.exists():
        st.error("Model veya veri seti dosyaları `results/` altında bulunamadı!")
        st.stop()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = ckpt["metadata"]

    model = HierarchicalScMambaBiMambaModel(
        vocab_size=60700,
        d_pretrained=512,
        d_model=256,
        n_bimamba_layers=2,
        panel_size=512,
        n_sigs=16,
        n_celltypes=14,
        n_diseases=4,
        n_cancers=11,
        use_bimamba=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    ds = np.load(ds_path)
    xai_cases = np.load(xai_path, allow_pickle=True) if xai_path.exists() else None

    # Sadece eğitimde HİÇ kullanılmamış Test Seti (split == 2) indeksleri
    test_indices = np.flatnonzero(ds["split"] == 2)

    return model, metadata, ds, test_indices, xai_cases


def run_kaggle_command(action: str, timeout: Optional[int] = None) -> Tuple[bool, str]:
    """Kaggle köprü komutlarını çalıştırır."""
    if timeout is None:
        if action == "up":
            timeout = 260
        elif action == "setup":
            timeout = 1800
        else:
            timeout = 60
    cmd = [sys.executable, str(PROJECT_ROOT / "kaggle_bridge" / "kaggle_ctl.py"), action]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT_ROOT))
        return (p.returncode == 0, (p.stdout or "") + (p.stderr or ""))
    except Exception as e:
        return (False, str(e))



def compute_sample_xai(model, ds, idx: int, metadata: Dict, xai_cases) -> Dict:
    """Herhangi bir test hücresi için ileri yayılım ve XAI Input x Gradient önem skorlarını hesaplar."""
    model.eval()
    pe = torch.from_numpy(ds["panel_expr"][idx : idx + 1]).float().requires_grad_(True)
    bs = torch.from_numpy(ds["bio_sigs_norm"][idx : idx + 1]).float().requires_grad_(True)
    tg = torch.from_numpy(ds["tokens_genes"][idx : idx + 1]).long()
    tv = torch.from_numpy(ds["tokens_vals"][idx : idx + 1]).float()
    mod = torch.from_numpy(ds["modality"][idx : idx + 1]).long()

    out = model(tg, tv, pe, bs, mod)

    # Karara en çok etki eden skordan gradyan atfı
    target_score = (
        out["logits_celltype"].max(dim=-1).values.sum()
        + out["logits_disease"].max(dim=-1).values.sum()
        + out["logits_cancer_var"].max(dim=-1).values.sum()
        + out["logits_csc"].max(dim=-1).values.sum()
    )
    target_score.backward()

    panel_attr = (pe.grad.detach().abs() * pe.detach()).squeeze(0).numpy()
    sig_attr = (bs.grad.detach().abs() * bs.detach().abs()).squeeze(0).numpy()

    dis_ref = xai_cases["disease_ref_panel_expr"] if xai_cases else np.zeros((4, 512), dtype=np.float32)
    can_ref = xai_cases["cancer_ref_panel_expr"] if xai_cases else np.zeros((11, 512), dtype=np.float32)

    return {
        "panel_expr": pe.detach().squeeze(0).numpy(),
        "panel_attr": panel_attr,
        "bio_sigs_norm": bs.detach().squeeze(0).numpy(),
        "sig_attr": sig_attr,
        "modality": np.array([int(ds["modality"][idx])], dtype=np.int64),
        "true_celltype": int(ds["cell_type"][idx]),
        "true_health": int(ds["is_diseased"][idx]),
        "true_disease": int(ds["disease_cat"][idx]),
        "true_cancer_bin": int(ds["is_cancer"][idx]),
        "true_cancer_var": int(ds["cancer_var"][idx]),
        "true_csc": int(ds["is_csc"][idx]),
        "prob_celltype": out["prob_celltype"].detach().squeeze(0).numpy(),
        "prob_health": out["prob_health"].detach().squeeze(0).numpy(),
        "prob_disease": out["prob_disease"].detach().squeeze(0).numpy(),
        "prob_cancer_bin": out["prob_cancer_bin"].detach().squeeze(0).numpy(),
        "prob_cancer_var": out["prob_cancer_var"].detach().squeeze(0).numpy(),
        "prob_csc": out["prob_csc"].detach().squeeze(0).numpy(),
        "sim_cell": out["sim_cell"].detach().squeeze(0).numpy(),
        "sim_disease": out["sim_disease"].detach().squeeze(0).numpy(),
        "sim_cancer": out["sim_cancer"].detach().squeeze(0).numpy(),
        "disease_ref_panel_expr": dis_ref,
        "cancer_ref_panel_expr": can_ref,
    }


def compute_pseudobulk_mixture(model, ds, sample_indices: List[int], metadata: Dict, xai_cases) -> Dict:
    """N adet test hücresini Dirichlet ağırlıklarıyla harmanlayarak anlık Pseudo-Bulk dokusu oluşturur."""
    rng = np.random.default_rng(42)
    weights = rng.dirichlet(np.ones(len(sample_indices)))

    # Karışım gen ifadeleri
    lin_counts = np.expm1(np.minimum(ds["panel_expr"][sample_indices], 12.0))
    mixed_lin = np.dot(weights, lin_counts)
    total = mixed_lin.sum()
    if total > 0:
        mixed_lin = (mixed_lin / total) * 1e4
    mixed_log1p = np.log1p(mixed_lin).astype(np.float32)

    # 16 Yolak skoru ortalaması
    mixed_sigs = np.dot(weights, ds["bio_sigs_norm"][sample_indices]).astype(np.float32)

    # Kromozomal token temsilini en ağırlıklı hücreden miras al
    top_cell_idx = sample_indices[int(np.argmax(weights))]
    tg = torch.from_numpy(ds["tokens_genes"][top_cell_idx : top_cell_idx + 1]).long()
    tv = torch.from_numpy(ds["tokens_vals"][top_cell_idx : top_cell_idx + 1]).float()

    pe = torch.from_numpy(mixed_log1p).unsqueeze(0).float().requires_grad_(True)
    bs = torch.from_numpy(mixed_sigs).unsqueeze(0).float().requires_grad_(True)
    mod = torch.tensor([1], dtype=torch.long)  # 1 = Bulk RNA-seq

    model.eval()
    out = model(tg, tv, pe, bs, mod)

    target_score = (
        out["logits_celltype"].max(dim=-1).values.sum()
        + out["logits_disease"].max(dim=-1).values.sum()
        + out["logits_cancer_var"].max(dim=-1).values.sum()
        + out["logits_csc"].max(dim=-1).values.sum()
    )
    target_score.backward()

    panel_attr = (pe.grad.detach().abs() * pe.detach()).squeeze(0).numpy()
    sig_attr = (bs.grad.detach().abs() * bs.detach().abs()).squeeze(0).numpy()

    # Doku içindeki baskın etiketler
    c_counts = np.bincount(ds["cell_type"][sample_indices], minlength=14)
    true_ct = int(np.argmax(c_counts))
    has_diseased = int(np.any(ds["is_diseased"][sample_indices] > 0))
    d_counts = np.bincount(ds["disease_cat"][sample_indices], minlength=4)
    true_ds = int(np.argmax(d_counts))
    has_cancer = int(np.any(ds["is_cancer"][sample_indices] > 0))
    v_counts = np.bincount(ds["cancer_var"][sample_indices], minlength=11)
    true_cv = int(np.argmax(v_counts[1:]) + 1) if has_cancer else 0
    has_csc = int(np.any(ds["is_csc"][sample_indices] > 0))

    dis_ref = xai_cases["disease_ref_panel_expr"] if xai_cases else np.zeros((4, 512), dtype=np.float32)
    can_ref = xai_cases["cancer_ref_panel_expr"] if xai_cases else np.zeros((11, 512), dtype=np.float32)

    return {
        "panel_expr": mixed_log1p,
        "panel_attr": panel_attr,
        "bio_sigs_norm": mixed_sigs,
        "sig_attr": sig_attr,
        "modality": np.array([1]),
        "true_celltype": true_ct,
        "true_health": has_diseased,
        "true_disease": true_ds,
        "true_cancer_bin": has_cancer,
        "true_cancer_var": true_cv,
        "true_csc": has_csc,
        "prob_celltype": out["prob_celltype"].detach().squeeze(0).numpy(),
        "prob_health": out["prob_health"].detach().squeeze(0).numpy(),
        "prob_disease": out["prob_disease"].detach().squeeze(0).numpy(),
        "prob_cancer_bin": out["prob_cancer_bin"].detach().squeeze(0).numpy(),
        "prob_cancer_var": out["prob_cancer_var"].detach().squeeze(0).numpy(),
        "prob_csc": out["prob_csc"].detach().squeeze(0).numpy(),
        "sim_cell": out["sim_cell"].detach().squeeze(0).numpy(),
        "sim_disease": out["sim_disease"].detach().squeeze(0).numpy(),
        "sim_cancer": out["sim_cancer"].detach().squeeze(0).numpy(),
        "disease_ref_panel_expr": dis_ref,
        "cancer_ref_panel_expr": can_ref,
    }


def prepare_xai_dict_for_figure(sample_dict: Dict) -> Dict:
    """Prepares sample dict so every per-sample field can be subscripted by case_idx=0."""
    res = {}
    for k, v in sample_dict.items():
        if k in ("disease_ref_panel_expr", "cancer_ref_panel_expr"):
            res[k] = v
        else:
            arr = np.asarray(v)
            if arr.ndim == 0:
                res[k] = np.array([arr.item()])
            elif k in (
                "panel_expr", "panel_attr", "bio_sigs_norm", "sig_attr",
                "prob_celltype", "prob_health", "prob_disease",
                "prob_cancer_bin", "prob_cancer_var", "prob_csc",
                "sim_cell", "sim_disease", "sim_cancer"
            ):
                if arr.ndim == 1:
                    res[k] = np.expand_dims(arr, axis=0)
                else:
                    res[k] = arr
            else:
                res[k] = arr.flatten()
    return res


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 140) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_image_base64(png_bytes: bytes, caption: str = "", max_width: str = "100%"):
    """Starlette/Uvicorn GZip ASGI medya hatasını aşmak için resmi doğrudan inline Base64 data-URI ile basar."""
    b64_str = base64.b64encode(png_bytes).decode("ascii")
    caption_html = (
        f"<p style='text-align: center; color: #94a3b8; font-size: 0.88rem; margin-top: 6px; font-weight: 500;'>{caption}</p>"
        if caption
        else ""
    )
    st.markdown(
        f"""
        <div style="width: 100%; text-align: center; margin: 10px 0;">
            <img src="data:image/png;base64,{b64_str}" 
                 style="max-width: {max_width}; width: 100%; height: auto; border-radius: 8px; border: 1px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.18);" 
                 alt="{caption or 'Klinik Rapor Görseli'}" />
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_panel_a_figure(rows_info: list, mod_str: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140, facecolor="#ffffff")
    y_pos = np.arange(len(rows_info))[::-1]
    confs = [r[3] for r in rows_info]
    bar_colors = ["#16a34a" if r[4] else "#dc2626" for r in rows_info]

    ax.barh(y_pos, confs, height=0.58, color=bar_colors, alpha=0.88, edgecolor="#0f172a", linewidth=0.8)
    ax.set_xlim(0, 118)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows_info], fontsize=9.5, fontweight="bold", color="#1e293b")
    ax.set_xlabel("Kalibre Edilmiş Model Güven Skoru (%)", fontsize=9.5, fontweight="bold")
    ax.axvline(95.0, color="#0284c7", linestyle="--", linewidth=1.2, label="%95 Hedef Eşiği")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    for idx_r, (stage_lbl, pred_str, true_str, cf, is_ok) in enumerate(rows_info):
        y_v = y_pos[idx_r]
        mark = "DOĞRU" if is_ok else f"Gerçek: {true_str}"
        ax.text(2.5, y_v, f"Tahmin: {pred_str}  [{mark}]", va="center", ha="left", fontsize=9.0, fontweight="bold", color="#ffffff" if cf > 45 else "#0f172a")
        ax.text(cf + 1.5, y_v, f"%{cf:.1f}", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#0f172a")

    ax.set_title(f"PANEL A: Hiyerarşik Klinik Teşhis Sonuçları & Kalibre Güven Yüzdeleri ({mod_str})", fontsize=11.5, fontweight="bold", color="#1e293b", pad=10)
    plt.tight_layout()
    return fig



def plot_panel_b_figure(expr_vec: np.ndarray, attr_vec: np.ndarray, ref_h_vec: np.ndarray, ref_c_vec: np.ndarray, panel_genes: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 4.4), dpi=140, facecolor="#ffffff")
    comb_imp = attr_vec / (attr_vec.max() + 1e-8) + 0.35 * (expr_vec / (expr_vec.max() + 1e-8))
    top_k = 18
    top_idx = np.argsort(comb_imp)[::-1][:top_k]

    h_ref = ref_h_vec[top_idx]
    matched_ref = ref_c_vec[top_idx]
    sample_top_expr = expr_vec[top_idx]
    norm_sal = (comb_imp[top_idx] / (comb_imp[top_idx].max() + 1e-8)) * max(float(sample_top_expr.max()), 2.0)

    mat = np.vstack([sample_top_expr, matched_ref, h_ref, norm_sal])
    im = ax.imshow(mat, aspect="auto", cmap="magma")

    labels = [f"{panel_genes[i]}\n(#{r+1})" if i < len(panel_genes) else f"G{i}" for r, i in enumerate(top_idx)]
    ax.set_xticks(np.arange(top_k))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5, fontweight="bold")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(
        [
            "Hasta İfadesi (log1p)",
            "Eşleşen Kanser/Hastalık Ref",
            "Sağlıklı Kontrol Ref",
            "BiMamba XAI Karar Ağırlığı",
        ],
        fontsize=8.8,
        fontweight="bold",
    )
    ax.set_title("PANEL B: Kararı Belirleyen Top-18 Gen Feature & XAI Önem Isı Haritası (Heatmap)", fontsize=11.5, fontweight="bold", color="#1e293b", pad=10)

    for r_i in range(mat.shape[0]):
        for c_i in range(mat.shape[1]):
            val = mat[r_i, c_i]
            tc = "white" if val < (mat.max() * 0.55) else "black"
            ax.text(c_i, r_i, f"{val:.1f}", ha="center", va="center", fontsize=7.8, color=tc, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


def plot_panel_c_figure(sim_c: np.ndarray, p_cv: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 3.2), dpi=140, facecolor="#ffffff")
    prob_c = p_cv * 100.0
    sim_scaled = np.clip((sim_c + 1.0) * 50.0, 0.0, 100.0)
    mat = np.vstack([prob_c, sim_scaled])

    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    short_c = [
        "Non-Cancer", "Lung LUAD", "NSCLC A549", "Leukemia CML", "Kidney RCC",
        "Endometrial UCEC", "Esophageal ESCA", "Myeloma MYE", "Thyroid THCA", "Ovarian OV", "Pancreas PAAD"
    ]
    ax.set_xticks(np.arange(len(short_c)))
    ax.set_xticklabels(short_c, rotation=35, ha="right", fontsize=8.5, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Model Olasılığı (%)", "Prototip Uyumu (%)"], fontsize=9.0, fontweight="bold")
    ax.set_title("PANEL C: 11 Kanser Varyantı Örüntü Uyum Isı Haritası", fontsize=11.5, fontweight="bold", color="#1e293b", pad=10)

    for r_i in range(2):
        for c_i in range(len(short_c)):
            v = mat[r_i, c_i]
            tc = "white" if v > 55 else "#0f172a"
            ax.text(c_i, r_i, f"{v:.1f}%", ha="center", va="center", fontsize=8.0, fontweight="bold", color=tc)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


def plot_panel_d_figure(sig_z: np.ndarray, sig_attr: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 3.2), dpi=140, facecolor="#ffffff")
    sig_norm = (sig_attr / (sig_attr.max() + 1e-8)) * 3.0
    mat = np.vstack([sig_z, sig_norm])

    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.8)
    short_sig = [
        "CSC Stemness", "Proliferation", "EMT Metastasis", "Hypoxia/Glyco",
        "P53 DNA Repair", "Viral ISG", "MS NeuroInflam", "TAM Angio SPP1",
        "TAM Comp C1QC", "T-Cell Exhaust", "Epithelial Lin", "Neural/Glial",
        "Hematopoietic", "CytoTRACE Ent", "Chr CNV Instab", "Modality Dens"
    ]
    ax.set_xticks(np.arange(len(short_sig)))
    ax.set_xticklabels(short_sig, rotation=35, ha="right", fontsize=8.2, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Hasta Z-Skoru", "XAI Karar Ağırlığı"], fontsize=9.0, fontweight="bold")
    ax.set_title("PANEL D: 16 Biyolojik Yolak, Kök Hücre (CSC) ve Kromozomal İmza Isı Haritası", fontsize=11.5, fontweight="bold", color="#1e293b", pad=10)

    for r_i in range(2):
        for c_i in range(len(short_sig)):
            v = mat[r_i, c_i]
            tc = "white" if abs(v) > 1.4 else "#0f172a"
            ax.text(c_i, r_i, f"{v:+.1f}", ha="center", va="center", fontsize=7.8, fontweight="bold", color=tc)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


def main():

    model, metadata, ds, test_indices, xai_cases = load_resources()

    cell_names = metadata.get("CELL_TYPES", [])
    disease_names = metadata.get("DISEASE_CATEGORIES", [])
    cancer_names = metadata.get("CANCER_VARIANTS", [])
    panel_genes = metadata.get("CORE_PANEL_GENES", [])
    sig_names = metadata.get("SIGNATURE_NAMES", [])

    # =========================================================================
    # ÜST BAŞLIK VE KAGGLE GPU BİLGİ ÇUBUĞU
    # =========================================================================
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-bottom: 16px;">
        <div>
            <h2 style="margin: 0; color: #0f172a;">🧬 scMamba2 + BiMamba Ar-Ge Test ve Karar Destek Tezgahı</h2>
            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.95rem;">
                scRNA-seq ve Bulk RNA-seq Hiyerarşik Teşhis & Açıklanabilir Yapay Zeka (XAI) Test Laboratuvarı
            </p>
        </div>
        <div style="text-align: right;">
            <span class="badge-info">Test Havuzu: 8,486 Görülmemiş Hücre</span>
            <span class="badge-ok" style="margin-left: 6px;">Composite Doğruluk: %98.11</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Kaggle GPU Canlı Durum Bandı (Genişletilebilir)
    tunnel_file = PROJECT_ROOT / "kaggle_bridge" / "active_tunnel.json"
    tunnel_url = None
    if tunnel_file.exists():
        try:
            t_data = json.loads(tunnel_file.read_text(encoding="utf-8"))
            tunnel_url = t_data.get("tunnel_url")
        except Exception:
            pass

    with st.expander("☁️ Kaggle 2x Tesla T4 GPU Köprüsü ve Kota Durumu (Tıkla / Yönet)", expanded=False):
        c_k1, c_k2, c_k3, c_k4 = st.columns([1.5, 1, 1, 1])
        with c_k1:
            st.markdown("**Kaggle Hesabı:** `mithatyasinelikta` | **GPU:** 2x Tesla T4 (32 GB VRAM)")
            if tunnel_url:
                st.markdown(f"**Tünel Durumu:** <span style='color:#10b981; font-weight:bold;'>🟢 AKTİF</span> (`{tunnel_url}`)", unsafe_allow_html=True)
            else:
                st.markdown("**Tünel Durumu:** <span style='color:#64748b;'>⚪ Henüz Başlatılmadı (veya Kapalı)</span>", unsafe_allow_html=True)
        with c_k2:
            if st.button("🔄 Durum & Kotayı Sorgula", use_container_width=True):
                with st.spinner("Kaggle API sorgulanıyor..."):
                    ok, res = run_kaggle_command("status")
                    st.code(res if ok else f"Hata: {res}", language="text")
        with c_k3:
            if st.button("⚡ GPU Sunucusunu Aç (up)", use_container_width=True):
                with st.spinner("Kaggle 2x Tesla T4 GPU sunucusu başlatılıyor ve Cloudflare tüneli kuruluyor (~60-90 sn sürebilir, lütfen bekleyin)..."):
                    ok, res = run_kaggle_command("up", timeout=260)
                    if ok:
                        st.success("GPU sunucusu ve Cloudflare tüneli başarıyla bağlandı!")
                        st.code(res, language="text")
                        st.rerun()
                    else:
                        st.error(f"İşlem çıktısı / Hata:\n{res}")
        with c_k4:
            if st.button("⏹️ GPU'yu Kapat (down)", use_container_width=True):
                with st.spinner("GPU sunucusu kapatılıyor (Sıfır kota israfı)..."):
                    ok, res = run_kaggle_command("down")
                    st.warning("GPU sunucusu kapatıldı. Kota sayacı durduruldu.")
                    st.rerun()


    # =========================================================================
    # SOL PANEL: EĞİTİM DIŞI TEST SETİNDEN ÖRNEK SEÇİCİ
    # =========================================================================
    st.sidebar.markdown("### 🧪 Ar-Ge Test Örnekleyici")
    st.sidebar.markdown("*Eğitimde hiç kullanılmamış (unseen) 8.486 test örneğinden seçim yapın:*")

    sampling_mode = st.sidebar.radio(
        "Test Modu Seçin:",
        [
            "🎲 Tek Hücre Rastgele Çek (Single-Cell)",
            "👥 Hücre Grubu / Bulk Karışımı Oluştur (Pseudo-Bulk)",
            "⭐ Temsilci Klinik Vaka Çalışmaları (8 Vaka)",
        ],
    )

    selected_sample_data = None
    sample_desc = ""

    if sampling_mode == "🎲 Tek Hücre Rastgele Çek (Single-Cell)":
        cat_filter = st.sidebar.selectbox(
            "Biyolojik Kategori Filtresi:",
            [
                "Tümü (Rastgele 8,486 Hücre)",
                "Sağlıklı Kontrol Hücreleri (Normal)",
                "Malign Kanser Hücreleri (Genel)",
                "Kanser Kök Hücreleri (CSC+ Pozitif)",
                "Akciğer Adenokarsinomu (LUAD)",
                "Akut Viral Enfeksiyon (COVID-19)",
                "Multipl Skleroz (MS Nöro-İnflamasyon)",
                "Kronik Miyeloid Lösemi (CML)",
                "Pankreas Kanseri (PAAD)",
                "Böbrek Hücreli Karsinom (KIDNEY)",
            ],
        )

        # Filtreye göre test indekslerini daralt
        if cat_filter == "Sağlıklı Kontrol Hücreleri (Normal)":
            candidate_pool = [i for i in test_indices if ds["disease_cat"][i] == 0]
        elif cat_filter == "Malign Kanser Hücreleri (Genel)":
            candidate_pool = [i for i in test_indices if ds["is_cancer"][i] == 1]
        elif cat_filter == "Kanser Kök Hücreleri (CSC+ Pozitif)":
            candidate_pool = [i for i in test_indices if ds["is_csc"][i] == 1]
        elif cat_filter == "Akciğer Adenokarsinomu (LUAD)":
            candidate_pool = [i for i in test_indices if ds["cancer_var"][i] == 1]
        elif cat_filter == "Akut Viral Enfeksiyon (COVID-19)":
            candidate_pool = [i for i in test_indices if ds["disease_cat"][i] == 1]
        elif cat_filter == "Multipl Skleroz (MS Nöro-İnflamasyon)":
            candidate_pool = [i for i in test_indices if ds["disease_cat"][i] == 2]
        elif cat_filter == "Kronik Miyeloid Lösemi (CML)":
            candidate_pool = [i for i in test_indices if ds["cancer_var"][i] == 3]
        elif cat_filter == "Pankreas Kanseri (PAAD)":
            candidate_pool = [i for i in test_indices if ds["cancer_var"][i] == 10]
        elif cat_filter == "Böbrek Hücreli Karsinom (KIDNEY)":
            candidate_pool = [i for i in test_indices if ds["cancer_var"][i] == 4]
        else:
            candidate_pool = list(test_indices)

        if not candidate_pool:
            candidate_pool = list(test_indices)

        if "random_seed" not in st.session_state:
            st.session_state["random_seed"] = 42

        c_btn1, c_btn2 = st.sidebar.columns([1.2, 1])
        with c_btn1:
            if st.button("🎲 Rastgele Hücre Çek", use_container_width=True):
                st.session_state["random_seed"] = int(np.random.randint(1, 999999))
        with c_btn2:
            st.markdown(f"<small style='color:#64748b;'>Havuz: {len(candidate_pool)}</small>", unsafe_allow_html=True)

        rng = np.random.default_rng(st.session_state["random_seed"])
        chosen_idx = int(rng.choice(candidate_pool))

        slider_idx = st.sidebar.slider(
            "Veya Test İndeksi Seçin:",
            min_value=0,
            max_value=len(candidate_pool) - 1,
            value=candidate_pool.index(chosen_idx) if chosen_idx in candidate_pool else 0,
        )
        final_test_idx = candidate_pool[slider_idx]

        with st.spinner("Model çıkarımı ve BiMamba XAI gradyan analizi hesaplanıyor..."):
            selected_sample_data = compute_sample_xai(model, ds, final_test_idx, metadata, xai_cases)
        sample_desc = f"Tekil Hücre (scRNA-seq) | Test Örnek İndeksi #{final_test_idx}"

    elif sampling_mode == "👥 Hücre Grubu / Bulk Karışımı Oluştur (Pseudo-Bulk)":
        st.sidebar.markdown("Toplu tümör doku biyopsilerini simüle etmek için test hücrelerini karıştırın:")
        n_mix = st.sidebar.slider("Karıştırılacak Hücre Sayısı:", min_value=5, max_value=80, value=25, step=5)
        bulk_filter = st.sidebar.selectbox(
            "Doku Arketipi Tercihi:",
            ["Rastgele Doku Karışımı", "Tümör Mikroçevresi (Kanser Ağırlıklı)", "Enflamatuar Doku (COVID/MS)", "Sağlıklı Referans Doku"],
        )

        if bulk_filter == "Tümör Mikroçevresi (Kanser Ağırlıklı)":
            mix_pool = [i for i in test_indices if ds["is_cancer"][i] == 1]
        elif bulk_filter == "Enflamatuar Doku (COVID/MS)":
            mix_pool = [i for i in test_indices if ds["disease_cat"][i] in (1, 2)]
        elif bulk_filter == "Sağlıklı Referans Doku":
            mix_pool = [i for i in test_indices if ds["disease_cat"][i] == 0]
        else:
            mix_pool = list(test_indices)

        if st.sidebar.button("👥 Yeni Doku Karışımı Sentezle", use_container_width=True):
            st.session_state["bulk_seed"] = int(np.random.randint(1, 999999))

        b_seed = st.session_state.get("bulk_seed", 101)
        rng_b = np.random.default_rng(b_seed)
        chosen_batch = rng_b.choice(mix_pool, size=min(n_mix, len(mix_pool)), replace=len(mix_pool) < n_mix).tolist()

        with st.spinner(f"{n_mix} test hücresinden sentetik doku biyopsisi sentezleniyor..."):
            selected_sample_data = compute_pseudobulk_mixture(model, ds, chosen_batch, metadata, xai_cases)
        sample_desc = f"Yapay Toplu Doku (Bulk RNA-seq) | {n_mix} Hücre Karışımı (Test Seti)"

    else:
        # Temsilci Vaka Çalışmaları
        if xai_cases:
            c_names = [str(x) for x in xai_cases["case_names"]]
            chosen_case_name = st.sidebar.selectbox("Klinik Vaka Seçin:", c_names)
            c_idx = c_names.index(chosen_case_name)
            selected_sample_data = {
                "panel_expr": xai_cases["panel_expr"][c_idx],
                "panel_attr": xai_cases["panel_attr"][c_idx],
                "bio_sigs_norm": xai_cases["bio_sigs_norm"][c_idx],
                "sig_attr": xai_cases["sig_attr"][c_idx],
                "modality": np.array([xai_cases["modality"][c_idx]]),
                "true_celltype": int(xai_cases["true_celltype"][c_idx]),
                "true_health": int(xai_cases["true_health"][c_idx]),
                "true_disease": int(xai_cases["true_disease"][c_idx]),
                "true_cancer_bin": int(xai_cases["true_cancer_bin"][c_idx]),
                "true_cancer_var": int(xai_cases["true_cancer_var"][c_idx]),
                "true_csc": int(xai_cases["true_csc"][c_idx]),
                "prob_celltype": xai_cases["prob_celltype"][c_idx],
                "prob_health": xai_cases["prob_health"][c_idx],
                "prob_disease": xai_cases["prob_disease"][c_idx],
                "prob_cancer_bin": xai_cases["prob_cancer_bin"][c_idx],
                "prob_cancer_var": xai_cases["prob_cancer_var"][c_idx],
                "prob_csc": xai_cases["prob_csc"][c_idx],
                "sim_cell": xai_cases["sim_cell"][c_idx],
                "sim_disease": xai_cases["sim_disease"][c_idx],
                "sim_cancer": xai_cases["sim_cancer"][c_idx],
                "disease_ref_panel_expr": xai_cases["disease_ref_panel_expr"],
                "cancer_ref_panel_expr": xai_cases["cancer_ref_panel_expr"],
            }
            sample_desc = f"Referans Klinik Vaka: {chosen_case_name}"

    if selected_sample_data is None:
        st.warning("Lütfen sol menüden bir test örneği seçin.")
        st.stop()

    # Tahminler ve Gerçek Etiketler
    p_ct = selected_sample_data["prob_celltype"]
    p_hl = selected_sample_data["prob_health"]
    p_ds = selected_sample_data["prob_disease"]
    p_cb = selected_sample_data["prob_cancer_bin"]
    p_cv = selected_sample_data["prob_cancer_var"]
    p_csc = selected_sample_data["prob_csc"]

    pred_ct, conf_ct = int(p_ct.argmax()), float(p_ct.max()) * 100.0
    pred_hl, conf_hl = int(p_hl.argmax()), float(p_hl.max()) * 100.0
    pred_ds, conf_ds = int(p_ds.argmax()), float(p_ds.max()) * 100.0
    pred_cb, conf_cb = int(p_cb.argmax()), float(p_cb.max()) * 100.0
    pred_cv, conf_cv = int(p_cv.argmax()), float(p_cv.max()) * 100.0
    pred_csc, conf_csc = int(p_csc.argmax()), float(p_csc.max()) * 100.0

    true_ct = selected_sample_data["true_celltype"]
    true_hl = selected_sample_data["true_health"]
    true_ds = selected_sample_data["true_disease"]
    true_cb = selected_sample_data["true_cancer_bin"]
    true_cv = selected_sample_data["true_cancer_var"]
    true_csc = selected_sample_data["true_csc"]

    health_map = ["Sağlıklı", "Hastalıklı"]
    cancer_bin_map = ["Kanser Değil", "Malign Kanser"]
    csc_map = ["Negatif (CSC-)", "POZİTİF (Kök Hücre / CSC+)"]

    mod_is_bulk = int(np.asarray(selected_sample_data["modality"]).ravel()[0]) == 1
    mod_str = "BULK RNA-seq (Doku)" if mod_is_bulk else "TEK-HÜCRE (scRNA-seq)"
    mod_badge = "<span class='badge-info'>BULK RNA-seq (Doku)</span>" if mod_is_bulk else "<span class='badge-ok'>TEK-HÜCRE (scRNA-seq)</span>"

    # Merkezi hesaplamalar (Tüm sekmeler için erişilebilir)
    expr_v = selected_sample_data["panel_expr"]
    attr_v = selected_sample_data["panel_attr"]
    comb_imp = attr_v / (attr_v.max() + 1e-8) + 0.35 * (expr_v / (expr_v.max() + 1e-8))

    tasks_info = [
        ("1. Hücre Tipi Tahmini", cell_names[pred_ct], cell_names[true_ct], conf_ct, pred_ct == true_ct),
        ("4a. Sağlık Durumu", health_map[pred_hl], health_map[true_hl], conf_hl, pred_hl == true_hl),
        ("4b. Hastalık Arketipi", disease_names[pred_ds], disease_names[true_ds], conf_ds, pred_ds == true_ds),
        ("4c. Malignite / Kanser", cancer_bin_map[pred_cb], cancer_bin_map[true_cb], conf_cb, pred_cb == true_cb),
        ("4d. Kanser Varyantı", cancer_names[pred_cv], cancer_names[true_cv], conf_cv, pred_cv == true_cv),
        ("4d-Ek. Kanser Kök Hücresi", csc_map[pred_csc], csc_map[true_csc], conf_csc, pred_csc == true_csc),
    ]
    correct_count = sum(1 for t in tasks_info if t[4])

    # =========================================================================
    # ANA İÇERİK SEKMELERİ
    # =========================================================================
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📋 1. Tahmin ve Doğruluk Kıyaslaması",
            "🔍 2. Tahmin Neye Göre Alındı? (XAI Kanıtları)",
            "📊 3. 4-Panelli Dinamik Teşhis Heatmap'i",
            "⚡ 4. Toplu Stres Testi (Batch Validation)",
        ]
    )

    # -------------------------------------------------------------------------
    # SEKME 1: TAHMİN SONUÇLARI VE GERÇEK ETİKET DOĞRULAMASI
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown(f"#### 🔬 İncelenen Örnek: **{sample_desc}**")
        st.markdown(f"Girdi Formatı: {mod_badge} | 512 Master Gen Paneli İfade Seviyeleri İncelendi", unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        cols_tab1 = [c1, c2, c3, c4, c5, c6]

        for col, (title, pred_val, true_val, conf, is_ok) in zip(cols_tab1, tasks_info):
            badge = "<span class='badge-ok'>✅ DOĞRU</span>" if is_ok else "<span class='badge-err'>❌ YANLIŞ</span>"
            border_color = "#86efac" if is_ok else "#fca5a5"
            with col:
                st.markdown(f"""
                <div class=\"decision-card\" style=\"border-left: 4px solid {border_color};\">
                    <div style=\"display:flex; justify-content:space-between; align-items:center;\">
                        <span class=\"card-title\">{title}</span>
                        {badge}
                    </div>
                    <div class=\"card-pred\">{pred_val}</div>
                    <div class=\"card-true\"><b>Gerçek:</b> {true_val}</div>
                    <div class=\"card-conf\">Kalibre Güven: %{conf:.1f}</div>
                </div>
                """, unsafe_allow_html=True)


        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if correct_count == 6:
            st.success("🎉 **Mükemmel Teşhis:** Model tüm 6 hiyerarşik karar aşamasını (6/6) eğitim dışı test örneği üzerinde %100 doğrulukla bildi!")
        else:
            st.warning(f"⚠️ **Hiyerarşik Uyum:** Model 6 karardan {correct_count} tanesini doğru bildi. Biyolojik alt varyant ve etki detayları için 2. sekmeyi inceleyin.")

    # -------------------------------------------------------------------------
    # SEKME 2: TAHMİN NEYE GÖRE ALINDI? (XAI KANITLARI)
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### 🔍 Karar Gerekçeleri ve Biyolojik Kanıtlar (XAI)")
        st.markdown("Modelin bu hiyerarşik teşhisi koyarken **hangi genlerin ifadesine ve hangi biyolojik yolaklara dayandığı** aşağıda listelenmiştir:")

        col_x1, col_x2 = st.columns([1.3, 1])

        with col_x1:
            st.markdown("#### 🧬 Kararı Belirleyen Top-15 Gen (BiMamba Input x Gradient)")
            expr_v = selected_sample_data["panel_expr"]
            attr_v = selected_sample_data["panel_attr"]
            comb_imp = attr_v / (attr_v.max() + 1e-8) + 0.35 * (expr_v / (expr_v.max() + 1e-8))
            top_genes_idx = np.argsort(comb_imp)[::-1][:15]

            df_genes = []
            for rank, g_i in enumerate(top_genes_idx, 1):
                g_name = panel_genes[g_i] if g_i < len(panel_genes) else f"Gene_{g_i}"
                p_exp = expr_v[g_i]
                h_ref = selected_sample_data["disease_ref_panel_expr"][0][g_i] if len(selected_sample_data["disease_ref_panel_expr"]) > 0 else 0.0
                role = GENE_ANNOTATIONS.get(g_name, "Genomik Tanı Paneli Ayırt Edici Biyobelirteci")
                df_genes.append({
                    "Sıra": f"#{rank}",
                    "Gen Sembolü": g_name,
                    "Hasta İfadesi (log1p)": f"{p_exp:.2f}",
                    "Sağlıklı Ref (log1p)": f"{h_ref:.2f}",
                    "BiMamba Karar Ağırlığı": f"{comb_imp[g_i]:.3f}",
                    "Biyolojik Rolü / Kanıt Açıklaması": role,
                })
            st.dataframe(pd.DataFrame(df_genes), use_container_width=True, hide_index=True)

        with col_x2:
            st.markdown("#### 🎯 11 Kanser Varyantı Prototip Uyumu (%)")
            sim_c = selected_sample_data["sim_cancer"]
            sim_c_pct = np.clip((sim_c + 1.0) * 50.0, 0.0, 100.0)
            df_proto = pd.DataFrame({
                "Kanser Varyantı": [c.replace("_", " ") for c in cancer_names],
                "Örüntü Benzerliği (%)": sim_c_pct,
                "Model Olasılığı (%)": p_cv * 100.0,
            }).sort_values(by="Model Olasılığı (%)", ascending=False)
            st.dataframe(
                df_proto.style.format({"Örüntü Benzerliği (%)": "{:.1f}%", "Model Olasılığı (%)": "{:.1f}%"})
                .bar(subset=["Model Olasılığı (%)"], color="#3b82f6", vmin=0, vmax=100)
                .bar(subset=["Örüntü Benzerliği (%)"], color="#f97316", vmin=0, vmax=100),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        st.markdown("#### 🧪 16 Biyolojik Yolak & Kök Hücre (CSC) Aktivasyon Sapmaları")
        sigs_z = selected_sample_data["bio_sigs_norm"]
        short_sig_names = [
            "CSC Kök Hücre İmzası", "Malign Proliferasyon", "EMT & Metastaz", "Hipoksi & Glikoliz",
            "P53 DNA Onarımı", "Viral İnterferon (COVID)", "MS Nöro-İnflamasyon", "Anjiyojenik TAM (SPP1)",
            "Komplement TAM (C1QC)", "T-Lenfosit Sitotoksisite", "Epitelyal Karsinom", "Nöral / Glial Soyu",
            "Hematopoietik Soyu", "CytoTRACE Entropisi", "Kromozomal CNV Sapması", "Modalite Yoğunluğu"
        ]
        fig_bar, ax_bar = plt.subplots(figsize=(14, 3.8), facecolor="#ffffff")
        bar_colors = ["#dc2626" if abs(v) > 1.5 else ("#2563eb" if v > 0 else "#64748b") for v in sigs_z]
        ax_bar.bar(short_sig_names, sigs_z, color=bar_colors, alpha=0.85, edgecolor="#1e293b", linewidth=0.6)
        ax_bar.axhline(0, color="#334155", linewidth=0.8)
        ax_bar.axhline(1.5, color="#dc2626", linestyle="--", alpha=0.5, label="Yüksek Aktivasyon (+1.5 Z)")
        ax_bar.axhline(-1.5, color="#2563eb", linestyle="--", alpha=0.5, label="Baskılanma (-1.5 Z)")
        ax_bar.set_ylabel("Hasta Z-Skoru (Sapma)", fontweight="bold")
        plt.xticks(rotation=35, ha="right", fontsize=8.5, fontweight="bold")
        ax_bar.grid(axis="y", linestyle=":", alpha=0.5)
        render_image_base64(fig_to_png_bytes(fig_bar), caption="16 Biyolojik Yolak Aktivasyon Z-Skorları (Sapma Dağılımı)")

    # -------------------------------------------------------------------------
    # SEKME 3: 4-PANELLİ DİNAMİK TEŞHİS HEATMAP'İ
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("### 📊 4-Panelli Klinik Teşhis ve Açıklanabilir Yapay Zeka (XAI) Raporu")
        st.markdown("Aşağıdaki panellerden her birinin hem **yüksek çözünürlüklü ısı haritasını / grafiğini** hem de **ayrıntılı veri tablosunu** doğrudan inceleyebilirsiniz:")

        # Master poster oluştur ve diske kaydet
        fig_data = prepare_xai_dict_for_figure(selected_sample_data)
        heatmap_fig = build_single_case_figure(
            case_idx=0,
            case_name=sample_desc,
            xai_data=fig_data,
            metadata=metadata,
        )
        report_png_file = PROJECT_ROOT / "latest_diagnostic_report.png"
        heatmap_fig.savefig(str(report_png_file), format="png", dpi=140, bbox_inches="tight")
        plt.close(heatmap_fig)
        png_bytes = report_png_file.read_bytes()

        # 1. BÖLÜM: 4-Panelli Birleşik Master Poster (Doğrudan ve Eksiksiz Görünür)
        render_image_base64(
            png_bytes,
            caption=f"scMamba2 + BiMamba 4-Panelli Klinik Teşhis & XAI Master Raporu ({sample_desc})",
        )

        col_dl1, col_dl2 = st.columns([2, 1])
        with col_dl1:
            st.info("💡 **Görsel Açıklaması:** Yukarıdaki master poster; Panel A (Teşhis/Güven), Panel B (Top-18 Biyobelirteç Geni), Panel C (11 Kanser Varyant Uyumu) ve Panel D (16 Biyolojik Yolak/CSC) analizlerini tek parça yüksek çözünürlükte (3000x2025 px) sunar.")
        with col_dl2:
            st.download_button(
                label="📥 4-Panelli Master Posteri PNG İndir",
                data=png_bytes,
                file_name=f"diagnostic_report_{sample_desc.replace(' ', '_').replace('/', '_')}.png",
                mime="image/png",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### 🔬 Panel Bazlı Derinlemesine Grafikler ve Sayısal Veri Tabloları")

        view_mode = st.radio(
            "Görüntüleme Modu:",
            [
                "📌 Tüm Panelleri ve Tabloları Alt Alta Göster (Eksiksiz Tam Rapor)",
                "📋 Yalnızca Panel A (Hiyerarşik Teşhis & Güven Çubukları)",
                "🧬 Yalnızca Panel B (Top-18 Biyobelirteç Geni & XAI Ağırlıkları)",
                "🎯 Yalnızca Panel C (11 Kanser Varyantı Örüntü Uyumu)",
                "🧪 Yalnızca Panel D (16 Biyolojik Yolak Z-Skorları & CSC İmzaları)",
            ],
            horizontal=True,
        )

        # Panel A
        if view_mode in ("📌 Tüm Panelleri ve Tabloları Alt Alta Göster (Eksiksiz Tam Rapor)", "📋 Yalnızca Panel A (Hiyerarşik Teşhis & Güven Çubukları)"):
            st.markdown("#### 1. Panel A: Hiyerarşik Klinik Teşhis Sonuçları & Kalibre Güven Çubukları")
            fig_a = plot_panel_a_figure(tasks_info, mod_str)
            render_image_base64(fig_to_png_bytes(fig_a), caption="1. Panel A: Hiyerarşik Klinik Teşhis Sonuçları & Kalibre Güven Çubukları")

            st.markdown("##### 📋 Panel A Sayısal Teşhis ve Güven Tablosu")
            df_panel_a = pd.DataFrame([
                {
                    "Teşhis Seviyesi": "1. Hücre Tipi (Cell Type)",
                    "Model Tahmini": cell_names[pred_ct],
                    "Gerçek Etiket (Ground Truth)": cell_names[true_ct],
                    "Model Güven Skoru (%)": f"%{conf_ct:.1f}",
                    "Doğruluk Durumu": "✅ Doğru" if pred_ct == true_ct else "❌ Hatalı",
                },
                {
                    "Teşhis Seviyesi": "4a. Sağlık Durumu (Health Status)",
                    "Model Tahmini": health_map[pred_hl],
                    "Gerçek Etiket (Ground Truth)": health_map[true_hl],
                    "Model Güven Skoru (%)": f"%{conf_hl:.1f}",
                    "Doğruluk Durumu": "✅ Doğru" if pred_hl == true_hl else "❌ Hatalı",
                },
                {
                    "Teşhis Seviyesi": "4b. Hastalık Arketipi (Disease Category)",
                    "Model Tahmini": disease_names[pred_ds],
                    "Gerçek Etiket (Ground Truth)": disease_names[true_ds],
                    "Model Güven Skoru (%)": f"%{conf_ds:.1f}",
                    "Doğruluk Durumu": "✅ Doğru" if pred_ds == true_ds else "❌ Hatalı",
                },
                {
                    "Teşhis Seviyesi": "4c. Malignite / Kanser mi?",
                    "Model Tahmini": cancer_bin_map[pred_cb],
                    "Gerçek Etiket (Ground Truth)": cancer_bin_map[true_cb],
                    "Model Güven Skoru (%)": f"%{conf_cb:.1f}",
                    "Doğruluk Durumu": "✅ Doğru" if pred_cb == true_cb else "❌ Hatalı",
                },
                {
                    "Teşhis Seviyesi": "4d. Kanser Varyantı (11 Tipli)",
                    "Model Tahmini": cancer_names[pred_cv],
                    "Gerçek Etiket (Ground Truth)": cancer_names[true_cv],
                    "Model Güven Skoru (%)": f"%{conf_cv:.1f}",
                    "Doğruluk Durumu": "✅ Doğru" if pred_cv == true_cv else "❌ Hatalı",
                },
                {
                    "Teşhis Seviyesi": "4d-Ek. Kanser Kök Hücresi (CSC)",
                    "Model Tahmini": csc_map[pred_csc],
                    "Gerçek Etiket (Ground Truth)": csc_map[true_csc],
                    "Model Güven Skoru (%)": f"%{conf_csc:.1f}",
                    "Doğruluk Durumu": "✅ Doğru" if pred_csc == true_csc else "❌ Hatalı",
                },
            ])
            st.dataframe(df_panel_a, use_container_width=True, hide_index=True)
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Panel B
        if view_mode in ("📌 Tüm Panelleri ve Tabloları Alt Alta Göster (Eksiksiz Tam Rapor)", "🧬 Yalnızca Panel B (Top-18 Biyobelirteç Geni & XAI Ağırlıkları)"):
            st.markdown("#### 2. Panel B: Kararı Belirleyen Top-18 Biyobelirteç Gen Feature & XAI Isı Haritası (Heatmap)")
            top18_idx = np.argsort(comb_imp)[-18:][::-1]
            ref_h = selected_sample_data["disease_ref_panel_expr"][0] if len(selected_sample_data["disease_ref_panel_expr"]) > 0 else np.zeros(512)
            ref_c = selected_sample_data["cancer_ref_panel_expr"][pred_cv] if len(selected_sample_data["cancer_ref_panel_expr"]) > pred_cv else np.zeros(512)
            fig_b = plot_panel_b_figure(selected_sample_data["panel_expr"], selected_sample_data["panel_attr"], ref_h, ref_c, panel_genes)
            render_image_base64(fig_to_png_bytes(fig_b), caption="2. Panel B: Kararı Belirleyen Top-18 Biyobelirteç Gen Feature & XAI Isı Haritası (Heatmap)")

            st.markdown("##### 🧬 Panel B Top-18 Biyobelirteç Gen Detay ve Fonksiyonel Tanım Tablosu")
            df_panel_b = []
            for rank, g_i in enumerate(top18_idx, 1):
                g_sym = panel_genes[g_i] if g_i < len(panel_genes) else f"Gene_{g_i}"
                p_val = float(expr_v[g_i])
                h_val = float(ref_h[g_i])
                c_val = float(ref_c[g_i])
                diff_healthy = p_val - h_val
                status_str = "🔺 Yüksek (+)" if diff_healthy > 0.5 else ("🔻 Baskılanmış (-)" if diff_healthy < -0.5 else "↔️ Normal")
                df_panel_b.append({
                    "Sıra": f"#{rank}",
                    "Gen Sembolü": g_sym,
                    "Hasta İfadesi (log1p)": f"{p_val:.2f}",
                    "Sağlıklı Ref (log1p)": f"{h_val:.2f}",
                    "Kanser Ref (log1p)": f"{c_val:.2f}",
                    "Sağlıklıya Göre Fark": f"{diff_healthy:+.2f}",
                    "İfade Durumu": status_str,
                    "BiMamba Karar Ağırlığı": f"{comb_imp[g_i]:.4f}",
                    "Klinik Açıklama": GENE_ANNOTATIONS.get(g_sym, "Kritik genomik ayırt edici biyobelirteç geni"),
                })
            st.dataframe(pd.DataFrame(df_panel_b), use_container_width=True, hide_index=True)
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Panel C
        if view_mode in ("📌 Tüm Panelleri ve Tabloları Alt Alta Göster (Eksiksiz Tam Rapor)", "🎯 Yalnızca Panel C (11 Kanser Varyantı Örüntü Uyumu)"):
            st.markdown("#### 3. Panel C: Hastalık & 11 Kanser Varyantı Prototip Örüntü Uyum Isı Haritası")
            fig_c = plot_panel_c_figure(selected_sample_data["sim_cancer"], p_cv)
            render_image_base64(fig_to_png_bytes(fig_c), caption="3. Panel C: Hastalık & 11 Kanser Varyantı Prototip Örüntü Uyum Isı Haritası")

            st.markdown("##### 🎯 Panel C Varyant Uyum ve Model Olasılıkları Tablosu")
            sim_c = selected_sample_data["sim_cancer"]
            sim_pct = np.clip((sim_c + 1.0) * 50.0, 0.0, 100.0)
            df_panel_c = []
            for idx_c, c_name in enumerate(cancer_names):
                is_predicted = (idx_c == pred_cv)
                is_true = (idx_c == true_cv)
                tag = "⭐ TAHMİN EDİLEN VARYANT" if is_predicted else ("🎯 GERÇEK VARYANT" if is_true else "-")
                df_panel_c.append({
                    "Kanser Varyantı": c_name.replace("_", " "),
                    "Örüntü Benzerliği (%)": sim_pct[idx_c],
                    "Model Tahmin Olasılığı (%)": p_cv[idx_c] * 100.0,
                    "Durum": tag,
                })
            df_c_sorted = pd.DataFrame(df_panel_c).sort_values(by="Model Tahmin Olasılığı (%)", ascending=False)
            st.dataframe(
                df_c_sorted.style.format({"Örüntü Benzerliği (%)": "{:.1f}%", "Model Tahmin Olasılığı (%)": "{:.1f}%"})
                .bar(subset=["Model Tahmin Olasılığı (%)"], color="#3b82f6", vmin=0, vmax=100)
                .bar(subset=["Örüntü Benzerliği (%)"], color="#f97316", vmin=0, vmax=100),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Panel D
        if view_mode in ("📌 Tüm Panelleri ve Tabloları Alt Alta Göster (Eksiksiz Tam Rapor)", "🧪 Yalnızca Panel D (16 Biyolojik Yolak Z-Skorları & CSC İmzaları)"):
            st.markdown("#### 4. Panel D: 16 Biyolojik Yolak, Kök Hücre (CSC) ve Kromozomal İmza Isı Haritası")
            fig_d = plot_panel_d_figure(selected_sample_data["bio_sigs_norm"], selected_sample_data["sig_attr"])
            render_image_base64(fig_to_png_bytes(fig_d), caption="4. Panel D: 16 Biyolojik Yolak, Kök Hücre (CSC) ve Kromozomal İmza Isı Haritası")

            st.markdown("##### 🧪 Panel D 16 Biyolojik Yolak Z-Skorları ve Patolojik Aktivasyon Tablosu")
            sig_z = selected_sample_data["bio_sigs_norm"]
            df_panel_d = []
            for idx_s, s_val in enumerate(sig_z):
                s_name = sig_names[idx_s] if idx_s < len(sig_names) else f"Sig_{idx_s}"
                if s_val > 1.5:
                    status_text = "🔴 YÜKSEK AKTİVASYON (Patolojik Artış)"
                elif s_val < -1.5:
                    status_text = "🔵 BASKILANMIŞ (İnhibisyon / Kayıp)"
                else:
                    status_text = "⚪ Normal Biyolojik Aralık"
                df_panel_d.append({
                    "Yolak / Biyolojik İmza": s_name,
                    "Hasta Z-Skoru (Sapma)": float(s_val),
                    "Yolak Durumu": status_text,
                })
            df_d_sorted = pd.DataFrame(df_panel_d).sort_values(by="Hasta Z-Skoru (Sapma)", ascending=False)
            st.dataframe(
                df_d_sorted.style.format({"Hasta Z-Skoru (Sapma)": "{:+.2f}"})
                .bar(subset=["Hasta Z-Skoru (Sapma)"], color="#ef4444", vmin=-3.0, vmax=3.0),
                use_container_width=True,
                hide_index=True,
            )



    # -------------------------------------------------------------------------
    # SEKME 4: TOPLU STRES TESTİ (BATCH VALIDATION BENCHMARKER)
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("### ⚡ Toplu Stres Testi (Batch Validation Benchmarker)")
        st.markdown("Eğitimde hiç görülmemiş **8,486 test hücresi** içinden rastgele bir grubu tek seferde test edin ve modelin doğruluk başarısını canlı ölçün:")

        col_b1, col_b2, col_b3 = st.columns([1, 1, 1.2])
        with col_b1:
            batch_n = st.selectbox("Test Edilecek Rastgele Hücre Sayısı:", [20, 50, 100, 200], index=1)
        with col_b2:
            batch_cat = st.selectbox(
                "Test Grubu Filtresi:",
                ["Tüm Test Seti (Karışık)", "Yalnızca Kanserli Hücreler", "Yalnızca Kanser Kök Hücreleri (CSC+)", "Yalnızca Sağlıklı Kontroller"],
            )
        with col_b3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            start_batch = st.button("🚀 Toplu Testi Başlat (Benchmark)", use_container_width=True)

        if start_batch:
            if batch_cat == "Yalnızca Kanserli Hücreler":
                pool_b = [i for i in test_indices if ds["is_cancer"][i] == 1]
            elif batch_cat == "Yalnızca Kanser Kök Hücreleri (CSC+)":
                pool_b = [i for i in test_indices if ds["is_csc"][i] == 1]
            elif batch_cat == "Yalnızca Sağlıklı Kontroller":
                pool_b = [i for i in test_indices if ds["disease_cat"][i] == 0]
            else:
                pool_b = list(test_indices)

            rng_batch = np.random.default_rng()
            sampled_b = rng_batch.choice(pool_b, size=min(batch_n, len(pool_b)), replace=False).tolist()

            progress_bar = st.progress(0)
            status_text = st.empty()

            records = []
            model.eval()

            tg_b = torch.from_numpy(ds["tokens_genes"][sampled_b]).long()
            tv_b = torch.from_numpy(ds["tokens_vals"][sampled_b]).float()
            pe_b = torch.from_numpy(ds["panel_expr"][sampled_b]).float()
            bs_b = torch.from_numpy(ds["bio_sigs_norm"][sampled_b]).float()
            mod_b = torch.from_numpy(ds["modality"][sampled_b]).long()

            t_start = time.time()
            with torch.no_grad():
                out_b = model(tg_b, tv_b, pe_b, bs_b, mod_b)
            t_elapsed = time.time() - t_start

            p_ct_b = out_b["prob_celltype"].argmax(dim=-1).numpy()
            p_hl_b = out_b["prob_health"].argmax(dim=-1).numpy()
            p_ds_b = out_b["prob_disease"].argmax(dim=-1).numpy()
            p_cb_b = out_b["prob_cancer_bin"].argmax(dim=-1).numpy()
            p_cv_b = out_b["prob_cancer_var"].argmax(dim=-1).numpy()
            p_csc_b = out_b["prob_csc"].argmax(dim=-1).numpy()

            t_ct_b = ds["cell_type"][sampled_b]
            t_hl_b = ds["is_diseased"][sampled_b]
            t_ds_b = ds["disease_cat"][sampled_b]
            t_cb_b = ds["is_cancer"][sampled_b]
            t_cv_b = ds["cancer_var"][sampled_b]
            t_csc_b = ds["is_csc"][sampled_b]

            acc_ct = (p_ct_b == t_ct_b).mean() * 100.0
            acc_hl = (p_hl_b == t_hl_b).mean() * 100.0
            acc_ds = (p_ds_b == t_ds_b).mean() * 100.0
            acc_cb = (p_cb_b == t_cb_b).mean() * 100.0
            acc_cv = (p_cv_b == t_cv_b).mean() * 100.0
            acc_csc = (p_csc_b == t_csc_b).mean() * 100.0
            comp_acc = np.mean([acc_ct, acc_hl, acc_ds, acc_cb, acc_cv, acc_csc])

            progress_bar.progress(100)
            status_text.success(f"Tamamlandı: {len(sampled_b)} test hücresi {t_elapsed*1000:.1f} ms içinde değerlendirildi.")

            st.markdown(f"#### 🏆 Toplu Test Sonuç Özeti ({len(sampled_b)} Rastgele Test Hücresi)")
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            m1.metric("Birleşik Doğruluk", f"%{comp_acc:.1f}")
            m2.metric("1. Hücre Tipi", f"%{acc_ct:.1f}")
            m3.metric("4a. Sağlık/Hastalık", f"%{acc_hl:.1f}")
            m4.metric("4b. Hastalık Arketipi", f"%{acc_ds:.1f}")
            m5.metric("4c. Kanser mi?", f"%{acc_cb:.1f}")
            m6.metric("4d. Kanser Varyantı", f"%{acc_cv:.1f}")
            m7.metric("4d-Ek. Kök Hücre (CSC)", f"%{acc_csc:.1f}")

            table_rows = []
            for i, idx_s in enumerate(sampled_b):
                is_all_ok = (
                    p_ct_b[i] == t_ct_b[i]
                    and p_hl_b[i] == t_hl_b[i]
                    and p_ds_b[i] == t_ds_b[i]
                    and p_cb_b[i] == t_cb_b[i]
                    and p_cv_b[i] == t_cv_b[i]
                    and p_csc_b[i] == t_csc_b[i]
                )
                table_rows.append({
                    "Test İndeksi": idx_s,
                    "Gerçek Hücre Tipi": cell_names[t_ct_b[i]],
                    "Tahmin Hücre Tipi": cell_names[p_ct_b[i]],
                    "Gerçek Hastalık": disease_names[t_ds_b[i]],
                    "Tahmin Hastalık": disease_names[p_ds_b[i]],
                    "Gerçek Varyant": cancer_names[t_cv_b[i]],
                    "Tahmin Varyant": cancer_names[p_cv_b[i]],
                    "Gerçek CSC": "CSC+" if t_csc_b[i] else "CSC-",
                    "Tahmin CSC": "CSC+" if p_csc_b[i] else "CSC-",
                    "Tüm Teşhis Doğru mu?": "✅ DOĞRU" if is_all_ok else "⚠️ KISMİ HATA",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
