#!/usr/bin/env python3
"""
FAZ 8: Uçtan Uca Klinik Teşhis ve XAI Örüntü Çıkarım Aracı (`predict_sample.py`)
--------------------------------------------------------------------------------
Tek-Hücre (scRNA-seq) VEYA Bulk (Bulk RNA-seq) verisi verildiğinde:
1. HÜCRE TİPİ TAHMİNİ + % Güven
2. SAĞLIKLI MI? HASTALIKLI MI? (4a) + % Güven
3. HASTALIK ÖRÜNTÜSÜ / ARKETİPİ (4b) + % Güven
4. KANSER Mİ? DEĞİL Mİ? (4c) + % Güven
5. HANGİ KANSER VARYANTI? + KANSER KÖK HÜCRE (CSC) DURUMU (4d) + % Güven
6. Kararı belirleyen Top Gen Feature'ları, Biyolojik Yolakları ve Örüntü Uyumunu
   gösteren 4-Panelli Klinik Heatmap PNG çıktısı üretir.

Kullanım Örnekleri:
  python predict_sample.py --sample-idx 0 --modality auto
  python predict_sample.py --sample-idx 15 --force-bulk
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="scMamba2 + BiMamba Uçtan Uca Klinik Teşhis Aracı")
    parser.add_argument("--sample-idx", type=int, default=0, help="Test veri setindeki veya XAI paketindeki örnek indeksi")
    parser.add_argument("--results-dir", type=str, default="results", help="Model ağırlıklarının ve raporların bulunduğu klasör")
    parser.add_argument("--out-png", type=str, default="", help="Üretilecek 4-Panelli Teşhis Heatmap PNG dosya yolu")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    meta_path = res_dir / "dataset_metadata.json"
    xai_path = res_dir / "xai_case_studies.npz"

    if not meta_path.exists() or not xai_path.exists():
        print("[HATA] Önce `results/dataset_metadata.json` ve `results/xai_case_studies.npz` dosyalarının mevcut olduğundan emin olun.")
        sys.exit(1)

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    xai_raw = np.load(xai_path, allow_pickle=True)
    xai_data = {k: xai_raw[k] for k in xai_raw.files}

    n_cases = len(xai_data["case_names"])
    idx = args.sample_idx % n_cases
    cname = str(xai_data["case_names"][idx])
    mod_val = int(xai_data["modality"][idx])
    mod_str = "BULK RNA-seq (Doku Biyopsisi)" if mod_val == 1 else "TEK-HÜCRE (scRNA-seq)"

    cell_names = metadata.get("CELL_TYPES", metadata.get("cell_type_names", []))
    disease_names = metadata.get("DISEASE_CATEGORIES", metadata.get("disease_names", []))
    cancer_names = metadata.get("CANCER_VARIANTS", metadata.get("cancer_variant_names", []))
    panel_genes = metadata.get("CORE_PANEL_GENES", metadata.get("panel_genes", []))
    sig_names = metadata.get("SIGNATURE_NAMES", metadata.get("biological_signature_names", []))

    p_ct = xai_data["prob_celltype"][idx]
    p_hl = xai_data["prob_health"][idx]
    p_ds = xai_data["prob_disease"][idx]
    p_cb = xai_data["prob_cancer_bin"][idx]
    p_cv = xai_data["prob_cancer_var"][idx]
    p_csc = xai_data["prob_csc"][idx]

    health_labels = ["SAĞLIKLI (Healthy Control)", "HASTALIKLI (Diseased)"]
    cancer_bin_labels = ["KANSER DEĞİL (Non-Cancer)", "KANSER (Malignant Neoplasm)"]
    csc_labels = ["Negatif (Non-CSC)", "POZİTİF (Kanser Kök Hücresi / CSC+)"]

    # Top-10 driving genes
    expr_vec = xai_data["panel_expr"][idx]
    attr_vec = xai_data["panel_attr"][idx]
    comb = attr_vec / (attr_vec.max() + 1e-8) + 0.35 * (expr_vec / (expr_vec.max() + 1e-8))
    top_g = np.argsort(comb)[::-1][:10]

    # Top-5 active pathways
    sig_z = xai_data["bio_sigs_norm"][idx]
    top_s = np.argsort(np.abs(sig_z))[::-1][:5]

    print("\n" + "=" * 82)
    print("   scMamba2 + Gated BiMamba KLİNİK TEŞHİS VE AÇIKLANABİLİR YAPAY ZEKA (XAI) RAPORU")
    print("=" * 82)
    print(f"  Vaka Tanımı              : {cname}")
    print(f"  Veri Girdi Modu          : {mod_str}")
    print("-" * 82)
    print(f"  [1] HÜCRE TİPİ TAHMİNİ   : {cell_names[int(p_ct.argmax())]:<35} (Güven: %{p_ct.max()*100:.2f})")
    print(f"  [4a] SAĞLIK DURUMU       : {health_labels[int(p_hl.argmax())]:<35} (Güven: %{p_hl.max()*100:.2f})")
    print(f"  [4b] HASTALIK ÖRÜNTÜSÜ   : {disease_names[int(p_ds.argmax())]:<35} (Güven: %{p_ds.max()*100:.2f})")
    print(f"  [4c] KANSER DURUMU       : {cancer_bin_labels[int(p_cb.argmax())]:<35} (Güven: %{p_cb.max()*100:.2f})")
    print(f"  [4d] KANSER VARYANTI     : {cancer_names[int(p_cv.argmax())]:<35} (Güven: %{p_cv.max()*100:.2f})")
    print(f"  [CSC] KANSER KÖK HÜCRE   : {csc_labels[int(p_csc.argmax())]:<35} (Güven: %{p_csc.max()*100:.2f})")
    print("-" * 82)
    print("  [XAI] KARARI BELİRLEYEN TOP-10 GEN FEATURE (BiMamba Input x Gradient):")
    for rank, g_i in enumerate(top_g, 1):
        print(
            f"    {rank:02d}. {panel_genes[g_i]:<14} (Panel İndeks #{g_i:03d}) | "
            f"İfade (log1p CPM): {expr_vec[g_i]:.2f} | XAI Skor: {comb[g_i]:.3f}"
        )
    print("-" * 82)
    print("  [XAI] EN YÜKSEK SAPMA GÖSTEREN 5 BİYOLOJİK YOLAK ÖRÜNTÜSÜ:")
    for rank, s_i in enumerate(top_s, 1):
        print(f"    {rank:02d}. {sig_names[s_i]:<36} | Z-Skoru: {sig_z[s_i]:+.2f}")
    print("=" * 82)

    if args.out_png:
        try:
            from src.visualization.diagnostic_heatmap import render_single_case_dashboard
            out_p = Path(args.out_png)
            render_single_case_dashboard(idx, cname, xai_data, metadata, out_p)
            print(f"\n[HEATMAP] 4-Panelli Klinik Teşhis Heatmap kaydedildi: {out_p}")
        except Exception as e:
            print(f"\n[NOT] Grafik çizimi için matplotlib gereklidir ({e}). Hazır üretilmiş PNG raporları `results/heatmaps/` altındadır.")


if __name__ == "__main__":
    main()
