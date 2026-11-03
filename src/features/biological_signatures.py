"""
FAZ 2: Biyolojik İmza (Signature), Kromozomal Sıralama ve Kalite Kontrol Modülü
-------------------------------------------------------------------------------
Referans Makaleler:
1. Gulati et al., Science (2020) - CytoTRACE: Transcriptional diversity/entropy predicts differentiation & stemness.
2. Tirosh et al., Science (2016) & Patel et al., Science (2014) - Chromosomal CNV & CSC stemness in single-cell RNA-seq.
3. Cheng et al., Cell (2021) - A pan-cancer single-cell transcriptional atlas of tumor infiltrating myeloid cells (SPP1+ vs C1QC+ TAMs).
4. Liberzon et al., Cell Systems (2015) - MSigDB Hallmark Gene Sets (EMT, Proliferation, Hypoxia, Interferon, P53).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

# 16 Kanonik Biyolojik Yolak ve Kanser Kök Hücresi (CSC) Gen Kümeleri
CANONICAL_GENE_SETS: Dict[str, List[str]] = {
    # 1. Kanser Kök Hücresi (CSC) Yüzey & Kök성 Faktörleri (Medema 2013 Nature Rev Cancer; Batlle & Clevers 2017 Nature Med)
    "CSC_Stemness_Markers": [
        "CD44", "PROM1", "ALDH1A1", "ALDH1A3", "SOX2", "NANOG", "POU5F1",
        "MYC", "KLF4", "BMI1", "ABCG2", "EPCAM", "LGR5", "THY1", "CD24",
        "ALCAM", "SOX9", "NES", "NOTCH1", "CTNNB1", "WNT5A", "SMO", "ITGA6"
    ],
    # 2. Malign Hücre Döngüsü ve Proliferasyon (Hallmark G2M / E2F Targets)
    "Malignant_Proliferation": [
        "MKI67", "TOP2A", "PCNA", "CCNB1", "CCNB2", "CDK1", "BIRC5",
        "AURKA", "AURKB", "UBE2C", "TYMS", "MCM2", "MCM4", "MCM6", "CENPF", "HMGB2"
    ],
    # 3. Epitelyal-Mezenkimal Geçiş (EMT) & Metastaz (Hallmark EMT - Kanser İnvazyon Örüntüsü)
    "EMT_Metastasis": [
        "VIM", "FN1", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "CDH2",
        "MMP2", "MMP9", "TGFBI", "COL1A1", "COL1A2", "COL3A1", "SPARC", "SERPINE1", "ACTA2"
    ],
    # 4. Tümör Hipoksi & Glikoliz (Warburg Etkisi - Hallmark Hypoxia)
    "Hypoxia_Glycolysis": [
        "VEGFA", "HIF1A", "LDHA", "SLC2A1", "PGK1", "ENO1", "CA9", "GAPDH",
        "PKM", "ALDOA", "TPI1", "ADM", "NDRG1", "BNIP3"
    ],
    # 5. P53 Yolağı, DNA Hasarı ve Apoptoz (Hallmark P53 Pathway)
    "P53_DNA_Repair": [
        "TP53", "CDKN1A", "CDKN2A", "BAX", "MDM2", "GADD45A", "BBC3",
        "PMAIP1", "FAS", "CCND1", "BRCA1", "BRCA2", "PARP1", "ATM"
    ],
    # 6. Viral Enfeksiyon & Tip-I/II İnterferon Fırtınası (COVID-19 Örüntüsü)
    "Viral_Interferon_ISG": [
        "ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "MX2", "OAS1", "OAS2",
        "OAS3", "IFI6", "IFI44L", "IFI27", "STAT1", "IRF7", "RSAD2", "BST2", "CXCL10"
    ],
    # 7. Multipl Skleroz (MS) Nöro-İnflamasyon & Demiyelinizasyon (DAM / DAA İmzası)
    "MS_NeuroInflammation": [
        "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "CD74", "C1QA", "C1QB", "C1QC",
        "APOE", "TREM2", "TYROBP", "GFAP", "AQP4", "S100B", "MBP", "PLP1", "MOG", "MAG"
    ],
    # 8. Pan-Kanser Anjiyojenik Tümör İlişkili Makrofaj (SPP1+ / INHBA+ / NLRP3+ TAM - Cheng et al. Cell 2021)
    "TAM_Angiogenic_SPP1": [
        "SPP1", "INHBA", "NLRP3", "IL1B", "GPNMB", "FN1", "VEGFA", "TIMP1",
        "MARCO", "EREG", "CXCL8", "CCL20", "PLAUR", "CTSB", "CTSD", "LGALS3"
    ],
    # 9. Pan-Kanser Doku Yerleşik/Komplement Makrofaj & Dendritik (C1QC+ / LYVE1+ / cDC - Cheng et al. Cell 2021)
    "TAM_Complement_C1QC": [
        "C1QA", "C1QB", "C1QC", "LYVE1", "MRC1", "FOLR2", "SELENOP",
        "APOE", "TREM2", "CD1C", "CLEC9A", "LAMP3", "CLEC10A", "FCER1A", "LILRA4"
    ],
    # 10. T/NK Hücre Sitotoksisite ve Tükenmişlik (Exhaustion) İmzası
    "Lymphoid_Cytotoxic_Exhaustion": [
        "CD3D", "CD3E", "CD4", "CD8A", "CD8B", "GZMB", "GZMA", "PRF1",
        "IFNG", "GNLY", "NKG7", "KLRD1", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX"
    ],
    # 11. Epitelyal Karsinom & Solunum/Pankreas Soyu (LUAD / PAAD / ESCA / THCA / UCEC / OV)
    "Epithelial_Carcinoma_Lineage": [
        "EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "CDH1", "MUC1",
        "SFTPB", "SFTPC", "SFTPA1", "NKX2-1", "SCGB1A1", "INS", "GCG", "SST", "PPY", "CPA1", "PRSS1"
    ],
    # 12. Nöronal ve Glial Beyin Soyu (Korteks Eksitatör / İnhibitör Nöron, Astrosit, Oligodendrosit)
    "Neural_Glial_Lineage": [
        "SNAP25", "SYT1", "RBFOX3", "SLC17A7", "CAMK2A", "GAD1", "GAD2",
        "PVALB", "SST", "VIP", "SV2C", "SLC1A2", "AQP4", "GFAP", "MOG", "PLP1", "MBP", "PDGFRA", "CSPG4"
    ],
    # 13. Hematopoietik, Miyeloid, B-Hücre/Miyelom ve Lösemi Soyu (CML / MYE / LYM)
    "Hematopoietic_Leukemia_Myeloma": [
        "PTPRC", "CD34", "GATA1", "GATA2", "HBB", "HBA1", "HBA2", "GYPA",
        "MPO", "ELANE", "CD33", "CD14", "LYZ", "FCGR3A", "MS4A1", "CD19", "CD79A", "SDC1", "MZB1", "IGHG1", "JCHAIN"
    ],
}

SIGNATURE_NAMES: List[str] = list(CANONICAL_GENE_SETS.keys()) + [
    "CytoTRACE_Transcriptional_Entropy",
    "Chromosomal_CNV_Instability_Index",
    "Modality_Density_Index",
]


def extract_or_load_chrom_coords(perirhinal_h5ad_path: str, cache_json_path: str) -> Dict[str, Tuple[int, int]]:
    """
    Extracts GRCh38 (Chromosome_int, Start_bp) for 59,357 human genes from Perirhinal Cortex.h5ad
    so we can sort genes along physical chromosomes (chr1..chr22, X=23, Y=24, M=25) and compute
    chromosomal arm copy-number imbalance (Pseudo-CNV).
    """
    cache_path = Path(cache_json_path)
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return {k: (v[0], v[1]) for k, v in raw.items()}

    chrom_map = {f"chr{i}": i for i in range(1, 23)}
    chrom_map.update({"chrX": 23, "chrY": 24, "chrM": 25, "chrMT": 25})

    gene_coords: Dict[str, Tuple[int, int]] = {}
    with h5py.File(perirhinal_h5ad_path, "r") as f:
        v = f["var"]

        def _read_col(col_name):
            obj = v[col_name]
            if isinstance(obj, h5py.Group) and "categories" in obj:
                cats = [x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x) for x in obj["categories"][:]]
                return [cats[c] if c >= 0 else "" for c in obj["codes"][:]]
            return [x.decode("utf-8", "ignore") if isinstance(x, bytes) else x for x in obj[:]]

        names = _read_col("feature_name")
        chroms = _read_col("Chromosome")
        starts = _read_col("Start")
        for g, ch, st in zip(names, chroms, starts):
            c_idx = chrom_map.get(str(ch), 26)
            gene_coords[str(g)] = (int(c_idx), int(st) if st is not None else 0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(gene_coords), encoding="utf-8")
    return gene_coords


def normalize_expression_matrix(X_dense: np.ndarray) -> np.ndarray:
    """
    Ensures biologically consistent log1p(CPM_10k) scale across all datasets:
    - If X contains raw UMI counts (max > 20), applies library-size normalization to 10,000 counts + log1p.
    - Clips negative values (if any batch-corrected residuals exist) to 0.
    """
    X = np.asarray(X_dense, dtype=np.float32)
    X = np.maximum(X, 0.0)
    max_val = float(X.max()) if X.size > 0 else 0.0
    if max_val > 20.0:
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        X = np.log1p((X / row_sums) * 1e4)
    return X


def compute_biological_signatures_and_csc(
    X_log1p: np.ndarray,
    gene_names: List[str],
    gene_coords: Dict[str, Tuple[int, int]],
    is_cancer_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 16-dimensional Biological Signature Matrix (N_cells, 16)
    plus the continuous CSC Stemness Score (N_cells,) and binary CSC+ flag (N_cells,).
    """
    n_cells, n_genes = X_log1p.shape
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    sig_matrix = np.zeros((n_cells, len(SIGNATURE_NAMES)), dtype=np.float32)

    # 1..13: Canonical Pathway Scores (mean log1p expression of present pathway genes, standardized)
    for s_idx, (sig_name, g_list) in enumerate(CANONICAL_GENE_SETS.items()):
        valid_indices = [gene_to_idx[g] for g in g_list if g in gene_to_idx]
        if valid_indices:
            sig_matrix[:, s_idx] = X_log1p[:, valid_indices].mean(axis=1)

    # 14: CytoTRACE Transcriptional Entropy (Shannon entropy of normalized expression per cell)
    exp_probs = np.expm1(np.minimum(X_log1p, 12.0))
    row_totals = exp_probs.sum(axis=1, keepdims=True)
    p = exp_probs / np.where(row_totals > 0, row_totals, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 1e-10, np.log(p), 0.0)
    entropy = -np.sum(p * log_p, axis=1) / max(np.log(max(n_genes, 2)), 1.0)
    sig_matrix[:, 13] = entropy.astype(np.float32)

    # 15: Chromosomal CNV Instability Index (Standard deviation of mean expression across 22 autosomes)
    chrom_means = []
    for c_id in range(1, 23):
        c_indices = [i for i, g in enumerate(gene_names) if gene_coords.get(g, (26, 0))[0] == c_id]
        if len(c_indices) >= 10:
            chrom_means.append(X_log1p[:, c_indices].mean(axis=1))
    if len(chrom_means) >= 5:
        chrom_stack = np.stack(chrom_means, axis=1)  # (N_cells, N_chroms)
        cell_mean = chrom_stack.mean(axis=1, keepdims=True)
        cnv_instability = np.std(chrom_stack - cell_mean, axis=1)
        sig_matrix[:, 14] = cnv_instability.astype(np.float32)

    # 16: Modality Density Index (Fraction of active genes - separates Single-Cell from Bulk RNA-seq)
    active_frac = (X_log1p > 0.05).mean(axis=1)
    sig_matrix[:, 15] = active_frac.astype(np.float32)

    # Compute composite CSC (Cancer Stem Cell) Score on cancer/malignant cells:
    # Combines CSC_Stemness_Markers (idx 0), Malignant_Proliferation (idx 1),
    # EMT_Metastasis (idx 2), Hypoxia (idx 3), and CytoTRACE_Transcriptional_Entropy (idx 13)
    def _robust_z(v):
        med = np.median(v)
        iqr = np.percentile(v, 75) - np.percentile(v, 25) + 1e-6
        return np.clip((v - med) / iqr, -4.0, 4.0)

    raw_stem = (
        0.35 * _robust_z(sig_matrix[:, 0])   # CSC Stemness Markers
        + 0.25 * _robust_z(sig_matrix[:, 13]) # CytoTRACE Entropy
        + 0.20 * _robust_z(sig_matrix[:, 2])  # EMT / Metastasis
        + 0.10 * _robust_z(sig_matrix[:, 1])  # Proliferation
        + 0.10 * _robust_z(sig_matrix[:, 14]) # CNV Instability
    )
    # Sigmoid mapping to [0, 1] probability scale, masked by cancer status
    csc_prob = 1.0 / (1.0 + np.exp(-raw_stem))
    csc_score = np.where(is_cancer_mask > 0, csc_prob, csc_prob * 0.15).astype(np.float32)

    # Top ~25% stemness cancer cells are flagged as CSC+ (1), non-cancer cells are always CSC- (0)
    if np.any(is_cancer_mask > 0):
        thresh = float(np.percentile(csc_score[is_cancer_mask > 0], 75))
        thresh = max(thresh, 0.55)
    else:
        thresh = 0.65
    is_csc = ((is_cancer_mask > 0) & (csc_score >= thresh)).astype(np.int64)

    return sig_matrix, csc_score, is_csc


def quantize_51_bins(X_log1p: np.ndarray, n_bins: int = 51) -> np.ndarray:
    """
    Quantizes non-zero expression values per sample into 1..(n_bins-1) quantile bins (0 stays 0).
    Crucial property: Scale-invariant across Single-Cell (scRNA-seq) and Bulk RNA-seq!
    """
    n_samples, n_genes = X_log1p.shape
    binned = np.zeros((n_samples, n_genes), dtype=np.int16)
    quantiles = np.linspace(0, 1, n_bins)  # 51 edges -> 50 positive bins (1..50)
    for i in range(n_samples):
        row = X_log1p[i]
        nz_mask = row > 0.0
        if np.any(nz_mask):
            nz_vals = row[nz_mask]
            edges = np.quantile(nz_vals, quantiles[1:-1])
            edges = np.unique(edges)
            bins_i = np.digitize(nz_vals, edges, right=True) + 1
            # Rescale to 1..(n_bins-1)
            if len(edges) > 0:
                bins_i = np.clip((bins_i * (n_bins - 1)) // (len(edges) + 1), 1, n_bins - 1)
            binned[i, nz_mask] = bins_i.astype(np.int16)
    return binned
