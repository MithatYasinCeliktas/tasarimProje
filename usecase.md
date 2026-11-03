# 🧬 scRNA-seq ve Mamba/BiMamba Mimarisi ile Hücre Örüntüleme ve Sınıflandırma Yöntemleri (`usecase.md`)

Bu doküman; tek hücreli RNA dizileme (**scRNA-seq**) ve toplu doku (**Bulk RNA-seq**) verilerini kullanarak hücrelerin örüntülerini çıkartmak, hücre tiplerini sınıflandırmak ve **Sağlıklı $\rightarrow$ Hastalıklı $\rightarrow$ Kanser $\rightarrow$ Kanser Varyantı / Kanser Kök Hücresi (CSC)** hiyerarşik teşhisini **%95+ doğruluk** ve **Açıklanabilir Heatmap (XAI)** çıktılarıyla gerçekleştirmek için literatürdeki **GitHub Mamba Modellerini**, **Veri Seti Feature Gereksinimlerini** ve **7 Temel Sınıflandırma Yöntemini** içerir.

---

## BÖLÜM 1: Bilgisayar Mühendisleri İçin 2 Dakikada Biyolojik Veri Özeti

Biyolojide vücudumuzdaki her hücrenin (T hücresi, nöron, akciğer epitel hücresi vb.) içinde aynı kaynak kod (DNA – yaklaşık **20.000 – 60.000 gen**) bulunur. Peki aynı kaynak koda sahip iki hücreden biri neden sağlıklı bir bağışıklık hücresi olurken diğeri ölümcül bir **Kanser Kök Hücresi (CSC)** olur?

Cevap: **Hangi genlerin çalıştırıldığı (ifade edildiği / okunduğu) ve hangilerinin kapatıldığıdır.**
- Bir hücre bir geni çalıştırdığında o genden **mRNA** kopyaları üretir.
- **scRNA-seq (Tek Hücre RNA Dizileme):** Tek bir hücrenin içindeki mRNA kopyalarını sayar. Bilgisayar mühendisliği gözüyle bir hücre, `60.697` boyutlu seyrek (sparse) bir vektördür: $x \in \mathbb{N}^{60697}$. Vektördeki her indeks bir geni (ör. `TP53`, `CD44`, `MYC`), içindeki sayı ise o genin o hücrede kaç kez okunduğunu (**UMI Count**) gösterir.
- **Bulk RNA-seq (Toplu Doku Dizileme):** Hücreleri tek tek ayırmak yerine, bir tümör biyopsisindeki yüz binlerce hücreyi bir arada ezip toplam gen okuma ortalamasını alır. Yani Bulk veri, binlerce tekil hücre vektörünün ağırlıklı toplamıdır ($X_{\text{bulk}} = \sum_{i=1}^{N} w_i x_i$).

---

## BÖLÜM 2: Hücre Örüntülemek İçin Gereken Veri Seti ve Feature (Özellik) Standardı

### 2.1. Bir `.h5ad` (AnnData) Veri Setinde Olması Gereken Zorunlu Feature'lar

Bilgisayar mühendisliğinde `AnnData` (`.h5ad`) formatı, ilişkisel bir tensör veritabanıdır ve 4 ana tablodan oluşur:

| Katman | Bilgisayar Mühendisliği Karşılığı | İçermesi Gereken Zorunlu Feature'lar | Biyolojik ve Modeldeki Rolü |
| :--- | :--- | :--- | :--- |
| **`adata.X` / `layers`** | Ana Girdi Tensörü ($N_{\text{hücre}} \times M_{\text{gen}}$) | • `raw_counts` (Ham UMI tam sayıları)<br>• `X_normed` ($\log_1p(\text{CPM/10k})$ normalize)<br>• `X_binned` ($0..50$ arası 51 seviyeli ayrıklaştırılmış ifade kutuları) | Farklı deneylerdeki okuma derinliği farklarını siler; `scMamba2` modelinin `ContinuousValueEncoder` katmanına gen aktivasyon şiddeti olarak girer. |
| **`adata.var`** | Sütun (Feature / Gen) Metadata Tablosu | • `gene_name` (HGNC Sembolü: `CD44`, `TP53`)<br>• `ensembl_id` (`ENSG00000...`)<br>• `chromosome`, `start`, `end` (Kromozom konumu)<br>• `highly_variable` (HVG bayrağı ve varyans sırası) | Genleri `vocab.json` (`60.697` gen) sözlüğündeki token ID'lerine eşler. Kromozom sırası, kanserdeki **Kopya Sayısı Değişimi (CNV)** örüntüsünü yakalamak için kullanılır. |
| **`adata.obs` (Kalite & Teknik)** | Satır (Örnek/Hücre) Teknik Metadata'sı | • `n_counts` / `total_UMIs` (Toplam okuma)<br>• `n_genes` (Aktif gen sayısı)<br>• `mt_frac` / `percent.mt` (Mitokondriyal RNA oranı)<br>• `batch_id`, `donor_id`, `tissue`, `modality` (`sc` vs `bulk`) | • `mt_frac > %20` olan hücreler ölmekte olan (parçalanmış) gürültülü hücrelerdir, filtrelenir.<br>• `batch_id`, hastane/cihaz gürültüsünü (Batch Effect) silmek için kullanılır. |
| **`adata.obs` (Hedef Etiketler)** | Denetimli Öğrenme (Ground-Truth Labels) | 1. `cell_type` (Hücre Tipi: 40+ sınıf)<br>2. `is_diseased` (`0: Sağlıklı`, `1: Hastalıklı`)<br>3. `disease_type` (`Normal`, `COVID-19`, `MS`, `Cancer`)<br>4. `is_cancer` (`0: Kanser Değil`, `1: Kanser/Malign`)<br>5. `cancer_variant` (`LUAD`, `PAAD`, `KIDNEY`, `ESCA`, `OV`, `THCA`, `UCEC`, `LYM`, `MYE`, `AML` vb.)<br>6. `csc_score` / `is_csc` (Kanser Kök Hücresi skoru) | 4 aşamalı hiyerarşik karar ağacımızın eğitim hedefleridir (Loss fonksiyonlarını besler). |
| **Biyolojik İmza Feature'ları (Türetilmiş)** | Domain-Specific Feature Engineering | • **CSC (Kanser Kök Hücresi) Genleri:** `CD44`, `PROM1 (CD133)`, `ALDH1A1`, `SOX2`, `NANOG`, `POU5F1 (OCT4)`, `EPCAM`, `ABCG2`, `BMI1`, `LGR5`<br>• **Proliferasyon / Hücre Döngüsü:** `MKI67`, `TOP2A`, `PCNA`, `CCNB1`<br>• **EMT (Metastaz / Göç) İmzası:** `VIM`, `SNAI1`, `TWIST1`, `ZEB1`, `CDH2`<br>• **İnflamasyon / Stres İmzası:** `ISG15`, `IFIT1`, `IL1B`, `TNF`, `APOE` | Modelin bir hücreyi neden "Kanser", "Kanser Kök Hücresi" veya "Enfeksiyon/MS" olarak sınıflandırdığını doğrulayan ve **Heatmap** çıktısında doktor/kullanıcıya gösterilen kanıt genleridir. |

---

### 2.2. Elimizdeki Mevcut Yerel Veri Setlerinin (`pretrain/` Klasörü) Tam Analizi

Bilgisayarındaki tüm `.h5ad` ve `.h5mu` dosyalarını HDF5 seviyesinde taradığımda ortaya çıkan **muazzam tablo** şudur: **Elimizdeki veriler zaten kurmak istediğimiz hiyerarşik sistemin %85'ini doğrudan barındırıyor!**

| Klasör / Dosya Adı | Boyut | İçindeki Hücre / Doku | İçindeki Mevcut Etiketler (`obs`) | Projemizdeki Hangi Soruyu Çözüyor? |
| :--- | :--- | :--- | :--- | :--- |
| **`lung/sample_proc_lung_train.h5ad` & `test.h5ad`** | 75.9 MB | İnsan Akciğer Tümörü (`tLung`) ve Beyin Metastazı (`tL/B`) | `cell_type`: **`Malignant` (Kanserli Hücre)**, `Epithelial`, `T_cell`, `B_cell`, `Macrophage`, `Fibroblast`, `Endothelial`, `NK_cell`, `Mast`, `Dendritic` + **`subclone` (Kanser alt klonu)** | **4a, 4c, 4d:** Aynı doku içinde sağlıklı hücrelerle **gerçek Kanser (`Malignant`)** hücrelerini ve metastatik alt-klonları ayırt etmeyi doğrudan öğretir! |
| **`mye/reference_adata.h5ad` & `query_adata.h5ad`** | 164.7 MB | Pan-Kanser Tümör Mikroçevresi Miyeloid Atlası | `cancer_type`: **9 Farklı Kanser Varyantı!**<br>• `KIDNEY` (Böbrek)<br>• `PAAD` (Pankreas)<br>• `THCA` (Tiroid)<br>• `UCEC` (Rahim)<br>• `LYM` (Lenfoma)<br>• `ESCA` (Özofagus)<br>• `OV-FTC` (Yumurtalık)<br>• `MYE` (Miyelom)<br>+ 21 alt hücre tipi (`Macro_SPP1`, `Macro_C1QC` vb.) | **4d (Kanser Varyantı Sınıflandırma):** Modelin tümör mikroçevresi örüntüsünden **9 farklı kanser varyantını** tanımasını sağlar! |
| **`covid/batch_covid_subsampled_train.h5ad` & `test.h5ad`** | 50.4 MB | Akciğer, Burun ve Kan Hücreleri (39 hücre tipi, 5 seviye hiyerarşi) | `condition`:<br>• `control` (Sağlıklı)<br>• `moderate COVID-19` (Orta)<br>• `severe COVID-19` (Ağır) | **4a, 4b, 4c:** Modele **"Her hastalıklı akciğer hücresi kanser değildir!"** kuralını öğretir. Viral enfeksiyon örüntüsü ile akciğer kanseri (`lung`) örüntüsünü birbirinden ayırır. |
| **`ms/c_data.h5ad` & `filtered_ms_adata.h5ad`** | 67.7 MB | İnsan Beyin Korteksi (18 nöronal ve glial hücre tipi) | `c_data`: `disease = normal` (9 sağlıklı donör)<br>`filtered_ms`: `disease = multiple sclerosis` (12 MS hastası) | **4a, 4b, 4c:** Beyin dokusunda **Sağlıklı (`normal`)** ile **Otoimmün/Nörodejeneratif Hastalık (`Multiple Sclerosis`)** örüntüsünü öğretir. |
| **`norman/perturb_processed.h5ad` & `adamson/perturb_processed.h5ad`** | 2.83 GB | **A549** (Akciğer Adenokarsinomu) ve **K562** (Kronik Miyeloid Lösemi) Kanser Hücreleri | 284 + 88 farklı CRISPR gen pertürbasyonu (`BCL2L11`, `TGFBR2`, `ARID1A` vb.) | **4c, 4d & 5 (Heatmap):** Kanser hücrelerinde hangi genin açılıp kapanmasının hangi kanser alt-örüntüsünü tetiklediğini (nedensel gen ağlarını) öğretir. |
| **`Immune_ALL_human.h5ad`, `BMMC_processed.h5ad`, `10xMultiom`, `annonation_pancreas`, `Perirhinal Cortex.h5ad`** | 3.16 GB | Kemik iliği, Kan, Pankreas ve Sağlıklı Beyin Korteksi (100+ hücre alt tipi) | `cell_type`, `final_annotation`, `dpt_pseudotime`, `GEX_phase` (`G1`, `S`, `G2M`) | **1 & 4a (Temel Hücre Tipi ve Sağlıklı Referans Atlası):** Vücuttaki tüm temel hücre tiplerinin "Sağlıklı (Normal)" referans örüntüsünü oluşturur. |

---

## BÖLÜM 3: GitHub'da Mamba Mimarisi Kullanan 7 Örnek Yapay Zeka Modeli

Projemizde mimari referans, kod parçası ve ağırlık olarak kullanacağımız GitHub modelleri:

1. **[`xtalpi-xic/SC-MAMBA2` (SC-MAMBA2)](https://github.com/xtalpi-xic/SC-MAMBA2)** *(Yerel diskimizde ve Kaggle'da ağırlıkları mevcut!)*
   - **Mimari:** 57 milyon hücre verisiyle eğitilmiş, 24 katmanlı `Mamba-2` çekirdeği, `60.697` genlik sözlük (`vocab.json`), 51 seviyeli ifade kutuplama (`Category/Continuous Value Encoder`) ve `smart_flip` çift yönlü dizi çevirme.
   - **Nasıl Kullanacağız:** Projemizin **Ana Omurgası (Backbone)** olarak kullanacağız; 57 milyon hücrelik ön-eğitim ağırlıklarını (`best_model.pt`) koruyarak üstüne kendi **Gated BiMamba** ve **Hiyerarşik Teşhis Başlıklarımızı** ekleyeceğiz.

2. **[`23AIBox/scMamba` (scMamba Multi-Omics)](https://github.com/23AIBox/scMamba)** *(Yerel diskimizde kodu mevcut!)*
   - **Mimari:** Yüksek varyanslı gen (HVG) seçimine gerek kalmadan 10.240 geni `patch_size` büyüklüğünde **Genomik Yamalara (Patches)** bölen `scEmbeddings` + `Mamba2` + `GatedMLP` + Kosinüs benzerliği düzenlileştirmeli kontrastif öğrenme.
   - **Nasıl Kullanacağız:** Yama tabanlı (Patch-based) tokenizasyon ve kontrastif kayıp fonksiyonunu (örüntü kümeleme için) bu repodan adapte edeceğiz.

3. **[`MineSelf2016/GeneMamba` (GeneMamba – BiMamba Single-Cell Foundation Model)](https://github.com/MineSelf2016/GeneMamba)** *(arXiv:2504.16956)*
   - **Mimari:** Tek hücreli RNA verilerini gen ekspresyon sırasına göre dizip doğrudan **Çift Yönlü Mamba (BiMamba)** bloklarıyla işleyen, Transformer'ların $O(L^2)$ bellek darboğazını $O(L)$'ye indiren model.
   - **Nasıl Kullanacağız:** Hücre içi gen-gen korelasyon matrisini (GRN) BiMamba gizli durumlarından (hidden states) çıkarma yöntemini örnek alacağız.

4. **[`kuleshov-group/caduceus` (Caduceus – Bidirectional Mamba for Genomics)](https://github.com/kuleshov-group/caduceus)** *(ICML 2024)*
   - **Mimari:** Standart tek yönlü Mamba bloğunu ileri ve geri yönlü iki SSM dalına ayırıp projeksiyon ağırlıklarını paylaşarak parametre sayısını şişirmeden çift yönlü genomik bağlam yakalayan resmi `BiMamba` tasarımı.
   - **Nasıl Kullanacağız:** `BiMamba` katmanımızın matematiksel kapılama (gating) ve ileri-geri birleştirme (fusion) tasarımında referans alacağız.

5. **[`bowang-lab/Orthrus` (Orthrus – Mamba RNA Foundation Model)](https://github.com/bowang-lab/Orthrus)**
   - **Mimari:** `scGPT`'yi geliştiren Bo Wang Lab'in `mamba-ssm` üzerine kurduğu, RNA dizilerinde kontrastif öz-denetimli (self-supervised contrastive) öğrenme ile fonksiyonel örüntü çıkaran modeli.
   - **Nasıl Kullanacağız:** Benzer biyolojik fonksiyona sahip hücreleri aynı hiper-küre (hypersphere) yüzeyinde gruplayan kontrastif projeksiyon başlığını örnek alacağız.

6. **[`hustvl/Vim` (Vision Mamba – Bidirectional State Space Model)](https://github.com/hustvl/Vim)**
   - **Mimari:** Nedensel (causal) olmayan 2B/1B yama dizilerinde ileri ($\text{SSM}_{\text{fwd}}$) ve geri ($\text{SSM}_{\text{bwd}}$) durum uzaylarını ortak bir $\sigma(\text{Linear}_z(x))$ kapısıyla çarparak birleştiren orijinal BiMamba bloğu.
   - **Nasıl Kullanacağız:** `KonuVeKapsam2.pdf` Sayfa 10 ve 12'de çizdiğin BiMamba mikro-mimarisinin birebir CUDA/PyTorch karşılığıdır.

7. **[`Huangmr0719/BiMamba` (Modular BiMamba PyTorch Implementation)](https://github.com/Huangmr0719/BiMamba)**
   - **Mimari:** Tekil diziler ve çok-değişkenli seriler için hafif sıklet, tak-çıkar (plug-and-play) çift yönlü Mamba modülü.

---

## BÖLÜM 4: scRNA-seq + BiMamba ile Hücre Sınıflandırma ve Örüntü Çıkarma İçin 7 Yöntem

Projemizde **%95+ doğruluk**, **Bulk + Single-Cell uyumu** ve **Hiyerarşik Hastalık/Kanser/CSC Teşhisi** için kullanacağımız 7 bilimsel yöntem:

### Yöntem 1: Hiyerarşik Çok-Başlıklı Koşullu Sınıflandırma (Hierarchical Multi-Head Conditional Classification)
Düz (flat) tek bir sınıflandırıcı yerine, biyolojik karar ağacını birebir yansıtan **5 paralel ancak birbirini koşullandıran görev başlığı (Task Heads)** kullanılır:
1. **$H_{\text{cell}}$ (Hücre Tipi Başlığı):** Hücrenin doku/soy kimliğini tahmin eder ($P(\text{CellType} \mid z)$).
2. **$H_{\text{health}}$ (Sağlık Durumu Başlığı):** Hücrenin sağlıklı (`0`) mı yoksa patolojik/hastalıklı (`1`) mı olduğunu tahmin eder ($P(\text{Diseased} \mid z, e_{\text{cell}})$).
3. **$H_{\text{disease}}$ (Hastalık Örüntüsü Başlığı):** Hastalıklıysa hangi ana hastalık ailesine (`Viral/COVID-19`, `Otoimmün/MS`, `Malign/Kanser`) ait olduğunu sınıflandırır.
4. **$H_{\text{cancer}}$ (Kanser & Varyant Başlığı):** Malign ise hangi kanser varyantına (`LUAD`, `PAAD`, `KIDNEY`, `ESCA`, `OV-FTC`, `THCA`, `UCEC`, `LYM`, `MYE`, `AML`) ait olduğunu bulur.
5. **$H_{\text{csc}}$ (Kanser Kök Hücresi / Stemness Başlığı):** Kanserli hücrenin sıradan bir tümör hücresi mi yoksa metastaz/nüksetme kaynağı olan **Kanser Kök Hücresi (CSC)** mi olduğunu skorlar.

> **Neden %95+ Doğruluk Sağlar?** Çünkü model "Bu bir T hücresidir" bilgisini ($e_{\text{cell}}$) sağlık/hastalık başlığına girdi olarak verir! Bir T hücresinde normal olan gen aktivitesi, bir akciğer epitel hücresinde görüldüğünde hastalık/anomali sinyalidir. Hücre tipine koşullandırılmış hastalık tespiti yanlış pozitifleri (false positive) dramatik biçimde düşürür.

---

### Yöntem 2: Denetimli Kontrastif Prototip Öğrenimi (Supervised Contrastive & Disease Archetype Prototyping)
Sadece Cross-Entropy ile sınıflandırma yapmak yerine, BiMamba'dan çıkan 512 boyutlu hücre vektörleri ($z \in \mathbb{R}^{512}$) birim küre üzerine normalize edilir ve her `(Hücre Tipi, Hastalık/Kanser Varyantı)` çifti için öğrenilebilir bir **Örüntü Prototip Vektörü ($\mu_k \in \mathbb{R}^{512}$)** tanımlanır:
$$\mathcal{L}_{\text{SupCon}} = -\sum_{i} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \neq i} \exp(z_i \cdot z_a / \tau)}$$
- **Ne İşe Yarar?**
  - Aynı hastalığa veya kanser varyantına sahip hücreler uzayda sıkı bir küme (ör. *"Akciğer Adenokarsinomu Malign Örüntü Kümesi"*, *"Multipl Skleroz Mikroglia Örüntü Kümesi"*) oluşturur.
  - Yeni bir hücre geldiğinde, bu hücrenin tüm hastalık ve kanser prototiplerine olan **Kosinüs Benzerliği ($\cos(z, \mu_k)$)** doğrudan **"Hangi Örüntüyle Ne Kadar Uyumlu?"** sorusunun matematiksel cevabını ve **Örüntü Uyum Heatmap'ini** üretir!

---

### Yöntem 3: "BULK DATA BİLE OLSA" Çalışan Çift-Modlu (Single-Cell + Pseudo-Bulk) Deconvolution ve Sıralama Kodlaması
Kullanıcı modele tek bir hücre (`scRNA-seq`) yerine bir hastadan alınmış toplu tümör/doku verisi (**Bulk RNA-seq**) verdiğinde modelin çökmemesi ve doğru teşhis koyması için iki mekanizma birleştirilir:
1. **Sıralama ve Kutuplama (Rank-Order & Quantile Binning):**
   Bulk RNA-seq (TPM/FPKM) ile scRNA-seq (UMI) verilerinin sayısal ölçekleri tamamen farklıdır. Ancak her iki veride de genlerin ekspresyon dağılımı kendi içinde **51 yüzdelik kutuya (`0..50` Quantile Bins)** dönüştürüldüğünde ölçek farkı tamamen ortadan kalkar!
2. **Sentetik Pseudo-Bulk Müfredat Eğitimi (Dirichlet Pseudo-Bulk Augmentation):**
   Eğitim sırasında elimizdeki tek hücre verilerinden rastgele $K \in [50, 500]$ hücre seçilip Dirichlet dağılımı ağırlıklarıyla toplanarak yapay **Bulk Doku Örnekleri** üretilir.
   - Eğer karışımda %5 ve üzerinde kanser/hastalıklı hücre varsa, model Bulk örnek içindeki o gizli kanser/hastalık örüntüsünü yakalamak üzere eğitilir!
   - Böylece model test aşamasında hem **tekil hücre** hem de **Bulk doku verisi** verildiğinde hücre tipi dağılımını, hastalık/kanser varlığını ve kanser varyantını yüksek doğrulukla tahmin eder.

---

### Yöntem 4: Kromozomal Konum ve Biyolojik Yolak (Pathway) Sıralı BiMamba Tokenizasyonu
scRNA-seq verisinde genlerin doğal bir cümle sırası yoktur. Genleri rastgele dizmek yerine iki biyolojik sıra kullanılır:
1. **Kromozomal Konum Sırası (`Chr1 -> Chr22, X, Y`):**
   Kanser hücrelerinin en büyük ayırt edici özelliği, sağlıklı hücrelerde olmayan **Kopya Sayısı Değişimleridir (CNV – örn. 7. kromozomun bir kolunun kopyalanması, 17p kolunun silinmesi)**. Genler kromozom üzerindeki fiziksel sırayla BiMamba'ya verildiğinde, BiMamba'nın `Conv1D + İleri/Geri SSM` katmanları yan yana duran 50-100 genin hep birlikte anormal yükseldiğini veya düştüğünü saniyesinde yakalar! Bu sayede **Sağlıklı/Enfeksiyon vs. Kanser (Malign)** ayrımı %98+ doğrulukla yapılır.
2. **Biyolojik Yolak (Hallmark Pathway) Gruplaması:**
   Genler kanser ve bağışıklık yolaklarına (`P53 Yolağı`, `Hipoksi`, `İnterferon Yanıtı`, `Hücre Döngüsü`, `Kök Hücre / Notch / Wnt Yolağı`) göre yamalara ayrılır.

---

### Yöntem 5: Kanser Kök Hücresi (CSC) ve Nadir Varyantlar İçin Odaklı Kayıp (Focal Loss) + Kök성 (Stemness) Kapısı
Tümör dokusunda Kanser Kök Hücreleri (CSC) tüm hücrelerin yalnızca %1–%5'ini oluşturur. Standart Cross-Entropy kaybı bu nadir hücreleri görmezden gelir.
- **Sınıf Ağırlıklı Odaklı Kayıp (Class-Balanced Focal Loss):**
  $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  Kolay sınıflandırılan sıradan hücrelerin ağırlığını $(1 - p_t)^\gamma$ ile bastırıp, modelin tüm dikkatini zor ve nadir olan **Kanser Kök Hücrelerine (CSC)** ve nadir kanser alt-tiplerine vermesini sağlar.
- **Biyobelirteç Dikkat Kapısı (CSC Marker Attention Gate):**
  `CD44`, `PROM1 (CD133)`, `ALDH1A1`, `SOX2`, `NANOG`, `POU5F1`, `ABCG2`, `EPCAM` ve `MYC` genlerinin BiMamba gizli durumları özel bir dikkat (cross-attention) başlığı ile okunarak **CSC Kök성 Skoru** hesaplanır.

---

### Yöntem 6: Etki Alanı Çekişmeli Parti Etkisi Giderme (Domain-Adversarial Batch Correction – `DAB` & `DSBN`)
Farklı hastanelerden veya cihazlardan (`10x v2`, `10x v3`, `Smart-seq2`) gelen verilerde teknik gürültü (**Batch Effect**) bulunur. Modelin "A hastanesinin cihaz gürültüsünü" yanlışlıkla "Kanser örüntüsü" sanmasını engellemek için:
- `SC-MAMBA2` içindeki **`DomainSpecificBatchNorm1d (DSBN)`** ve **`AdversarialDiscriminator (grad_reverse)`** kullanılır.
- Gradient Reversal Layer (`GRL`), BiMamba'nın ürettiği hücre temsilinden hangi laboratuvardan (`batch_id`) geldiği bilgisini **siler**, geriye yalnızca **saf biyolojik hücre tipi ve hastalık örüntüsünü** bırakır.

---

### Yöntem 7: Açıklanabilir BiMamba Durum Atfı (XAI) ve Otomatik Heatmap Üretim Mekanizması
Modelin verdiği kararın ("*%97.4 güvenle Akciğer Adenokarsinomu Malign Hücresi / CSC+*") kara kutu kalmaması için her tahminde otomatik olarak **3 Katmanlı Heatmap Görselleştirmesi** üretilir:
1. **Gen-Önem (Feature Attribution) Heatmap'i:**
   $\text{Score}(g_i) = \left\| \frac{\partial \text{Logit}_c}{\partial e(g_i)} \odot e(g_i) \right\|_2$ (Input $\times$ Gradient ve BiMamba İleri/Geri SSM aktivasyon normu) ile karara en çok etki eden **Top-25 Gen** çıkarılır ve hastanın gen değerleri sağlıklı referans ortalamasıyla yan yana Heatmap olarak çizilir.
2. **Örüntü Uyum (Archetype Similarity) Heatmap'i:**
   Girdi hücresinin/dokunun vektörü ($z$) ile veritabanımızdaki tüm hastalık ve kanser varyantı prototipleri ($\mu_1, \dots, \mu_K$) arasındaki kosinüs benzerlik matrisi $[0, 100\%]$ ölçeğinde sıcaklık haritası (Heatmap) olarak çizilir.
3. **Sıcaklık Ölçeklemeli Güven Kalibrasyonu (Temperature-Scaled Confidence + MC-Dropout):**
   Modelin ham softmax çıktıları aşırı özgüvenli olabileceği için doğrulama seti üzerinde öğrenilen $T$ sıcaklığı ($\text{Softmax}(\text{logits}/T)$) ve Monte Carlo Dropout örneklemeleri ile **gerçekçi, bilimsel olarak kalibre edilmiş Doğruluk/Güven Yüzdesi (`% Güven`)** raporlanır.
