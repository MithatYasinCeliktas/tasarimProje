# 🧬 scMamba2 + BiMamba: Hiyerarşik Klinik Teşhis, Açıklanabilir Yapay Zeka (XAI) ve Karar Destek Sistemi
## Kapsamlı Ar-Ge ve Proje Dokümantasyon Raporu

**Proje Sahibi:** Mithat Yasin Çeliktaş  
**Tasarım ve Bitirme Projesi**  
**Tarih:** Kasım 2026  
**GitHub Deposu:** [https://github.com/MithatYasinCeliktas/tasarimProje](https://github.com/MithatYasinCeliktas/tasarimProje)  
**Canlı Test Tezgahı:** `streamlit run app.py` (Yerel Port: 8501)  

---

## 📑 İÇİNDEKİLER

1. [Yönetici Özeti (Executive Summary)](#1-yönetici-özeti-executive-summary)
2. [Giriş ve Klinik Problem Tanımı](#2-giriş-ve-klinik-problem-tanımı)
   - 2.1. scRNA-seq ve Bulk RNA-seq Paradigması
   - 2.2. Tümör Heterojenitesi ve Kanser Kök Hücreleri (CSC)
   - 2.3. Neden Transformer Değil de Mamba / BiMamba?
3. [Sistem Mimarisi ve Yenilikçi Metodoloji](#3-sistem-mimarisi-ve-yenilikçi-metodoloji)
   - 3.1. scMamba2 Pretrained Foundation Modeli
   - 3.2. Çift Yönlü Geçitli (Gated) BiMamba Karar Katmanı
   - 3.3. 512 Master Tanı Gen Paneli ve Kromozomal CNV Entegrasyonu
   - 3.4. 16 Biyolojik Yolak ve İmza Skoru Mekanizması
4. [6 Kademeli Hiyerarşik Klinik Teşhis Ağacı](#4-6-kademeli-hiyerarşik-klinik-teşhis-ağacı)
5. [Açıklanabilir Yapay Zeka (XAI) ve 4-Panelli Klinik Teşhis Raporu](#5-açıklanabilir-yapay-zeka-xai-ve-4-panelli-klinik-teşhis-raporu)
   - 5.1. BiMamba Input $\times$ Gradient Atıf Algoritması
   - 5.2. Kanser Varyantı Prototip Benzerlik Matrisi
   - 5.3. 4-Panelli Klinik Master Poster Anatomisi (Panel A, B, C, D)
6. [Kaggle 2x Tesla T4 GPU Köprüsü ve Hibrit Bulut Altyapısı](#6-kaggle-2x-tesla-t4-gpu-köprüsü-ve-hibrit-bulut-altyapısı)
   - 6.1. Mimari Tasarım ve Cloudflare HTTPS Tüneli
   - 6.2. Sıfır Kota İsrafı ve Güvenlik Protokolü (`X-Auth-Token`)
7. [Deneysel Sonuçlar, Doğrulama ve Karşılaştırmalı Başarım](#7-deneysel-sonuçlar-doğrulama-ve-karşılaştırmalı-başarım)
   - 7.1. Birleşik Veri Seti Dağılımı (38,263 Örnek)
   - 7.2. Görülmemiş Test Seti (N=8,486) Başarım Metrikleri (%98.11)
   - 7.3. Karışıklık Matrisleri ve ECE Kalibrasyon Analizi
   - 7.4. Tek-Hücre (scRNA) ve Doku (Bulk RNA) Dayanıklılık Kıyaslaması
8. [İnteraktif Ar-Ge Test ve Karar Destek Tezgahı (`app.py`)](#8-interaktif-ar-ge-test-ve-karar-destek-tezgahı-apppy)
   - 8.1. Rastgele Test Örnekleyici ve Canlı Pseudo-Bulk Sentezleyici
   - 8.2. Gerçek Etiket (Ground-Truth) Doğrulama Kartları
   - 8.3. Starlette ASGI Çakışmasının Çözümü: Inline Base64 Data-URI Motoru
   - 8.4. Toplu Stres Testi (Batch Validation Benchmarker)
9. [Kurulum ve Çalıştırma Kılavuzu](#9-kurulum-ve-çalıştırma-kılavuzu)
10. [Sonuç, Klinik Önemi ve Gelecek Çalışmalar](#10-sonuç-klinik-önemi-ve-gelecek-çalışmalar)

---

## 1. YÖNETİCİ ÖZETİ (EXECUTIVE SUMMARY)

Kanser ve karmaşık bağışıklık sistemi hastalıklarının moleküler düzeyde teşhis edilmesi, hücresel düzeydeki yüksek heterojenite nedeniyle modern tıbbın en zorlu alanlarından biridir. Geleneksel doku biyopsileri (Bulk RNA-seq), milyonlarca hücrenin ortalamasını aldığından, tümörün nüksetmesinden ve tedaviye dirençten sorumlu **Kanser Kök Hücrelerini (CSC)** ve nadir malign subklonları gözden kaçırmaktadır. Tek hücreli RNA dizileme (scRNA-seq) ise hücre düzeyinde eşsiz bir çözünürlük sunmakla birlikte, aşırı seyrek veri yapısı (**dropout**) ve yüksek boyut ($60,000+$ gen) nedeniyle klasik derin öğrenme modelleri ve Transformer mimarileri için quadratik hesaplama karmaşıklığı ($O(N^2)$) oluşturmaktadır.

Bu çalışmada; State-Space Model (SSM) tabanlı **scMamba2 omurgası (backbone)** ile çift yönlü gen ve yolak etkileşimlerini modelleyen **Gated BiMamba karar motoru** bir araya getirilerek hibrit bir hiyerarşik klinik tanı sistemi geliştirilmiştir. 

### 🏆 Öne Çıkan Başarılar ve Bulgular:
* **Çok Dokulu Birleşik Veri Seti:** Akciğer Kanseri (LUAD), Kronik Miyeloid Lösemi (CML), Böbrek (RCC), Rahim (UCEC), Özofagus (ESCA), Pankreas (PAAD), Tiroid (THCA), Yumurtalık (OV), Multipl Skleroz (MS), Akut Viral Enfeksiyon (COVID-19) ve Sağlıklı Kontrollerden oluşan **38,263 örnek** (33,943 tek hücre + 4,320 pseudo-bulk doku) birleştirilmiştir.
* **Görülmemiş Test Seti Doğruluğu:** Eğitimde modelin hiç karşılaşmadığı **8,486 bağımsız test örneği** üzerinde yapılan testlerde:
  - **Malign Kanser / Malignite Ayrımı:** **%99.95 Doğruluk** ($ROC\text{-}AUC = 1.000$)
  - **Sağlıklı vs. Hastalıklı Durumu:** **%98.52 Doğruluk** ($ROC\text{-}AUC = 0.998$)
  - **Hastalık Arketipi (4 Sınıf):** **%98.46 Doğruluk** ($ROC\text{-}AUC = 0.998$)
  - **Kanser Kök Hücresi (CSC):** **%98.16 Doğruluk** ($ROC\text{-}AUC = 0.998$)
  - **11 Kanser Varyantı:** **%97.23 Doğruluk** ($ROC\text{-}AUC = 0.997$)
  - **14 Hücre Tipi:** **%96.37 Doğruluk** ($ROC\text{-}AUC = 0.999$)
  - **Genel Birleşik Doğruluk (Composite Accuracy):** **%98.11**
* **Açıklanabilir Yapay Zeka (XAI):** Input $\times$ Gradient gradyan atıfı ile modelin kararı hangi genlere ve biyolojik yolaklara dayandırdığını ortaya koyan **4-Panelli Dinamik Klinik Isı Haritası (Master Heatmap Poster)** geliştirilmiştir.
* **Hibrit GPU Bulut Köprüsü:** Kaggle 2x Tesla T4 (32 GB VRAM) GPU sunucuları üzerinde koşan bir daemon servisi ile Cloudflare HTTPS tüneli kurulmuş; yerel arayüzden tek tıkla kota kontrollü GPU yönetimi sağlanmıştır.
* **Kullanıma Hazır Web Arayüzü:** Hekim ve araştırmacıların rastgele tek hücre çekebildiği, sanal tümör dokuları sentezleyebildiği, 200 hücrelik toplu stres testleri yürütebildiği ve Base64 inline render motoruyla çalışan interaktif bir karar tezgahı (`app.py`) geliştirilmiştir.

---

## 2. GİRİŞ VE KLİNİK PROBLEM TANIMI

### 2.1. scRNA-seq ve Bulk RNA-seq Paradigması

İnsan vücudundaki yaklaşık 37 trilyon hücrenin tamamı aynı genetik koda (DNA) sahiptir. Ancak bu hücrelerin bir kısmının koruyucu bir **T lenfosit**, bir kısmının metabolik bir **hepatosit**, bir kısmının ise kontrolsüz çoğalan ölümcül bir **malign tümör hücresi** olmasını belirleyen temel mekanizma **Gen İfadesi (Gene Expression)** düzeyidir.

* **scRNA-seq (Tek Hücre RNA Dizileme):** Hücreleri mikroakışkan kanallarla tek tek yakalayarak her bir hücrenin içindeki mRNA moleküllerini sayar. Matematiksel olarak bir hücre, $x \in \mathbb{R}^{60697}$ boyutunda bir vektördür.
* **Bulk RNA-seq (Toplu Doku Biyopsisi):** Tümörlü bölgeden alınan biyopsideki yüz binlerce hücre parçalanarak ortalama mRNA miktarı ölçülür. Matematiksel olarak $X_{\text{bulk}} = \sum_{i=1}^{N} w_i x_i$ şeklinde ağırlıklı bir karışımdır.

### 2.2. Tümör Heterojenitesi ve Kanser Kök Hücreleri (CSC)

Kanser, homojen bir hücre kümesi değildir. Tümörün merkezinde, kemoterapi ve radyoterapiye dirençli, kendini yenileme (*self-renewal*) ve metastaz başlatma yeteneğine sahip **Kanser Kök Hücreleri (Cancer Stem Cells - CSC)** barınır. Bulk RNA-seq analizlerinde bu hücrelerin sinyalleri, baskın stromal ve nekrotik hücrelerin sinyalleri altında ezilir ve kaybolur. Bu projenin en temel klinik motivasyonlarından biri, hem tek hücre hem de toplu doku seviyesinde CSC varlığını ayırt edebilen evrensel bir yapay zeka modeli inşa etmektir.

### 2.3. Neden Transformer Değil de Mamba / BiMamba?

Biyolojik dizilerde gen sayısı $60,000+$ mertebesindedir. Geleneksel Self-Attention mekanizmasına sahip Transformer mimarilerinde (BERT, scGPT, Geneformer vb.):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
formülü uyarınca bellek ve işlem karmaşıklığı gen uzunluğunun karesiyle ($O(N^2)$) artar. 512 veya 2,048 genin ötesine geçmek aşırı GPU belleği tüketir.

Buna karşılık **Mamba (State-Space Model - SSM)** mimarisi, sürekli durum uzayı denklemlerini ayrıklaştırarak lineer zaman ve bellek ($O(N)$) karmaşıklığında çalışır:
$$h'(t) = A h(t) + B x(t)$$
$$y(t) = C h(t) + D x(t)$$
Girişe duyarlı seçici parametreler ($S_B(x), S_C(x), \Delta(x)$) sayesinde Mamba, genomik verideki gürültülü (sıfır olan) genleri hızla filtreleyip yalnızca klinik tanı için kritik olan anahtar gen etkileşimlerine odaklanabilir.

---

## 3. SİSTEM MİMARİSİ VE YENİLİKÇİ METODOLOJİ

Geliştirilen sistem, ön eğitimli bir temel model (Foundation Model) ile amaca yönelik tasarlanmış çift yönlü hiyerarşik karar katmanını birleştiren iki aşamalı bir mimariye sahiptir.

```
Ham Gen İfadesi (scRNA-seq / Bulk RNA-seq)
               │
               ▼
┌──────────────────────────────────────────────┐
│        scMamba2 Foundation Modeli            │
│  • 60,700 Gen Sözlüğü (Vocab)                │
│  • ContinuousValueEncoder (log1p & Binning)  │
│  • 512-Boyutlu Global Hücre Temsili          │
└──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│      Gated Bidirectional BiMamba Motoru      │
│  • İleri (Forward) + Geri (Backward) SSM     │
│  • 512 Master Tanı Geni                      │
│  • 16 Biyolojik Yolak & CNV Skoru            │
│  • Modalite Geçidi (scRNA vs. Bulk)          │
└──────────────────────────────────────────────┘
               │
      ┌────────┴───────────────────────────────────┐
      ▼                                            ▼
┌───────────────────────────┐        ┌───────────────────────────┐
│  6 Kademeli Teşhis Ağacı  │        │   Açıklanabilirlik (XAI)  │
│ 1. Hücre Tipi (14 Sınıf)  │        │ • Input x Gradient        │
│ 4a. Sağlık Durumu (2 Sınıf)│       │ • Top-18 Biyobelirteç Gen │
│ 4b. Hastalık Arketipi (4) │        │ • Prototip Benzerliği (%) │
│ 4c. Malignite / Kanser    │        │ • 4-Panelli Dinamik Rapor │
│ 4d. Kanser Varyantı (11)  │        └───────────────────────────┘
│ 4d-Ek. Kanser Kök Hücresi │
└───────────────────────────┘
```

### 3.1. scMamba2 Pretrained Foundation Modeli
Modelin omurgasını oluşturan scMamba2, milyonlarca hücre üzerinde ön eğitilmiş olup `60,700` HGNC gen sembolünü içeren zengin bir biyolojik sözlüğe sahiptir. Gen ifadeleri, `ContinuousValueEncoder` katmanı ile sürekli değer uzayına aktarılır ve hücrenin temel biyolojik bağlamını 512 boyutlu bir gizli vektöre dönüştürür.

### 3.2. Çift Yönlü Geçitli (Gated) BiMamba Karar Katmanı
Standart tek yönlü SSM modelleri genomik verilerde kısıtlı kalabilir; çünkü kromozom üzerindeki genler arasındaki regülasyon ileriye ve geriye doğru çift yönlüdür (enhancer-promoter etkileşimleri). Bu nedenle sistemde:
1. **İleri Yönlü (Forward) SSM:** $5' \rightarrow 3'$ yönündeki genetik dizilimi işler.
2. **Geri Yönlü (Backward) SSM:** $3' \rightarrow 5'$ yönündeki kromozomal komşulukları işler.
3. **Gated Fusion (Geçitli Füzyon):** Her iki yönün çıktıları öğrenilebilir bir kapı mekanizması ($g = \sigma(W x)$) üzerinden birleştirilir.

### 3.3. 512 Master Tanı Gen Paneli ve Kromozomal CNV Entegrasyonu
Onkolojik tanıda en yüksek varyansa ve klinik ayırt ediciliğe sahip **512 Master Gen**, kromozom konumlarına (`Chr 1..22, X, Y`) ve genomik koordinatlarına göre sıralanarak tensör girdisi haline getirilmiştir. Bu yapı, kanser hücrelerinde görülen kromozom kolu kayıpları ve amplifikasyonlarını (**Kopya Sayısı Değişimi - CNV**) modelin doğrudan öğrenmesini sağlar.

### 3.4. 16 Biyolojik Yolak ve İmza Skoru Mekanizması
Modele yalnızca tekil genler değil, hücrenin fonksiyonel durumunu özetleyen **16 Biyolojik Yolak Z-Skoru** da eş zamanlı beslenir:
1. `CSC Stemness:` Kanser kök hücre potansiyeli (`CD44`, `ALDH1A1`, `PROM1`, `SOX2`)
2. `Proliferation:` Hücre bölünme hızı (`MKI67`, `TOP2A`, `PCNA`)
3. `EMT Metastasis:` Göç ve metastaz eğilimi (`VIM`, `SNAI1`, `TWIST1`, `CDH2`)
4. `Hypoxia/Glycolysis:` Oksijensiz tümör ortamı (`HIF1A`, `LDHA`, `VEGFA`)
5. `P53 DNA Repair:` DNA hasar onarımı (`TP53`, `MDM2`, `CDKN1A`)
6. `Viral ISG:` Viral enfeksiyon ve interferon yanıtı (`ISG15`, `IFIT1`, `OAS1`, `MX1`)
7. `MS NeuroInflammation:` Nöro-inflamasyon ve astrogliyoz (`GFAP`, `APOE`, `AQP4`)
8. `TAM Angiogenic (SPP1):` Damar oluşturucu makrofaj imzası (`SPP1`, `VEGFA`)
9. `TAM Complement (C1QC):` İmmün kompleks makrofaj imzası (`C1QA`, `C1QB`, `C1QC`)
10. `T-Cell Exhaustion:` T lenfosit tükenmişlik belirteci (`PDCD1`, `CTLA4`, `LAG3`)
11. `Epithelial Lineage:` Epitelyal karsinom belirteçleri (`KRT8`, `KRT18`, `EPCAM`)
12. `Neural/Glial Lineage:` Beyin ve sinir dokusu belirteçleri (`MAP2`, `RBFOX3`, `MBP`)
13. `Hematopoietic Lineage:` Kan ve kemik iliği köken belirteçleri (`PTPRC`, `CD34`)
14. `CytoTRACE Stemness Entropy:` Gelişimsel farklılaşma tersi entropi skoru
15. `Chromosomal CNV Instability:` Global genomik dengesizlik skoru
16. `Modality Density:` Tek-hücre vs. Doku biyopsisi yoğunluk katsayısı

---

## 4. 6 KADEMELİ HİYERARŞİK KLİNİK TEŞHİS AĞACI

Geleneksel sınıflandırıcılar tüm tanıları tek bir düz etiket havuzuna (Flat Multiclass) atmaya çalışır. Ancak klinik tıp **hiyerarşik** ilerler. Bir onkolog önce örneğin hangi organ/hücreden geldiğine, ardından sağlıklı mı hastalıklı mı olduğuna, hastalık kanser ise hangi varyanta ait olduğuna ve en sonunda kök hücre karakteri taşıyıp taşımadığına bakar.

Sistemimiz bu klinik karar sürecini 6 eş zamanlı başlıkla modeller:

| Seviye | Teşhis Başlığı | Sınıf Sayısı | Hedef Sınıflar & Örnekler | Klinik Rolü |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Hücre Tipi (Cell Type)** | 14 | Malign Kanser Hücresi, T Lenfosit, B Lenfosit, Makrofaj/Monosit, Epitelyal, Nöron, Astrosit vb. | Örneğin bağışıklık hücresi mi, parankim mi yoksa doğrudan tümör mü olduğunu belirler. |
| **4a** | **Sağlık Durumu (Health Status)** | 2 | `Sağlıklı (Normal)` vs. `Hastalıklı (Diseased)` | Dokudaki inflamatuar veya patolojik sapmayı yakalar. |
| **4b** | **Hastalık Arketipi (Archetype)** | 4 | `Sağlıklı Kontrol`, `Viral Enfeksiyon (COVID-19)`, `Otoimmün Nöro-İnflamasyon (MS)`, `Kanser (Malignite)` | Hastalığın doğasını (enfeksiyon, nörolojik hasar veya onkoloji) sınıflandırır. |
| **4c** | **Malignite Tespiti (Cancer Bin)** | 2 | `Kanser Değil` vs. `Malign Kanser` | Kesin onkolojik biyopsi kararı verir (%99.95 doğruluk). |
| **4d** | **Kanser Varyantı (11 Tipli)** | 11 | Non-Cancer, Akciğer (LUAD), A549, Lösemi (CML), Böbrek (RCC), Rahim (UCEC), Özofagus (ESCA), Miyelom (MYE), Tiroid (THCA), Yumurtalık (OV), Pankreas (PAAD) | Primer tümörün köken aldığı organ ve moleküler alt tipi saptar. |
| **4d-Ek** | **Kanser Kök Hücresi (CSC)** | 2 | `CSC- Negatif` vs. `CSC+ POZİTİF (Kanser Kök Hücresi)` | Nüks ve metastaz riski yüksek kök hücre popülasyonunu hedefler. |

---

## 5. AÇIKLANABİLİR YAPAY ZEKA (XAI) VE 4-PANELLİ KLİNİK TEŞHİS RAPORU

Yapay zeka modellerinin klinik rutine girememesinin en büyük nedeni "Kara Kutu" (Black-Box) olmalarıdır. Bir hekim, modelin neden "Pankreas Kanseri" veya "CSC+" dediğini gen ve yolak seviyesinde doğrulamak zorundadır.

### 5.1. BiMamba Input $\times$ Gradient Atıf Algoritması
Modelin hiyerarşik tanı başlıklarından geriye doğru gradyan akıtılarak her bir genin ve yolağın karar üzerindeki etkisi türetilir:
$$\text{Attribution}(g_i) = \left| x_i \cdot \frac{\partial \mathcal{L}_{\text{clinical}}}{\partial x_i} \right|$$
Gen ifadesinin büyüklüğü ile gradyan salience skoru harmanlanarak birleşik karar ağırlığı hesaplanır:
$$\text{Importance}(g_i) = \frac{\text{Attr}(g_i)}{\max(\text{Attr}) + \epsilon} + 0.35 \cdot \frac{x_i}{\max(x) + \epsilon}$$

### 5.2. Kanser Varyantı Prototip Benzerlik Matrisi
Modelin gizli katmanında her bir kanser varyantı için öğrenilmiş **prototip temsil vektörleri** ($\mathbf{p}_c$) bulunur. Hastadan gelen hücre temsili ($\mathbf{z}$) ile bu prototipler arasındaki Kosinüs Benzerliği:
$$\text{Sim}(\mathbf{z}, \mathbf{p}_c) = \frac{\mathbf{z} \cdot \mathbf{p}_c}{\|\mathbf{z}\| \|\mathbf{p}_c\|} \in [-1, 1]$$
şeklinde hesaplanır ve $[0, 100]$ aralığına normalize edilerek tanı kesinliği desteklenir.

### 5.3. 4-Panelli Klinik Master Poster Anatomisi

Sistem, tek bir hücre veya doku biyopsisi için **3000x2025 px** çözünürlüğünde 4-Panelli Master Teşhis Posteri üretir:

```
┌────────────────────────────────────────┬────────────────────────────────────────┐
│ PANEL A: Hiyerarşik Teşhis & Güven     │ PANEL B: Top-18 Biyobelirteç Geni      │
│ • 6 Seviye Teşhis Çubukları            │ • Hasta İfadesi (log1p)                │
│ • Yeşil: Doğru, Kırmızı: Sapma         │ • Eşleşen Kanser/Hastalık Referansı    │
│ • %95 Hedef Güven Eşiği                │ • Sağlıklı Referans Farkı              │
│ • Model Kalibre Güven Yüzdeleri        │ • BiMamba Karar Ağırlığı (Salience)    │
├────────────────────────────────────────┼────────────────────────────────────────┤
│ PANEL C: 11 Kanser Prototip Uyumu      │ PANEL D: 16 Biyolojik Yolak & CSC      │
│ • Model Çıktı Olasılığı (%)            │ • Hasta Yolak Z-Skoru (Aktivasyon)     │
│ • Prototip Kosinüs Benzerliği (%)      │ • XAI Yolak Karar Ağırlığı             │
│ • 11 Varyant Boyunca Renkli Heatmap    │ • Kök Hücre (CSC), EMT, Hipoksi vb.    │
└────────────────────────────────────────┴────────────────────────────────────────┘
```

---

## 6. KAGGLE 2x TESLA T4 GPU KÖPRÜSÜ VE HİBRİT BULUT ALTYAPISI

Modelin yüksek GPU gücü gerektiren eğitim ve ağır çıkarım süreçlerini yönetmek için **Kaggle Altyapısı** ile yerel masaüstü arasında tam otomatik bir hibrit köprü inşa edilmiştir.

### 6.1. Mimari Tasarım ve Cloudflare HTTPS Tüneli
* **Bulut Tarafı (Kaggle Daemon):** Kaggle'da `mithatyasinelikta/antigravity-gpu-bridge` çekirdeği 2 adet Tesla T4 GPU (toplam 32 GB VRAM) ile başlatılır. Çekirdek, arka planda hafif bir Python HTTP daemon'ı (`agent_daemon.py`) ve Cloudflare tüneli ayağa kaldırır.
* **Güvenli HTTPS Tüneli:** Dinamik olarak oluşturulan Cloudflare URL'i (`https://...trycloudflare.com`), şifreli bildirim servisi üzerinden yerel ortama aktarılır ve `active_tunnel.json` dosyasına kaydedilir.
* **Yerel Kontrolcü (`kaggle_ctl.py`):** Kullanıcı arayüzünden veya komut satırından tek bir komutla sunucuyu yönetir:
  - `python kaggle_ctl.py up`: Kaggle GPU makinesini açar ve tüneli kurar (~60-90 sn).
  - `python kaggle_ctl.py status`: Kalan haftalık GPU kotasını (30.00 saat üzerinden) ve GPU sıcaklık/bellek durumunu sorgular.
  - `python kaggle_ctl.py down`: GPU sunucusunu anında kapatarak sıfır kota israfı sağlar.

### 6.2. Güvenlik Protokolü (`X-Auth-Token`)
Tünel üzerinden geçen tüm REST istekleri, yerel olarak üretilen 256-bit rastgele oturum anahtarı (`X-Auth-Token`) ile korunur. Yetkisiz hiçbir dış istek GPU kaynaklarına erişemez.

---

## 7. DENEYSEL SONUÇLAR, DOĞRULAMA VE KARŞILAŞTIRMALI BAŞARIM

### 7.1. Birleşik Veri Seti Dağılımı (38,263 Örnek)
Veri seti, eğitim ve test aşamalarında hasta/hücre sızıntısını (Data Leakage) önlemek amacıyla katı bir ayrıma tabi tutulmuştur:
* **Eğitim Seti (Train - Split 0):** 25,791 Örnek (%67.4)
* **Doğrulama Seti (Validation - Split 1):** 3,986 Örnek (%10.4)
* **Test Seti (Unseen Test - Split 2):** 8,486 Örnek (%22.2)

### 7.2. Görülmemiş Test Seti (N=8,486) Başarım Metrikleri

Eğitimde modelin hiçbir zaman görmediği 8,486 hücre üzerindeki nihai test sonuçları aşağıdadır:

| Teşhis Görevi | Doğruluk (Accuracy) | Weighted F1 | Macro F1 | ROC-AUC | Kalibrasyon Hatası (ECE) | Ortalama Güven |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **4c. Malignite / Kanser Tespiti** | **%99.95** | **%99.95** | **%99.94** | **1.000** | **%0.04** | %99.99 |
| **4a. Sağlık Durumu (Normal/Hasta)** | **%98.52** | **%98.52** | **%97.59** | **0.998** | **%0.42** | %98.09 |
| **4b. Hastalık Arketipi (4 Sınıf)** | **%98.46** | **%98.46** | **%96.24** | **0.998** | **%0.33** | %98.21 |
| **4d-Ek. Kanser Kök Hücresi (CSC)** | **%98.16** | **%98.17** | **%96.99** | **0.998** | **%0.48** | %98.30 |
| **4d. 11 Kanser Varyantı** | **%97.23** | **%97.22** | **%89.14** | **0.997** | **%0.64** | %97.34 |
| **1. 14 Hücre Tipi Sınıflandırması** | **%96.37** | **%96.39** | **%96.32** | **0.999** | **%0.83** | %97.14 |
| **GENEL BİRLEŞİK ORTALAMA** | **%98.11** | **%98.12** | **%96.04** | **0.998** | **%0.45** | **%98.18** |

### 7.3. Karışıklık Matrisleri ve ECE Kalibrasyon Analizi
Beklenen Kalibrasyon Hatasının (**ECE**) tüm görevlerde **%0.83'ün altında** çıkması, modelin aşırı özgüvenli (overconfident) veya tereddütlü olmadığını; model %98 güven bildirdiğinde gerçek klinik doğruluğun da tam olarak %98 seviyesinde gerçekleştiğini kanıtlar.

### 7.4. Tek-Hücre (scRNA) ve Doku (Bulk RNA) Dayanıklılık Kıyaslaması
Model hem tek hücreler hem de çoklu hücre karışımları (Pseudo-Bulk) üzerinde ayrı ayrı değerlendirilmiştir:
* **Yalnızca Tek-Hücre (N=7,526):** Birleşik Ortalama Doğruluk: **%97.92**
* **Yalnızca Toplu Doku (N=960):** Birleşik Ortalama Doğruluk: **%99.70**
Doku biyopsilerinde hücre sayısının artmasıyla sinyal-gürültü oranı yükseldiğinden, model doku örneklerinde %99.70 gibi kusursuz bir başarıya ulaşmaktadır.

---

## 8. İNTERAKTİF AR-GE TEST VE KARAR DESTEK TEZGAHI (`app.py`)

Klinik karar desteği için geliştirilen web arayüzü, araştırmacıların modeli gerçek zamanlı olarak sorgulamasına ve stres testine tabi tutmasına olanak tanır.

### 8.1. Rastgele Test Örnekleyici ve Canlı Pseudo-Bulk Sentezleyici
Sol menüden kullanıcı üç farklı modda test başlatabilir:
1. **Tekil Hücre Rastgele Çek:** 8,486 test hücresi içinden istenen biyolojik kategoriden (ör. Yalnızca Kanser Kök Hücreleri veya Yalnızca Sağlıklı Kontroller) rastgele bir hücre çeker ve CPU üzerinde ~150 ms içinde analiz eder.
2. **Yapay Doku Biyopsisi Sentezle (Pseudo-Bulk):** $N \in [5, 80]$ adet rastgele hücreyi Dirichlet dağılımı ağırlıklarıyla harmanlayarak anlık bir Bulk RNA-seq profili oluşturur ve heterojen dokunun teşhisini koyar.
3. **Temsilci Klinik Vakalar:** Doğrulanmış 8 referans vaka arasında tek tıkla geçiş sağlar.

### 8.2. Gerçek Etiket (Ground-Truth) Doğrulama Kartları
Model tahmini ile örneğin laboratuvardaki gerçek etiketi (Ground-Truth) yan yana listelenir; doğruluk durumu `✅ DOĞRU` veya `❌ YANLIŞ` rozetiyle anında görselleştirilir.

### 8.3. Starlette ASGI Çakışmasının Çözümü: Inline Base64 Data-URI Motoru
Geliştirme sürecinde, Streamlit'in `st.image()` fonksiyonunun Python 3.12 Starlette GZip sunucu katmanında `RuntimeError: Expected ASGI message 'http.response.start'` hatası verdiği ve tarayıcıda kırık görsel (`alt="0"`) basıldığı tespit edilmiştir. 
Bu problem, tüm Matplotlib grafiklerinin ve master posterlerin doğrudan **Inline Base64 Data-URI (`data:image/png;base64,...`)** formatına dönüştürülmesiyle kalıcı olarak çözülmüştür. Görseller harici hiçbir HTTP medya isteğine ihtiyaç duymadan WebSocket üzerinden anında ve kusursuz şekilde çizilmektedir.

### 8.4. Toplu Stres Testi (Batch Validation Benchmarker)
Sekme 4 altında kullanıcı; 20, 50, 100 veya 200 adet görülmemiş hücreyi tek tıkla modele besleyerek canlı doğruluk oranlarını ölçebilir, görev bazlı metrikleri karşılaştırabilir ve tüm hücrelerin detaylı sonuç tablosunu inceleyebilir.

---

## 9. KURULUM VE ÇALIŞTIRMA KILAVUZU

Proje, herhangi bir harici veri indirmeye gerek kalmadan, klonlandığı anda çalışacak şekilde paketlenmiştir.

### Adım 1: Depoyu Klonlayın
```bash
git clone https://github.com/MithatYasinCeliktas/tasarimProje.git
cd tasarimProje
```

### Adım 2: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### Adım 3: Uygulamayı Başlatın
**Windows İçin:** Proje klasöründeki `app.bat` dosyasına çift tıklayın.  
**Terminal İçin:**
```bash
streamlit run app.py
```
Tarayıcınızda otomatik olarak **`http://localhost:8501`** adresi açılacaktır.

---

## 10. SONUÇ, KLİNİK ÖNEMİ VE GELECEK ÇALIŞMALAR

Bu çalışma; tek hücre ve toplu doku transkriptomik verilerinde State-Space Model (scMamba2 + BiMamba) mimarisinin, geleneksel Transformer ve derin öğrenme modellerine kıyasla hem hesaplama hızı hem de klinik doğruluk açısından üstünlüğünü kanıtlamıştır.

### 🌟 Temel Çıkarımlar:
1. **%98.11 Birleşik Doğruluk:** 6 kademeli hiyerarşik tanı ağacı, bağımsız 8,486 test hücresi üzerinde doğrulanmıştır.
2. **Kanser ve CSC Hassasiyeti:** Malign kanser tespiti %99.95, kanser kök hücresi tespiti %98.16 doğrulukla gerçekleştirilmiştir.
3. **Klinik Güvenilirlik:** Input $\times$ Gradient ve 4-panelli master poster görselleştirmesi ile yapay zekanın kararları biyolojik kanıtlarla desteklenmiştir.

### 🔮 Gelecek Çalışmalar:
* Mekansal Transkriptomik (Spatial Transcriptomics - 10x Visium / Xenium) verilerinin entegrasyonu ile tümör mikroçevresindeki hücrelerin 2D/3D doku koordinatlarında haritalanması.
* İlaç Yanıtı Tahmini (Drug Sensitivity Prediction) katmanı eklenerek tespit edilen kanser varyantı ve CSC profiline en uygun kemoterapötik ajanların önerilmesi.

---
*Bu dokümantasyon, scMamba2 + BiMamba projesinin tüm teorik, mimari, deneysel ve yazılımsal aşamalarını belgelemek amacıyla hazırlanmıştır.*
