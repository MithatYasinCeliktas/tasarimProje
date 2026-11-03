# 🧬 scMamba2 + BiMamba: Hiyerarşik Klinik Teşhis & Açıklanabilir Yapay Zeka (XAI) Karar Destek Sistemi

Tek hücreli RNA dizileme (**scRNA-seq**) ve doku biyopsisi (**Bulk RNA-seq**) profillerini kullanarak **hücre tipi**, **sağlık durumu**, **hastalık arketipi**, **malignite**, **11 farklı kanser varyantı** ve **kanser kök hücresi (CSC)** hiyerarşisini yüksek doğrulukla (%98+) tespit eden, **açıklanabilir yapay zeka (XAI)** destekli klinik karar tezgahı.

---

## 🌟 Öne Çıkan Özellikler

1. **6 Kademeli Hiyerarşik Karar Ağacı:**
   - **Seviye 1:** 14 Temel Hücre Tipi Sınıflandırması
   - **Seviye 4a:** Sağlıklı vs. Hastalıklı Ayrımı
   - **Seviye 4b:** Hastalık Arketipi (Normal, Akut Viral COVID-19, Multipl Skleroz, Kanser)
   - **Seviye 4c:** Malignite Tespiti (Kanser Değil vs. Kanser)
   - **Seviye 4d:** 11 Farklı Kanser Varyantı (LUAD, A549, CML, RCC, UCEC, ESCA, MYE, THCA, OV, PAAD vb.)
   - **Seviye 4d-Ek:** Kanser Kök Hücresi (CSC+ Pozitif / CSC- Negatif)
2. **Kaggle 2x Tesla T4 GPU Köprüsü & Cloudflare Tüneli:**
   - Yerel bilgisayardan tek tıkla Kaggle GPU sunucusunu başlatma (`up`), kota sorgulama (`status`) ve kota tasarrufu için kapatma (`down`).
3. **Açıklanabilir Yapay Zeka (XAI) Paneli:**
   - **Top-18 Biyobelirteç Geni:** BiMamba Input $\times$ Gradient gradyan atıfı ile kararı belirleyen anahtar genler.
   - **16 Biyolojik Yolak:** Kök hücre (CSC), EMT metastaz, proliferasyon, hipoksi, p53 ve viral yanıt sapma Z-skorları.
   - **11 Kanser Prototip Örüntü Uyumu:** Kosinüs benzerliği ve kalibre olasılık dağılımı.
4. **4-Panelli Dinamik Teşhis Isı Haritası (Heatmap):**
   - Panel A (Güven Çubukları), Panel B (Top-18 Gen Heatmap), Panel C (11 Kanser Varyant Uyumu), Panel D (16 Yolak/CSC İmzası) master posteri (3000x2025 px) ve anında HD PNG indirme.
5. **Toplu Stres Testi (Batch Validation Benchmarker):**
   - 8,486 görülmemiş test hücresi üzerinde 20, 50, 100 veya 200 hücrelik anlık canlı doğruluk benchmark'ı.

---

## 🏗️ Mimari Şema

```
scRNA-seq / Bulk RNA-seq Girdisi
           │
           ▼
┌──────────────────────────────────────┐
│     scMamba2 Pretrained Backbone     │ (60,700 Gen Sözlüğü & ContinuousValueEncoder)
└──────────────────────────────────────┘
           │ (512-d Hücre Temsili)
           ▼
┌──────────────────────────────────────┐
│      Gated BiMamba Karar Motoru      │ (Çift Yönlü Gen & Yolak Çapraz Etkileşimi)
└──────────────────────────────────────┘
           │
     ┌─────┴──────────────────────────────────────────────────────┐
     ▼                                                            ▼
Hiyerarşik Teşhis Başlıkları (6 Görev)               XAI Açıklanabilirlik Katmanı
(Cell Type, Health, Disease, Cancer, Variant, CSC)   (Gradient Atıfları & 4-Panelli Heatmap)
```

---

## 🚀 Hızlı Başlangıç

### 1. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 2. Arayüzü Başlatın

**Windows Kolay Başlatma:**
Klasördeki `app.bat` dosyasına çift tıklayabilirsiniz.

**Terminalden Başlatma:**
```bash
streamlit run app.py
```
Uygulama tarayıcınızda otomatik olarak `http://localhost:8501` adresinde açılacaktır.

---

## 📁 Proje Dizin Yapısı

```
├── app.py                     # İnteraktif Streamlit Web Karar Tezgahı
├── app.bat                    # Tek tıkla yerel başlatıcı script
├── requirements.txt           # Python bağımlılıkları
├── kaggle_bridge/             # Kaggle 2x Tesla T4 GPU köprüsü & Cloudflare tüneli
│   ├── kaggle_ctl.py          # GPU yönetim motoru (up/down/status)
│   └── kernel/agent_daemon.py # Kaggle üzerinde koşan GPU sunucu çekirdeği
├── src/
│   ├── models/                # HierarchicalScMambaBiMambaModel mimarisi
│   └── visualization/         # 4-Panelli teşhis ve XAI heatmap motoru
├── results/
│   ├── best_hierarchical_bimamba.pt    # Eğitilmiş model ağırlıkları (~20.8 MB)
│   ├── unified_hierarchical_dataset.npz# 8,486 hücrelik unseen test seti (~30.7 MB)
│   └── xai_case_studies.npz            # Doğrulanmış vaka çalışmaları
└── usecase.md                 # Bilgisayar mühendisliği ve biyolojik mimari dokümantasyonu
```

---

## 👤 Geliştirici
**Mithat Yasin Çeliktaş**  
*Tasarım ve Bitirme Projesi — scMamba2 & BiMamba ile Hiyerarşik Hücre Teşhisi*
