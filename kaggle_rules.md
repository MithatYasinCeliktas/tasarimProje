# 🛡️ Kaggle 2x T4 GPU Kullanım ve Yönetim Kuralları (`kaggle_rules.md`)

Bu doküman, `scRNAseq` (`scMamba2 + BiMamba` Kanser Kök Hücresi Tespiti) projesinde hem **Antigravity (AI Ajanı)** hem de **kullanıcı** tarafından Kaggle 30 saatlik ücretsiz GPU sunucusu kullanılırken uyulması zorunlu olan mühendislik ve operasyon kurallarını tanımlar.

---

## KURAL 1: 30 Saatlik GPU Kotasını Koruma Kuralları (Sıfır İsraf Politikası)

1. **Yalnızca İhtiyaç Anında Aç (`Just-in-Time GPU`):**
   - GPU sunucusu (`.\kaggle.bat up`) **yalnızca** aktif olarak model testi, veri ön işleme, çıkarım (inference) veya eğitim (training) çalıştırılacağı zaman açılır.
   - Yerel bilgisayarda mimari tasarım yapılırken, kod yazılırken, makale/sunum hazırlanırken veya hata analizi tartışılırken GPU sunucusu **kesinlikle kapalı (`COMPLETE`)** tutulur.
2. **İş Bitiminde Zorunlu Kapatma (`down`):**
   - Canlı GPU oturumunda yapılan test veya eğitim tamamlandığı anda, kullanıcıya yanıt dönmeden önce (veya çalışma biter bitmez) `.\kaggle.bat down` komutu çalıştırılır.
   - Kapatma işleminden sonra `.\kaggle.bat status` ile çekirdek durumunun `KernelWorkerStatus.COMPLETE` olduğu doğrulanır.
3. **Otomatik Kapanma Koruyucusu (`Watchdog`):**
   - [`agent_daemon.py`](file:///c:/Users/Tuğhan/Desktop/scRNAseq/kaggle_bridge/kernel/agent_daemon.py) içindeki `IDLE_TIMEOUT_SEC` (20 dakika hareketsizlikte otomatik kapanma) mekanizması asla devre dışı bırakılmaz.
   - 20 dakikadan uzun sürecek eğitimler mutlaka arka plan görevi (`bash_bg`) olarak başlatılır veya periyodik `/status` yoklaması yapılır; böylece Watchdog eğitim sürerken sunucuyu açık tutar, eğitim biter bitmez hareketsizlik süresi dolduğunda sunucuyu otomatik kapatır.

---

## KURAL 2: Donanım ve VRAM Yönetimi Kuralları (2x Tesla T4 – 32 GB VRAM)

1. **Çift GPU (`NvidiaTeslaT4`) Zorunluluğu:**
   - [`kernel-metadata.json`](file:///c:/Users/Tuğhan/Desktop/scRNAseq/kaggle_bridge/kernel/kernel-metadata.json) dosyasında `"enable_gpu": true` ve `"machine_shape": "NvidiaTeslaT4"` parametreleri asla değiştirilmez. (`machine_shape` silinirse Kaggle varsayılan olarak tekli P100 16 GB GPU atar).
2. **BiMamba İçin Bellek Dostu Eğitim:**
   - Tek yönlü Mamba-2 yerine çift yönlü **BiMamba** (ileri + geri SSM) kullanıldığında aktivasyon ve VRAM maliyeti ~2 katına çıkar.
   - `Out of Memory (OOM)` hatasını önlemek için tüm ileri/geri yayılım (forward/backward pass) işlemlerinde:
     - **Mixed Precision (`torch.autocast("cuda", dtype=torch.float16)`)** zorunludur.
     - Büyük batch boyutları için **Gradient Accumulation** ve çift GPU kullanımı (`torch.nn.DataParallel` veya `DDP` ile `cuda:0` ve `cuda:1`) tercih edilir.
3. **Tensör Boyutu Bütünlüğü:**
   - `scMamba2` içerisine entegre edilen her yeni `BiMamba` bloğu, giriş ve çıkışta `[Batch_Size, Sequence_Length, Hidden_Dimension]` tensör boyutunu birebir korumak zorundadır.

---

## KURAL 3: Kalıcı VRAM (`Stateful Worker`) ve Süreç İzolasyonu Kuralları

1. **Ana Sunucuda `torch` Import Yasağı:**
   - [`agent_daemon.py`](file:///c:/Users/Tuğhan/Desktop/scRNAseq/kaggle_bridge/kernel/agent_daemon.py) (ana tünel sunucusu) içinde asla doğrudan `torch`, `mamba_ssm` veya `scmamba2` import edilmez. Tüm derin öğrenme kodları izole `ag_gpu_worker.py` alt-sürecinde (`.\kaggle.bat py`) çalıştırılır.
2. **`os._exit(0)` Yasağı:**
   - Python kodları veya test scriptleri içinde kernel'i yeniden başlatmak amacıyla **asla `os._exit(0)` kullanılmaz** (bu komut arka plan sunucusunu tamamen kapatır).
   - Eğer GPU VRAM belleği sıfırlanmak veya yeni kurulan bir paket yüklenmek isteniyorsa `/restart_worker` uç noktası kullanılır; bu işlem tüneli ve sunucuyu kapatmadan sadece GPU Python işçisini 0.5 saniyede sıfırlar.
3. **Ağır Veri Setlerini Bellekte Sıcak Tutma:**
   - `best_model.pt` (288 MB) ve `.h5ad` / `.h5mu` (500 MB – 2.2 GB) veri setleri bir oturumda GPU/RAM belleğine (`GLOBAL_NS`) **yalnızca bir kez** yüklenir.
   - Sonraki kod denemelerinde model ve veri seti diskten tekrar okunmaz; bellekteki mevcut nesneler üzerinden milisaniyelik iterasyonlar yapılır.

---

## KURAL 4: `scMamba2` Bağımlılık ve Ortam Standartları

1. **Sabitlenmiş Kütüphane Sürümleri (Altın Stack):**
   C++ ABI ve CUDA uyuşmazlığı yaşamamak için her kurulumda şu sürümler standarttır:
   - `torch==2.3.0+cu121`, `torchvision==0.18.0+cu121`, `torchaudio==2.3.0+cu121`
   - `torchtext==0.18.0` (`--no-deps` ile)
   - `causal-conv1d==1.4.0` (`cu122torch2.3cxx11abiFALSE-cp312` hazır wheel)
   - `mamba-ssm==2.2.2` (`cu122torch2.3cxx11abiFALSE-cp312` hazır wheel, `--no-deps` ile)
   - `transformers==4.43.4` (`GreedySearchDecoderOnlyOutput` API uyumu için)
   - `scgpt` (`--no-deps` ile, mevcut `torch` sürümünü bozmaması için)
2. **Salt-Okunur (`/kaggle/input`) ve Yazılabilir (`/kaggle/working`) Ayrımı:**
   - `/kaggle/input` altındaki veri setleri salt-okunurdur.
   - `scmamba2` kaynak kodu her zaman `/kaggle/working/scmamba2` dizinine kopyalanır, `from scmamba import` hataları `\bscmamba\b -> scmamba2` regex'i ile düzeltilir ve `sys.path.insert(0, "/kaggle/working")` en üstte tanımlanır.

---

## KURAL 5: Veri, Kod ve Checkpoint Kalıcılığı (Veri Kaybını Önleme)

1. **Tek Doğruluk Kaynağı (`Single Source of Truth`) Yerel Bilgisayardır:**
   - Yazılan tüm model kodları, eğitim scriptleri ve konfigürasyonlar **önce yerel bilgisayarda (`c:\Users\Tuğhan\Desktop\scRNAseq`)** oluşturulur/güncellenir, ardından `.\kaggle.bat upload` ile Kaggle GPU sunucusuna gönderilir.
2. **Kapatmadan Önce Çıktıları İndirme Zorunluluğu:**
   - Kaggle sunucusu kapatılmadan (`.\kaggle.bat down` çalıştırılmadan) önce, `/kaggle/working` altında üretilen tüm önemli çıktılar (`.pt` model ağırlıkları, `.json`/`.csv` metrikler, UMAP/loss grafikleri) **mutlaka** `.\kaggle.bat download` komutuyla yerel bilgisayara indirilir.

---

## KURAL 6: Güvenlik ve Bağlantı Kuralları

1. **Kimlik Bilgisi Güvenliği:**
   - `~/.kaggle/access_token`, `~/.kaggle/kaggle.json` ve `agent_daemon.py` içindeki `AUTH_TOKEN` / `NTFY_TOPIC` bilgileri gizlidir; herkese açık (public) platformlarda paylaşılmaz.
2. **Windows Türkçe Karakter (`ğ`) SSL Standardı:**
   - Kullanıcı dizinindeki Türkçe karakter (`Tuğhan`) nedeniyle OpenSSL sertifika zinciri hatası oluşmaması için Kaggle CLI'da `verify_ssl = false` ayarı ve Python istemcilerinde `ssl._create_unverified_context()` kullanımı korunur.
