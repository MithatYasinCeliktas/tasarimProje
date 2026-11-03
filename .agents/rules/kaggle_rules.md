---
description: Kaggle 2x T4 GPU sunucusu kullanım, kota koruma (30 saat) ve scMamba2+BiMamba çalışma kuralları
---

# Kaggle 2x T4 GPU Kullanım Kuralları

Projede Kaggle GPU sunucusu (`c:\Users\Tuğhan\Desktop\scRNAseq\kaggle.bat` / `kaggle_bridge/kaggle_ctl.py`) kullanılırken `c:\Users\Tuğhan\Desktop\scRNAseq\kaggle_rules.md` ve `c:\Users\Tuğhan\Desktop\scRNAseq\kaggle_workflow.md` dosyalarındaki kurallara kesin olarak uyulmalıdır:

1. **Sıfır Kota İsrafı:** 30 saatlik haftalık GPU kotasını korumak için yalnızca aktif GPU işlemi yapılacağı zaman `.\kaggle.bat up` ile sunucu açılır; işlem biter bitmez (veya yerel kod geliştirme/tartışma aşamasına geçilirken) mutlaka `.\kaggle.bat down` ile sunucu kapatılır ve `.\kaggle.bat status` ile `COMPLETE` durumu doğrulanır.
2. **Çıktıları İndirme Zorunluluğu:** `.\kaggle.bat down` çalıştırılmadan önce `/kaggle/working` altında oluşan tüm önemli model ağırlıkları (`.pt`), loglar ve grafikler `.\kaggle.bat download` ile yerel bilgisayara indirilmelidir.
3. **Süreç İzolasyonu (`os._exit(0)` Yasağı):** Uzak sunucuda `os._exit(0)` kullanılmaz; GPU Python belleğini sıfırlamak için `/restart_worker` kullanılır. Model ve büyük `.h5ad` verileri `GLOBAL_NS` içinde bellekte sıcak tutulur.
4. **2x Tesla T4 Donanımı:** `kernel-metadata.json` içinde `"machine_shape": "NvidiaTeslaT4"` her zaman korunur.
