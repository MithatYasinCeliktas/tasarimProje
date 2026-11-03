@echo off
chcp 65001 >nul
title scMamba2 + BiMamba Ar-Ge Test ve Kaggle Kontrol Tezgahı
echo ============================================================
echo   scMamba2 + BiMamba Ar-Ge Test Tezgahı Başlatılıyor...
echo   Tarayıcınız açılacak: http://localhost:8501
echo ============================================================
streamlit run app.py --browser.gatherUsageStats=false
pause
