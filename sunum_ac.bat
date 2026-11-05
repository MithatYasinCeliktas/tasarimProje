@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==============================================================================
echo  scMamba2 + BiMamba Proje Tasarim Dersi Ara Raporu Sunumu Aciliyor...
echo ==============================================================================
if exist "scMamba2_BiMamba_Proje_Tasarim_Ara_Raporu_Sunumu.pptx" (
    start "" "scMamba2_BiMamba_Proje_Tasarim_Ara_Raporu_Sunumu.pptx"
) else (
    start "" "sunum.pptx"
)
exit /b 0
