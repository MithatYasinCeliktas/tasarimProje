@echo off
chcp 65001 >nul
python "%~dp0kaggle_bridge\kaggle_ctl.py" %*
