@echo off
chcp 65001 >nul
cd /d "%~dp0scripts"
set KMP_DUPLICATE_LIB_OK=TRUE
set USE_TF=0
echo Loading index, please wait ...
echo.
"D:\anaconda3\envs\yolo\python.exe" agent.py
echo.
pause
