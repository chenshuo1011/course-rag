@echo off
chcp 65001 >nul
title 技术拆解演示
cd /d "%~dp0scripts"

set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONIOENCODING=utf-8

echo 正在加载模型，大约需要 10-20 秒...
echo.

"D:\anaconda3\envs\yolo\python.exe" demo_tech.py

echo.
echo ============================================================
echo  运行结束。按任意键关闭窗口。
echo ============================================================
pause >nul
