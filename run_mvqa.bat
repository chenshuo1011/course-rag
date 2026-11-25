@echo off
chcp 65001 >nul
cd /d %~dp0scripts
echo.
echo   遥感图像问答服务
echo   打开 http://127.0.0.1:8001
echo   首次启动加载检测模型约 20 秒，页面会显示状态
echo.
"D:\anaconda3\envs\yolo\python.exe" -m uvicorn mvqa_server:app --host 127.0.0.1 --port 8001
pause
