@echo off
chcp 65001 >nul
cd /d "%~dp0scripts"
set KMP_DUPLICATE_LIB_OK=TRUE
echo 遥感图像问答服务
echo 打开 http://127.0.0.1:8001
echo 首次启动加载检测模型约 20 秒，页面会显示加载状态
echo 关闭这个窗口即停止服务。
echo.
set PYTHONIOENCODING=utf-8
"D://anaconda3//envs//yolo//python.exe" -m uvicorn mvqa_server:app --host 127.0.0.1 --port 8001
pause
