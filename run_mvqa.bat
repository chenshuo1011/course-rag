@echo off
cd /d "%~dp0mvqa"
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONIOENCODING=utf-8
echo.
echo   Remote-sensing image QA service
echo   Open: http://127.0.0.1:8001
echo   First start loads the detector, about 20 seconds.
echo   Close this window to stop.
echo.
"D://anaconda3//envs//yolo//python.exe" -m uvicorn mvqa_server:app --host 127.0.0.1 --port 8001
pause
