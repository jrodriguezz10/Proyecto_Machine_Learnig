@echo off
title Monitor IoT ^& ML - UPAO
echo ========================================================
echo    INICIANDO MONITOR DE CALIDAD DE AIRE ^& ALARMA IOT
echo                   UPAO ML - CICLO 8
echo ========================================================
cd /d "%~dp0"

if exist "..\.venv\Scripts\python.exe" (
    echo [OK] Activando entorno virtual del proyecto (.venv)...
    "..\.venv\Scripts\python.exe" app.py
) else if exist "..\venv\Scripts\python.exe" (
    echo [OK] Activando entorno virtual del proyecto (venv)...
    "..\venv\Scripts\python.exe" app.py
) else (
    echo [*] Usando Python del sistema...
    python app.py
)
pause

