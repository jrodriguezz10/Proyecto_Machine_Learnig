@echo off
title Monitor IoT ^& ML - UPAO
echo ========================================================
echo    INICIANDO MONITOR DE CALIDAD DE AIRE ^& ALARMA IOT
echo                   UPAO ML - CICLO 8
echo ========================================================
cd /d "%~dp0"

set "PYTHON_EXE="

if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\.venv\Scripts\python.exe"
    goto :run
)

if exist "..\venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\venv\Scripts\python.exe"
    goto :run
)

set "PYTHON_EXE=python"

:run
echo [*] Ejecutando servidor con: %PYTHON_EXE%
"%PYTHON_EXE%" app.py
if errorlevel 1 (
    echo.
    echo [ERROR] El servidor se detuvo con codigo de error.
)
pause
