@echo off
echo.
echo ===== See-through Desktop UI =====
echo.
cd /d "%~dp0\.."

set LOG=ui\tmp\ui_launcher.log
if not exist "ui\tmp" mkdir "ui\tmp"

if exist "assets" goto :skip_assets
if not exist "common\assets" goto :no_assets
mklink /J "assets" "common\assets" >nul 2>&1
:skip_assets

rem -- scegli un interprete PYTHON 3.11/3.12 (MAI 3.13/3.14: numpy/torch senza wheel)
set PY=
if not "%SEE_THROUGH_PY%"=="" if exist "%SEE_THROUGH_PY%" set PY=%SEE_THROUGH_PY%
if "%PY%"=="" if exist ".venv-st\Scripts\python.exe" set PY=%~dp0..\.venv-st\Scripts\python.exe
if "%PY%"=="" if exist "%USERPROFILE%\anaconda3\envs\see_through\python.exe" set PY=%USERPROFILE%\anaconda3\envs\see_through\python.exe
if "%PY%"=="" if exist "%USERPROFILE%\miniconda3\envs\see_through\python.exe" set PY=%USERPROFILE%\miniconda3\envs\see_through\python.exe
if "%PY%"=="" if exist "%LOCALAPPDATA%\anaconda3\envs\see_through\python.exe" set PY=%LOCALAPPDATA%\anaconda3\envs\see_through\python.exe
if "%PY%"=="" if exist "%LOCALAPPDATA%\miniconda3\envs\see_through\python.exe" set PY=%LOCALAPPDATA%\miniconda3\envs\see_through\python.exe
if "%PY%"=="" if exist "C:\ProgramData\anaconda3\envs\see_through\python.exe" set PY=C:\ProgramData\anaconda3\envs\see_through\python.exe
if "%PY%"=="" if exist "C:\ProgramData\miniconda3\envs\see_through\python.exe" set PY=C:\ProgramData\miniconda3\envs\see_through\python.exe

if "%PY%"=="" goto :no_py
echo Python scelto: %PY%

"%PY%" ui\ensure_ui_deps.py
if errorlevel 1 goto :deps_fail

goto :run_app

:deps_fail
echo Dipendenze UI non installabili (serve la rete). Rilancia quando online.
pause
exit /b 1

:run_app
set PYTHONPATH=%CD%;%CD%\common;%CD%\annotators;%PYTHONPATH%
echo Avvio applicazione...
"%PY%" ui\ui\launch.py %* > "%LOG%" 2>&1
set RC=%errorlevel%
echo.
echo Exit code: %RC%
type "%LOG%"
echo.
echo Fine. Il log sopra spiega l'esito.
pause
exit /b %RC%

:no_assets
echo common\assets non esiste nel clone: impossibile avviare.
pause
exit /b 1

:no_py
echo.
echo Nessun Python 3.11/3.12 valido trovato.
echo Il Python di sistema (3.14) NON funziona con questo repo: numpy/torch
echo non hanno wheel e la compilazione da sorgente fallisce.
echo.
echo Soluzione n.1  - usa  uv  per creare il venv:
echo    uv venv --python 3.11 ".venv-st"
echo    uv pip install --python ".venv-st\Scripts\python.exe" -e common -e annotators -r ui\requirements-ui-core.txt -e annotators
echo poi rilancia questo bat.
echo.
echo Soluzione n.2  - imposta SEE_THROUGH_PY al path di un python 3.11/3.12.
echo.
pause
exit /b 1