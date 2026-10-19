@echo off
rem ==========================================================================
rem  FundiScan - Ejecuta todo en Windows (doble clic):
rem   1) busca Python 3.10-3.13   2) crea el entorno .venv   3) instala librerias
rem   4) ejecuta el notebook (datos + 3 modelos + metricas + exporta el modelo)
rem   5) copia el modelo a la app y abre la app en http://127.0.0.1:7860
rem  El avance queda en logs\estado.txt ; los detalles en logs\*.txt
rem  Clave de Gemini (opcional): crear clave_gemini.txt con la clave adentro.
rem ==========================================================================
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist logs mkdir logs
set LOG=logs\instalacion.txt
echo ==== INICIO %date% %time% > logs\estado.txt
echo FundiScan: instalando y ejecutando. No cierres esta ventana. Avance en logs\estado.txt

rem ---- 1. Buscar Python ------------------------------------------------------
set PY=
for %%C in ("py -3.12" "py -3.11" "py -3.13" "py -3.10" "%USERPROFILE%\anaconda3\python.exe" "python") do (
  if not defined PY (
    %%~C -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>&1 && set PY=%%~C
  )
)
if not defined PY (
  echo ERROR: no se encontro Python 3.10 a 3.13 >> logs\estado.txt
  goto :fin
)
echo PASO 1 OK Python: %PY% >> logs\estado.txt
%PY% --version >> logs\estado.txt 2>&1

rem ---- 2. Entorno virtual ----------------------------------------------------
if not exist .venv\Scripts\python.exe %PY% -m venv .venv >> %LOG% 2>&1
set VPY=.venv\Scripts\python.exe
if not exist %VPY% (
  echo ERROR: no se pudo crear .venv, ver %LOG% >> logs\estado.txt
  goto :fin
)
echo PASO 2 OK entorno .venv >> logs\estado.txt

rem ---- 3. Librerias ----------------------------------------------------------
echo PASO 3 instalando librerias (varios minutos)... >> logs\estado.txt
%VPY% -m pip install --upgrade pip >> %LOG% 2>&1
set TORCHIDX=--index-url https://download.pytorch.org/whl/cpu
where nvidia-smi >nul 2>&1 && set TORCHIDX=--index-url https://download.pytorch.org/whl/cu126
%VPY% -c "import torch" >nul 2>&1 || %VPY% -m pip install torch torchvision %TORCHIDX% >> %LOG% 2>&1
%VPY% -c "import torch" >nul 2>&1 || (
  echo ERROR instalando PyTorch, ver %LOG% >> logs\estado.txt
  goto :fin
)
%VPY% -m pip install timm ultralytics transformers kagglehub "gradio==6.30.0" google-genai pandas scikit-learn matplotlib jupyter nbconvert ipykernel >> %LOG% 2>&1
if errorlevel 1 (
  echo ERROR instalando librerias, ver %LOG% >> logs\estado.txt
  goto :fin
)
%VPY% -m ipykernel install --prefix .venv --name fundiscan --display-name "FundiScan (.venv)" >> %LOG% 2>&1
%VPY% -c "import torch;print('PASO 3 OK librerias | torch',torch.__version__,'| GPU:',torch.cuda.is_available())" >> logs\estado.txt 2>&1

rem ---- 4. Notebook (se salta si el modelo ya existe) -------------------------
if exist modelo\efficientnet_b0_fundicion.pt if exist modelo\config.json (
  echo PASO 4 se omite: el modelo ya existe en modelo\ >> logs\estado.txt
  goto :app
)
echo PASO 4 ejecutando el notebook %time% (en CPU puede tardar 2-3 horas)... >> logs\estado.txt
%VPY% -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=fundiscan notebooks\entrenamiento.ipynb > logs\notebook.txt 2>&1
if errorlevel 1 (
  echo ERROR en el notebook, ver logs\notebook.txt >> logs\estado.txt
  goto :fin
)
if not exist notebooks\trabajo\export\modelo\config.json (
  echo ERROR: el notebook no exporto el modelo >> logs\estado.txt
  goto :fin
)
xcopy /y /e /i notebooks\trabajo\export\modelo modelo >> %LOG% 2>&1
xcopy /y /e /i notebooks\trabajo\export\ejemplos ejemplos >> %LOG% 2>&1
echo PASO 4 OK notebook ejecutado y modelo copiado %time% >> logs\estado.txt

rem ---- 5. App ----------------------------------------------------------------
:app
if exist clave_gemini.txt set /p GEMINI_API_KEY=<clave_gemini.txt
echo PASO 5 app corriendo en http://127.0.0.1:7860 (cerrar esta ventana la detiene) >> logs\estado.txt
echo.
echo  App FundiScan corriendo en http://127.0.0.1:7860  (cerrar esta ventana la detiene)
start "" cmd /c "timeout /t 45 >nul & start http://127.0.0.1:7860"
%VPY% app.py > logs\app.txt 2>&1
echo ERROR: la app se detuvo, ver logs\app.txt >> logs\estado.txt

:fin
echo ==== FIN %date% %time% >> logs\estado.txt
echo.
echo Termino. Revisa logs\estado.txt
pause
