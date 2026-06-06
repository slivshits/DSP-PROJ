@echo off
REM Build the standalone Windows executable: dist\FuzzFace.exe
setlocal
cd /d "%~dp0\.."

python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
if errorlevel 1 exit /b 1

pyinstaller --noconfirm --clean --windowed --onefile ^
  --name FuzzFace ^
  --collect-all soundfile ^
  --exclude-module matplotlib ^
  gui.py
if errorlevel 1 exit /b 1

echo.
echo Built: dist\FuzzFace.exe
endlocal
