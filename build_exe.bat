@echo off
REM ===========================================================================
REM  Build AirSketch  (GUI launcher + app runner)
REM  Output: dist\AirSketch\AirSketch.exe  (a folder build, + reusable AirSketch.spec)
REM
REM  --onedir (NOT --onefile): onefile unpacks ~250 MB to %TEMP%\_MEIxxxxx on every
REM  launch, which endpoint antivirus on locked-down machines can corrupt mid-extract
REM  (the "pyi_rth_pkgres / base_library.zip not found" crash). --onedir has no temp
REM  extraction, starts faster, and is the reliable choice for a heavy app.
REM
REM  Models are NOT bundled (GBs). The exe finds the models\ folder by walking up
REM  from its own location (resolve_root), so running it from inside the project
REM  tree just works. To ship standalone, put a models\ folder next to AirSketch.exe.
REM
REM  Prereqs:  pip install pyinstaller
REM ===========================================================================
setlocal
cd /d "%~dp0"

python -m PyInstaller --noconfirm --onedir --windowed --name AirSketch ^
  --icon airsketch.ico ^
  --paths src ^
  --collect-all openvino ^
  --collect-all openvino_genai ^
  --collect-all openvino_tokenizers ^
  --collect-all cv2 ^
  --collect-all mediapipe ^
  --collect-submodules airsketch ^
  --exclude-module torch ^
  --exclude-module torchvision ^
  --exclude-module matplotlib ^
  --exclude-module tensorflow ^
  launcher.py

echo.
echo Done. The app is at: dist\AirSketch\AirSketch.exe
echo Keep the whole dist\AirSketch\ folder together; run AirSketch.exe inside it.
endlocal
