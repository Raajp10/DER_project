@echo off
echo.
echo ============================================================
echo   DER Zero-Day Anomaly Explanation Demo
echo ============================================================
echo.

cd /d D:\updated_dataset\demo_ui

echo [1/3] Preparing demo data...
python prepare_demo_data.py
if errorlevel 1 (
    echo [WARN] prepare_demo_data.py failed - fallback data will be used.
    echo        This is OK for demo purposes.
    echo.
)

echo.
echo [2/3] Installing npm dependencies...
npm install
if errorlevel 1 (
    echo [ERROR] npm install failed.
    echo         Make sure Node.js is installed: https://nodejs.org/
    pause
    exit /b 1
)

echo.
echo [3/3] Starting development server...
echo.
echo   Demo will open at: http://localhost:5173
echo   Press Ctrl+C to stop the server.
echo.
npm run dev

pause
