@echo off
REM Phase 1 Model Benchmark — Windows Batch Runner
REM Usage: run_phase1_models.bat [--skip-windows] [--skip-training]
SETLOCAL

SET SCRIPT_DIR=%~dp0
SET ROOT=%SCRIPT_DIR%..

echo ============================================================
echo  PHASE 1 MODEL BENCHMARK
echo ============================================================

python "%SCRIPT_DIR%run_phase1_models.py" %*

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Pipeline failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Phase 1 complete.
pause
