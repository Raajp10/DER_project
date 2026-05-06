@echo off
echo ============================================================
echo DER Cyber-Physical Anomaly Detection Dataset Pipeline
echo ============================================================
cd /d D:\DER_Project_update
python scripts_updated\run_updated_dataset_pipeline.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo PIPELINE FAILED. Check D:\DER_Project_update\logs\pipeline_errors.log
    echo.
) else (
    echo.
    echo PIPELINE COMPLETED SUCCESSFULLY.
    echo.
)
pause
