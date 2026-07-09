@echo off
REM ============================================================
REM Automated signal generator runner (Task Scheduler entry)
REM Appends every run to logs\signals.log (single rolling file)
REM ============================================================

cd /d C:\Users\fahim_ev79dv0\Projects\trading_system_v2

if not exist logs mkdir logs

echo. >> logs\signals.log
echo ===== RUN START %date% %time% ===== >> logs\signals.log

call venv\Scripts\activate.bat
python -m live.signal_generator >> logs\signals.log 2>&1

echo ===== RUN END %date% %time% ===== >> logs\signals.log