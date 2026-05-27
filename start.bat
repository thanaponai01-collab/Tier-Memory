@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Agent Memory Daemon

:: Add this folder to PATH permanently (only runs once effectively)
echo %PATH% | find /i "%~dp0" >nul 2>&1
if errorlevel 1 (
    set "THISPATH=%~dp0"
    if "!THISPATH:~-1!"=="\" set "THISPATH=!THISPATH:~0,-1!"
    powershell -NoProfile -Command "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path','User') + ';!THISPATH!', 'User')"
    echo [setup] Added mem command to PATH (restart terminals to apply)
)

:: ── 1. Memory daemon ─────────────────────────────────────────────────────────
echo.
echo Checking memory daemon...
python -m memory_system.cli status 2>nul | find "running" >nul
if errorlevel 1 (
    echo  Starting memory daemon...
    python -m memory_system.cli daemon start
    timeout /t 2 /nobreak >nul
) else (
    echo  Memory daemon already running.
)

:: Show live status
python -m memory_system.cli status

:: ── 3. Dashboard ─────────────────────────────────────────────────────────────
echo.
echo  Opening dashboard...
python -m memory_system.cli dashboard

echo.
echo  Ready. Type  mem run  in any project folder to start.
echo  Close this window anytime — daemon keeps running in background.
echo.
pause
