@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Agent Memory Daemon

:: Add this folder to PATH permanently (only runs once effectively)
echo %PATH% | find /i "%~dp0" >nul 2>&1
if errorlevel 1 (
    set "THISPATH=%~dp0"
    if "!THISPATH:~-1!"=="\" set "THISPATH=!THISPATH:~0,-1!"
    powershell -NoProfile -Command "$p = '!THISPATH!'; [Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path','User') + ';' + $p, 'User')"
    echo [setup] Added mem command to PATH (restart terminals to apply)
)

:: -- 0. Ollama embedding backend ---------------------------------------------
:: The memory system embeds every query/ingest via ollama/nomic-embed-text on
:: :11434. If Ollama is down, search/retrieve/ingest fail with WinError 10061
:: while the daemon + dashboard still look healthy. Start it first.
echo.
echo Checking Ollama (embedding backend)...
curl -s -m 3 http://127.0.0.1:11434/api/tags >nul 2>&1
if not errorlevel 1 goto ollama_ready

echo  Starting Ollama...
start "" /b "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
set /a _otries=0
:ollama_wait
timeout /t 1 /nobreak >nul
curl -s -m 3 http://127.0.0.1:11434/api/tags >nul 2>&1
if not errorlevel 1 goto ollama_ready
set /a _otries+=1
if !_otries! lss 15 goto ollama_wait
echo  Warning: Ollama did not respond on :11434 after 15 s.
echo  Search / retrieve / ingest will fail until it is running.
goto ollama_done

:ollama_ready
echo  Ollama is up.
:ollama_done

:: -- 1. Memory daemon ----------------------------------------------------------
echo.
echo Checking memory daemon...
python -m memory_system.cli status 2>nul | find /i ": running (" >nul
if not errorlevel 1 goto daemon_already_running

echo  Starting memory daemon...
python -m memory_system.cli daemon start
set /a _tries=0
:wait_loop
timeout /t 1 /nobreak >nul
python -m memory_system.cli status 2>nul | find /i ": running (" >nul
if not errorlevel 1 goto daemon_ready
set /a _tries+=1
if !_tries! lss 10 goto wait_loop
echo  Warning: daemon did not report running after 10 s.
goto daemon_ready

:daemon_already_running
echo  Memory daemon already running.

:daemon_ready

:: Show live status
python -m memory_system.cli status

:: -- 3. Dashboard --------------------------------------------------------------
echo.
echo  Opening dashboard...
start "" python -m memory_system.cli dashboard

echo.
echo  Ready. Type  mem run  in any project folder to start.
echo  Close this window anytime -- daemon keeps running in background.
echo.
pause
