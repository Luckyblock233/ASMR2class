@echo off
rem ASR2class one-click launcher.
rem Single console window: shows live logs; closing it quits the app.
rem NOTE: keep this file pure ASCII - cmd parses .bat in the system ANSI
rem codepage (GBK on Chinese Windows) and UTF-8 CJK text breaks parsing.
title ASR2class
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo [ASR2class] uv not found.
  echo Please install uv first: https://docs.astral.sh/uv/getting-started/installation/
  echo Then double-click this file again.
  pause
  exit /b 1
)

uv sync
if errorlevel 1 (
  echo [ASR2class] Dependency install failed. Check your network and retry.
  pause
  exit /b 1
)

uv run python -m src.main %*
if errorlevel 1 (
  echo.
  echo [ASR2class] App exited with error %errorlevel%. Screenshot the log above for help.
  pause
)
