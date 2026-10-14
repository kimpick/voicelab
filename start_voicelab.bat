@echo off
rem VoiceLab server. Reads VOICELAB_GSV_ROOT from .env
chcp 65001 > nul
cd /d %~dp0
if exist .env for /f "usebackq tokens=1,* delims==" %%a in (".env") do if "%%a"=="VOICELAB_GSV_ROOT" set "GSV=%%b"
if "%GSV%"=="" set "GSV=C:\GPT-SoVITS"
title voicelab
"%GSV%\runtime\python.exe" -X utf8 -u app\server.py
pause
