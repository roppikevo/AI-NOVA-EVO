@echo off
rem NOVA-EVO installer for Windows: needs Python 3.10+ from https://www.python.org/downloads/
cd /d "%~dp0"
set NOVA_PY=python
where py >nul 2>nul && set NOVA_PY=py -3
%NOVA_PY% install.py %*
if errorlevel 1 pause
