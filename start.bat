@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo 第一次运行，正在安装...
  python -m venv .venv || (echo 没找到 python，请先安装 Python 3.10+ & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install -q -e . || (pause & exit /b 1)
)
if not exist "config.yaml" (
  copy config.example.yaml config.yaml >nul
  echo 已生成 config.yaml，请用记事本打开，改好带 ★ 的三处后再双击本文件。
  notepad config.yaml
  pause
  exit /b 0
)
".venv\Scripts\hermes-media.exe" -c config.yaml ui
pause
