#!/usr/bin/env bash
# 一键启动节点界面（Linux / macOS）
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "第一次运行，正在安装..."
  python3 -m venv .venv
  .venv/bin/pip install -q -e .
fi
if [ ! -f config.yaml ]; then
  cp config.example.yaml config.yaml
  echo "已生成 config.yaml，请改好带 ★ 的三处后再运行 ./start.sh"
  exit 0
fi
exec .venv/bin/hermes-media -c config.yaml ui
