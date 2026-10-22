#!/usr/bin/env bash
# Собирает пакет для Yandex Cloud Functions: build/ngm-trader.zip (зависимости + пакет trader).
# Запускать из корня репозитория.
set -euo pipefail
rm -rf build && mkdir -p build/pkg
pip install --quiet --target build/pkg -r trader/requirements.txt
mkdir -p build/pkg/trader
cp trader/*.py trader/requirements.txt build/pkg/trader/
(cd build/pkg && python3 -m zipfile -c ../ngm-trader.zip .)
ls -lh build/ngm-trader.zip
