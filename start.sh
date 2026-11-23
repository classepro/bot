#!/usr/bin/env bash
# Lancement de l'interface web (Streamlit) — accessible depuis l'extérieur du sandbox.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" -m streamlit run crashpredict/ui/app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-8501}" \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.enableCORS false \
  --server.enableXsrfProtection false
