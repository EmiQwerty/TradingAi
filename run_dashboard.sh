#!/bin/bash

# Script di launch per la dashboard Streamlit
# Uso: ./run_dashboard.sh

echo "🚀 Avvio ML Trading Dashboard..."
echo ""
echo "Dashboard disponibile su: http://localhost:8501"
echo "Premi Ctrl+C per fermare"
echo ""

# Installa dipendenze se non presenti
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installando dipendenze dashboard..."
    pip install -q -r requirements-dashboard.txt
fi

# Lancia la dashboard
streamlit run ml/dashboard.py --logger.level=info
