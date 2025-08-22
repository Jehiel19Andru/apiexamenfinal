#!/bin/bash
set -e

# Pega aquí la NUEVA URL de tu archivo .pkl
# ¡CAMBIO AQUÍ!
DATASET_URL="https://github.com/TerrazasJr316/MailGuard-web/releases/download/v2.0-data/datasets.pkl"

# Revisa si el archivo de datos ya existe
if [ ! -f "datasets.pkl" ]; then
    echo "Dataset preprocesado no encontrado. Descargando..."
    curl -L $DATASET_URL -o datasets.pkl
    echo "Descarga completa."
else
    echo "Dataset preprocesado ya existe."
fi

echo "Iniciando la aplicación..."
streamlit run app.py --server.port $PORT