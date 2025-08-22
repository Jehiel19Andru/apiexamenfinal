#!/bin/bash
set -e

# Pegar la URL del archivo .pkl
DATASET_URL="https://github.com/Jehiel19Andru/apiexamenfinal/releases/download/v3.0-midataset/datasets.pkl"

# Revisa si el archivo de datos ya existe
if [ ! -f "datasets.pkl" ]; then
    echo "Dataset preprocesado no encontrado. Descargando..."
    curl -L $DATASET_URL -o datasets.pkl
    echo "Descarga completa."
else
    echo "Dataset preprocesado ya existe."
fi

# Inicia la aplicación con Gunicorn
echo "Iniciando la aplicación Flask con Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT app:app
