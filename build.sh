#!/bin/bash
set -e

# --- Pega aquí las URLs de tus archivos .joblib (modelo y vectorizador) ---
# ¡IMPORTANTE! Estas URLs deben ser de los archivos que subiste a GitHub.
MODEL_URL="https://github.com/Jehiel19Andru/apiexamenfinal/releases/download/v4.0-model/spam_classifier_model.joblib"
VECTORIZER_URL="https://github.com/Jehiel19Andru/apiexamenfinal/releases/download/v4.0-model/tfidf_vectorizer.joblib"

# --- Descargar el modelo y el vectorizador pre-entrenados ---
echo "Verificando si los modelos pre-entrenados existen..."
if [ ! -f "spam_classifier_model.joblib" ]; then
    echo "Modelo no encontrado. Descargando desde GitHub..."
    curl -L $MODEL_URL -o spam_classifier_model.joblib
else
    echo "Modelo ya existe."
fi

if [ ! -f "tfidf_vectorizer.joblib" ]; then
    echo "Vectorizer no encontrado. Descargando desde GitHub..."
    curl -L $VECTORIZER_URL -o tfidf_vectorizer.joblib
else
    echo "Vectorizer ya existe."
fi

# Instalar las dependencias de Python
echo "Instalando dependencias..."
pip install -r requirements.txt

# Iniciar la aplicación con Gunicorn (el runner para Flask)
echo "Iniciando la aplicación Flask con Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT app:app
