import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from joblib import dump

print("Iniciando el entrenamiento del modelo localmente...")

# Cargar el dataset (asegúrate de haber ejecutado convert_data.py primero)
DATASET_PATH = 'datasets.pkl'
try:
    # Cargar el DataFrame de Pandas
    df = pd.read_pickle(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{DATASET_PATH}'. Por favor, ejecuta 'convert_data.py' primero.")
    exit()
except Exception as e:
    print(f"Error al cargar el archivo '{DATASET_PATH}': {e}")
    exit()

# Separar datos y etiquetas
X = df['text']
y = df['label'].map({'spam': 1, 'ham': 0}).values

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el vectorizador
print("Entrenando el vectorizador Tfidf...")
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)

# Entrenar el modelo
print("Entrenando el modelo SGDClassifier...")
model = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-3, random_state=42)
X_train_transformed = vectorizer.transform(X_train)
model.fit(X_train_transformed, y_train)

# Guardar el modelo y el vectorizador en archivos .joblib
try:
    dump(model, 'spam_classifier_model.joblib')
    dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("¡Modelo y vectorizador guardados con éxito!")
    print("Ahora sube 'spam_classifier_model.joblib' y 'tfidf_vectorizer.joblib' a una nueva Release en GitHub.")
    print("Luego, actualiza tu app.py y build.sh en tu repositorio.")
except Exception as e:
    print(f"Error al guardar los modelos: {e}")