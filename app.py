# app.py
import os
import pickle
from flask import Flask, request, jsonify, render_template

# Configuración de la aplicación Flask
app = Flask(__name__)

# Rutas de archivos y modelos
MODEL_PATH = 'spam_classifier_model.joblib'
VECTORIZER_PATH = 'tfidf_vectorizer.joblib'

# --- Cargar el modelo y el vectorizador al iniciar la aplicación ---
# Usamos un try-except para manejar errores si los archivos no existen
# Esto es crucial para el despliegue en Render
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Modelos cargados con éxito.")
except Exception as e:
    print(f"Error al cargar modelos: {e}. El modelo no está disponible.")
    model = None
    vectorizer = None

# --- Rutas de la API ---

@app.route('/')
def home():
    # La página principal ahora no necesita entrenar nada
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Asegurarse de que los modelos se hayan cargado correctamente
    if not model or not vectorizer:
        return jsonify({'error': 'Modelo no disponible. Intente de nuevo más tarde.'}), 503

    try:
        data = request.get_json(force=True)
        text = data['text']

        text_transformed = vectorizer.transform([text])
        prediction = model.predict(text_transformed)
        is_spam = bool(prediction[0])

        response = {
            'text': text,
            'is_spam': is_spam,
            'prediction_label': 'spam' if is_spam else 'ham'
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Punto de entrada principal
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
