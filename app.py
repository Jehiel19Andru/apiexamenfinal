# app.py
from flask import Flask, render_template, request
from model_logic import train_and_evaluate

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    error = None
    sample_size = None
    MAX_DATA = 75419

    if request.method == 'POST':
        try:
            sample_size_str = request.form.get('sample_size', '')
            if sample_size_str:
                sample_size = int(sample_size_str)
            else:
                sample_size = None

            # 🚨 Validación
            if sample_size is not None:
                if sample_size <= 0:
                    raise ValueError("No se permiten números negativos ni cero.")
                if sample_size > MAX_DATA:
                    raise ValueError(f"No se permiten valores mayores a {MAX_DATA}.")

            # Entrenamiento
            results = train_and_evaluate(sample_size=sample_size)

        except (ValueError, TypeError) as e:
            error = str(e) if str(e) else "Por favor, ingresa un número válido."
        except Exception as e:
            error = f"Error al entrenar el modelo: {e}"

    return render_template('index.html', results=results, error=error, last_sample_size=sample_size)


if __name__ == '__main__':
    app.run(debug=True)