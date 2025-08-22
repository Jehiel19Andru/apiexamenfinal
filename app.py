# app.py
from flask import Flask, render_template, request
from model_logic import train_and_evaluate, get_total_data

app = Flask(__name__)

# Obtener los datos totales una sola vez al iniciar la aplicación
try:
    TOTAL_DATA = get_total_data()
except Exception as e:
    TOTAL_DATA = None 

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    error = None
    sample_size = None
    last_sample_size = None

    if request.method == 'POST':
        last_sample_size = request.form.get('sample_size', '')
        try:
            if last_sample_size:
                sample_size = int(last_sample_size)
            else:
                sample_size = None

            # Validación dinámica
            if sample_size is not None:
                if sample_size <= 0:
                    raise ValueError("Por favor, ingresa un número positivo mayor que cero.")
                if TOTAL_DATA is not None and sample_size > TOTAL_DATA:
                    raise ValueError(f"El tamaño de la muestra ({sample_size}) excede el límite de datos disponibles ({TOTAL_DATA}).")

            # Si pasa la validación, entrena el modelo
            results = train_and_evaluate(sample_size=sample_size)
        except (ValueError, TypeError) as e:
            error = str(e) if str(e) else "Por favor, ingresa un número válido."
        except Exception as e:
            error = f"Error al entrenar el modelo: {e}"
    else:
        last_sample_size = ''

    return render_template('index.html', results=results, error=error, last_sample_size=last_sample_size, total_data=TOTAL_DATA)

if __name__ == '__main__':
    # Esta línea no se utiliza en Render, pero es útil para pruebas locales
    app.run(debug=True)
