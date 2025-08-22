import pandas as pd
import os
import sys

# La ruta del archivo de índice del dataset
INDEX_FILE_PATH = 'datasets/datasets/trec07p/full/index'
OUTPUT_PKL_FILE = 'datasets.pkl'

# La cantidad de correos de muestra que el modelo puede cargar en la memoria de Render.
# Es la misma cantidad que tu compañero usó para su API.
SAMPLE_SIZE = 50000

# Función para leer el contenido de un correo
def load_email_content(path):
    try:
        with open(path, encoding="latin-1", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return ""

def load_trec07p_index(index_file):
    """
    Carga el archivo de índice y construye las rutas completas a los correos.
    """
    if not os.path.exists(index_file):
        print(f"--- ¡ERROR! ---")
        print(f"No se pudo encontrar el archivo de índice en la ruta: '{index_file}'")
        print("Por favor, verifica que la ruta sea correcta.")
        sys.exit()

    data = []
    base_dir = os.path.dirname(index_file)
    
    with open(index_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            label = parts[0]
            relative_path_from_index = parts[1]
            ruta_absoluta = os.path.abspath(os.path.join(base_dir, relative_path_from_index))
            content = load_email_content(ruta_absoluta)
            data.append({'label': label, 'text': content, 'ruta_completa': ruta_absoluta})
            
    return pd.DataFrame(data)

# --- Proceso de Conversión ---
print("Cargando el índice del dataset...")
df = load_trec07p_index(INDEX_FILE_PATH)

print(f"Se encontraron {len(df)} registros en el archivo de índice.")

# Filtramos los correos que se leyeron correctamente
df_validos = df[df['text'].str.strip() != ""]

if len(df_validos) == 0:
    print("\n--- ¡ERROR CRÍTICO! ---")
    print("No se encontró NINGÚN archivo de correo o no se pudo leer su contenido.")
    if not df.empty:
        print(f"Por ejemplo, se buscó un archivo en una ruta como esta: '{df['ruta_completa'].iloc[0]}'")
    sys.exit()

print(f"¡Éxito! Se encontraron {len(df_validos)} archivos de correo válidos.")

# --- ¡Paso clave para el despliegue! ---
# Tomamos una muestra aleatoria para reducir el tamaño del archivo final.
if len(df_validos) > SAMPLE_SIZE:
    df_final = df_validos.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Se ha tomado una muestra de {SAMPLE_SIZE} correos para el archivo final.")
else:
    df_final = df_validos
    print("El número de correos es menor que el tamaño de la muestra. Se usará el dataset completo.")

print(f"Guardando el DataFrame en '{OUTPUT_PKL_FILE}'...")
# Guardamos solo las columnas necesarias para la app web: 'label' y 'text'
df_final = df_final[['label', 'text']]
df_final.to_pickle(OUTPUT_PKL_FILE)

print("¡Conversión completada con éxito! ✨")
print(f"Ahora puedes usar '{OUTPUT_PKL_FILE}' en tu aplicación. Contiene el contenido de los correos.")
