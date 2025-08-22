# convert_dataset.py
import pandas as pd
import os
import sys

# --- ¡CORRECCIÓN! ---
# Usamos la ruta exacta que proporcionaste para el archivo 'index'.
INDEX_FILE_PATH = 'datasets/datasets/trec07p/full/index'
OUTPUT_PKL_FILE = 'datasets.pkl'

def load_trec07p_index(index_file):
    """
    Carga el archivo de índice y construye las rutas completas a los correos
    de forma robusta.
    """
    # Verificamos si el archivo 'index' existe antes de continuar
    if not os.path.exists(index_file):
        print(f"--- ¡ERROR! ---")
        print(f"No se pudo encontrar el archivo de índice en la ruta: '{index_file}'")
        print("Por favor, verifica que la ruta sea correcta.")
        sys.exit() # Detiene la ejecución del script

    data = []
    # Obtenemos el directorio base del archivo 'index' para construir las rutas relativas
    base_dir = os.path.dirname(index_file)
    
    with open(index_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            label = parts[0]
            # La ruta en el archivo index (ej: ../data/inmail.1) es relativa.
            # La unimos con el directorio base para obtener la ruta absoluta correcta.
            relative_path_from_index = parts[1]
            ruta_absoluta = os.path.abspath(os.path.join(base_dir, relative_path_from_index))
            data.append({'label': label, 'ruta_completa': ruta_absoluta})
            
    return pd.DataFrame(data)

# --- Proceso de Conversión ---
print("Cargando el índice del dataset...")
df = load_trec07p_index(INDEX_FILE_PATH)

print(f"Se encontraron {len(df)} registros en el archivo de índice.")
print("Verificando que los archivos de correo existan en el disco...")

# Verificamos la existencia de los archivos usando la nueva ruta absoluta
df['file_exists'] = df['ruta_completa'].apply(os.path.exists)
df_validos = df[df['file_exists']].drop(columns=['file_exists'])

# --- Verificación para evitar dataset vacío ---
if len(df_validos) == 0:
    print("\n--- ¡ERROR CRÍTICO! ---")
    print("No se encontró NINGÚN archivo de correo en las rutas esperadas.")
    if not df.empty:
        print(f"Por ejemplo, se buscó un archivo en una ruta como esta: '{df['ruta_completa'].iloc[0]}'")
    print("Por favor, verifica que los archivos de correo realmente existan en el disco.")
    sys.exit()

print(f"¡Éxito! Se encontraron {len(df_validos)} archivos de correo válidos.")
print(f"Guardando el DataFrame en '{OUTPUT_PKL_FILE}'...")
df_validos.to_pickle(OUTPUT_PKL_FILE)

print("¡Conversión completada con éxito! ✨")
print(f"Ahora puedes usar '{OUTPUT_PKL_FILE}' en tu aplicación.")