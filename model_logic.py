# model_logic.py
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import joblib

# ------------------------------------------------------------
# Función para obtener el total de datos
# ------------------------------------------------------------
def get_total_data():
    index_file = "datasets.pkl"
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"No se encontró '{index_file}'.")
    df_full = pd.read_pickle(index_file)
    return len(df_full)

# ------------------------------------------------------------
# Función para leer cada correo
# ------------------------------------------------------------
def load_email(path):
    try:
        # This will fail on Render because the email files are not present
        with open(path, encoding="latin-1", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return ""

# ------------------------------------------------------------
# Función principal: entrenar y evaluar
# ------------------------------------------------------------
def train_and_evaluate(sample_size=None):
    index_file = "datasets.pkl"
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"No se encontró '{index_file}'.")

    # Load dataset
    df_full = pd.read_pickle(index_file)

    # Check for the 'text' column. If it doesn't exist, try to load the emails
    if 'text' not in df_full.columns:
        print("The 'text' column does not exist. Attempting to load emails from disk...")
        df_full = df_full.rename(columns={'ruta_completa': 'full_path'})
        df_full["text"] = df_full["full_path"].apply(load_email)
        
    df_full = df_full[df_full["text"].str.strip() != ""]

    if df_full.empty:
        raise ValueError("No se cargaron correos válidos. Esto podría deberse a que el archivo datasets.pkl solo contiene rutas de archivo y no el contenido de los correos.")

    # Get sample if requested
    if sample_size and sample_size < len(df_full):
        df = df_full.sample(n=sample_size, random_state=42)
    else:
        df = df_full
        sample_size = len(df_full)

    print(f"Entrenando el modelo con {len(df)} correos...")

    X = df[["text"]]
    y = df["label"].apply(lambda x: 1 if x == "spam" else 0)

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    text_preprocessor = ColumnTransformer(
        transformers=[("tfidf", TfidfVectorizer(
            max_features=500,
            stop_words="english",
        ), "text")]
    )

    classifier = LogisticRegression(
        C=0.05, 
        solver='lbfgs',
        max_iter=2000,
        random_state=42
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", text_preprocessor),
            ("classifier", classifier)
        ]
    )

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_val)
    y_prob = model_pipeline.predict_proba(X_val)[:, 1]

    image_dir = "static/images"
    os.makedirs(image_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred, values_format="d", ax=ax, display_labels=['Ham', 'Spam'])
    ax.set_title("Matriz de Confusión")
    cm_path = os.path.join(image_dir, "confusion_matrix.png")
    fig.savefig(cm_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_val, y_prob, ax=ax)
    ax.set_title("Curva ROC")
    roc_path = os.path.join(image_dir, "roc_curve.png")
    fig.savefig(roc_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(y_val, y_prob, ax=ax)
    ax.set_title("Curva Precision-Recall")
    pr_path = os.path.join(image_dir, "pr_curve.png")
    fig.savefig(pr_path)
    plt.close(fig)

    results = {
        "accuracy": f"{accuracy_score(y_val, y_pred):.2%}",
        "f1_score": f"{f1_score(y_val, y_pred):.2%}",
        "confusion_matrix_url": cm_path,
        "roc_curve_url": roc_path,
        "pr_curve_url": pr_path,
        "data_used": sample_size,
        "total_data": len(df_full)
    }

    # --- CAMBIO CLAVE: Devolvemos el modelo entrenado y los resultados ---
    return model_pipeline, results
