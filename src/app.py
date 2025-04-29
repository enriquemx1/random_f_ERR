from utils import db_connect
engine = db_connect()

# Paso 1: Carga del conjunto de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Verificar que la carpeta exista y crearla si no existe
models_dir = "../models/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Cargar los datos
url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
df = pd.read_csv(url)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 2: Construye un random forest
best_model = RandomForestClassifier(n_estimators=60, max_depth=None, random_state=42)
best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Exactitud del modelo: {accuracy_best:.2f}")

# Paso 3: Guarda el modelo en Pickle
model_path = os.path.join(models_dir, "ranfor_classifier_nestimators-60_42.pkl")

with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"Modelo guardado correctamente en {model_path}")