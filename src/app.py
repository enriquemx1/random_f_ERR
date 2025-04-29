from utils import db_connect
engine = db_connect()

# your code here
# Your code here
import pandas as pd

# URL del dataset
url = "https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv"

# Cargar datos
df = pd.read_csv(url)

# Crear carpeta y guardar el archivo
import os
raw_data_path = "./data/raw"
os.makedirs(raw_data_path, exist_ok=True)
df.to_csv(f"{raw_data_path}/AB_NYC_2019.csv", index=False)

print(" Datos cargados y almacenados en './data/raw'")

print("Información general:")
print(df.info())

print("\n Estadísticas descriptivas:")
print(df.describe())

print("\n Primeras filas:")
print(df.head())

print("\n Valores nulos por columna:")
print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Distribución de precios
plt.figure(figsize=(10,5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Distribución de Precios en Airbnb NYC")
plt.show()

columns_to_keep = ['neighbourhood_group', 'neighbourhood', 'room_type', 'price', 'minimum_nights', 'number_of_reviews']
df_filtered = df[columns_to_keep]

print(" Variables seleccionadas:")
print(df_filtered.head())

from sklearn.model_selection import train_test_split

train, test = train_test_split(df_filtered, test_size=0.2, random_state=42)

processed_data_path = "/workspaces/machine-learning-python-template/data/processed"
os.makedirs(processed_data_path, exist_ok=True)

train.to_csv(f"{processed_data_path}/train.csv", index=False)
test.to_csv(f"{processed_data_path}/test.csv", index=False)
