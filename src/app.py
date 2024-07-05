import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv")
print(data)
print(data.info())
print(data.describe())
print(data.shape)

# Selección de variables predictoras y objetivo
X = data[['age', 'bmi', 'children', 'smoker', 'region']]
y = data['charges']

# Convertir variables categóricas en variables numéricas
X = pd.get_dummies(X, drop_first=True)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicialización y entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Imprimir los parámetros del modelo
print(f"Intercepto (a): {model.intercept_}")
print(f"Coeficientes (b): {model.coef_}")

# Predicción del modelo
y_pred = model.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Error cuadrático medio: {mse}")
print(f"Coeficiente de determinación (R^2): {r2}")
print(f"Error absoluto medio: {mae}")

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, color='green', label='Datos Reales')
plt.scatter(y_test, y_pred, color='blue', label='Predicciones')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Línea de referencia')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.legend()
plt.show()

# ---------------------------------------------------------------------------------------------------------

# # Preprocesamiento de los datos
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), ['age', 'bmi', 'children']),
#         ('cat', OneHotEncoder(drop='first'), ['smoker', 'region'])
#     ])

# # Crear y entrenar el modelo dentro de un pipeline
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('model', LinearRegression())
# ])

# pipeline.fit(X_train, y_train)

# # Imprimir los parámetros del modelo
# model = pipeline.named_steps['model']
# preprocessor = pipeline.named_steps['preprocessor']
# coef = model.coef_
# intercept = model.intercept_
# print(f"Intercepto (a): {intercept}")
# print(f"Coeficientes (b): {coef}")

# # Predicción del modelo
# y_pred = pipeline.predict(X_test)

# # Evaluación del modelo
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)

# print(f"Error cuadrático medio: {mse}")
# print(f"Coeficiente de determinación (R^2): {r2}")
# print(f"Error absoluto medio: {mae}")

# # Visualización
# plt.scatter(y_test, y_pred, color='blue', label='Valores Reales')
# plt.scatter(y_test, y_test, color='green', label='Predicciones')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Línea de referencia')
# plt.xlabel('Valores Reales')
# plt.ylabel('Predicciones')
# plt.title('Predicciones vs Valores Reales')
# plt.legend()
# plt.show()

# ---------------------------------------------------------------------------------------------------------

# # EDA (aunque ya esté hecho, algunas recomendaciones adicionales)
# # 1. Visualizar la distribución de la variable objetivo
# plt.figure(figsize=(10, 6))
# sns.histplot(data['charges'], kde=True)
# plt.title('Distribución de los costos del seguro')
# plt.show()

# # 2. Visualizar la relación entre variables numéricas y la variable objetivo
# numeric_features = ['age', 'bmi', 'children']
# fig, axes = plt.subplots(1, 3, figsize=(20, 6))
# for i, feature in enumerate(numeric_features):
#     sns.scatterplot(data=data, x=feature, y='charges', hue='smoker', ax=axes[i])
# plt.tight_layout()
# plt.show()

# # 3. Visualizar la relación entre variables categóricas y la variable objetivo
# categorical_features = ['sex', 'smoker', 'region']
# fig, axes = plt.subplots(1, 3, figsize=(20, 6))
# for i, feature in enumerate(categorical_features):
#     sns.boxplot(data=data, x=feature, y='charges', ax=axes[i])
# plt.tight_layout()
# plt.show()

# # Preparación de los datos
# X = data.drop('charges', axis=1)
# y = data['charges']

# # Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Preprocesamiento y modelado
# categorical_features = ['sex', 'smoker', 'region']
# numeric_features = ['age', 'bmi', 'children']

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', 'passthrough', numeric_features),
#         ('cat', OneHotEncoder(drop='first'), categorical_features)
#     ])

# model = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', LinearRegression())
# ])

# # Entrenamiento del modelo
# model.fit(X_train, y_train)

# # Predicciones
# y_pred = model.predict(X_test)

# # Evaluación del modelo
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Error cuadrático medio: {mse}")
# print(f"Coeficiente de determinación (R^2): {r2}")

# # Visualización de predicciones vs valores reales
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel('Valores reales')
# plt.ylabel('Predicciones')
# plt.title('Predicciones vs Valores reales')
# plt.show()