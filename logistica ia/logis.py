import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score

# Cargar datos de entrenamiento y prueba
try:
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Preprocesamiento y creación de nuevas características
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
test_data['Embarked'] = test_data['Embarked'].fillna(test_data['Embarked'].mode()[0])
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())

# Crear nuevas características
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

# Extraer títulos de los nombres
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Mapear títulos poco frecuentes
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
train_data['Title'] = train_data['Title'].apply(lambda x: x if x in common_titles else 'Rare')
test_data['Title'] = test_data['Title'].apply(lambda x: x if x in common_titles else 'Rare')

# Convertir variables categóricas a numéricas
label_encoder = LabelEncoder()
for col in ['Sex', 'Title']:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    test_data[col] = label_encoder.transform(test_data[col])

# Convertir 'Embarked' a valores numéricos y aplicar one-hot encoding
train_data = pd.get_dummies(train_data, columns=['Pclass', 'Embarked'])
test_data = pd.get_dummies(test_data, columns=['Pclass', 'Embarked'])

# Asegurar compatibilidad de columnas entre train_data y test_data
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[train_data.columns]

# Selección de características
features = ['Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Title']
features += [col for col in train_data.columns if col.startswith('Pclass_') or col.startswith('Embarked_')]
X = train_data[features]
y = train_data['Survived']
X_test = test_data[features]

# Dividir el conjunto de datos para validación
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Ajuste de hiperparámetros para regresión logística
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear'],
    'class_weight': ['balanced']
}
grid = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, cv=5, scoring='accuracy', verbose=1)
grid.fit(X_train, y_train)

# Mejor hiperparámetro y evaluación
print("Mejor conjunto de hiperparámetros para regresión logística:", grid.best_params_)
model = grid.best_estimator_

# Evaluación del modelo de regresión logística
y_pred_log = model.predict(X_valid)
log_accuracy = accuracy_score(y_valid, y_pred_log)
print(f"Precisión del modelo de regresión logística en validación: {log_accuracy:.2f}")

# Random Forest como modelo alternativo
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_valid)
rf_accuracy = accuracy_score(y_valid, y_pred_rf)
print(f"Precisión del modelo Random Forest en validación: {rf_accuracy:.2f}")

# Predicciones finales en el conjunto de prueba con el mejor modelo
final_model = model if log_accuracy > rf_accuracy else rf_model
test_predictions = final_model.predict(X_test)
print("Predicciones finales en el conjunto de prueba:", test_predictions)
