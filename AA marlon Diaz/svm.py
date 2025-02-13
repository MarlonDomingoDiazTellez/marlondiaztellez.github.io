import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Cargar datos
file_path = "cirrhosis.csv"
data = pd.read_csv(file_path)

# Preprocesamiento de datos
# Convertir variables categóricas a valores numéricos
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Ascites'] = label_encoder.fit_transform(data['Ascites'])
data['Hepatomegaly'] = label_encoder.fit_transform(data['Hepatomegaly'])
data['Spiders'] = label_encoder.fit_transform(data['Spiders'])
data['Edema'] = label_encoder.fit_transform(data['Edema'])
data['Drug'] = label_encoder.fit_transform(data['Drug'])

# Manejar valores faltantes
# Imputar solo columnas numéricas
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Convertir la variable objetivo 'Stage' a clases discretas
def categorize_stage(stage):
    if stage <= 2:
        return 0  # Etapa temprana
    elif stage <= 4:
        return 1  # Etapa intermedia
    else:
        return 2  # Etapa avanzada

data['Stage'] = data['Stage'].apply(categorize_stage)

# Eliminar columnas irrelevantes
X = data.drop(columns=['ID', 'Status', 'Stage'])  # Características
Y = data['Stage']  # Variable objetivo

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balancear clases usando SMOTE
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_scaled, Y)

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)

# Búsqueda de hiperparámetros óptimos
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, Y_train)

# Mejor modelo encontrado
best_svm = grid_search.best_estimator_

# Hacer predicciones
Y_pred = best_svm.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

# Matriz de confusión
conf_matrix = confusion_matrix(Y_test, Y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Temprana', 'Intermedia', 'Avanzada'], yticklabels=['Temprana', 'Intermedia', 'Avanzada'])
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# Mostrar resultados
print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")
print(f"Precisión del modelo: {accuracy:.2f}")
print("Reporte de clasificación:")
print(report)
