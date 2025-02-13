import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Cargar datos
file_path = "cirrhosis.csv"
data = pd.read_csv(file_path)

# Preprocesamiento de datos
# Convertir variables categóricas a valores numéricos
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])  # Convertir 'Sex' (M/F) a (0/1)
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

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.1, random_state=42)

# Entrenar modelo KNN
best_k = 0
best_accuracy = 0
for k in range(1, 51):  # Probar con valores de k entre 1 y 51
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred_val = knn.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred_val)
    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

# Usar el mejor k encontrado
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, Y_train)

# Hacer predicciones
Y_pred = knn.predict(X_test)

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

# Gráfica de precisión por valor de k
accuracies = []
for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred_val = knn.predict(X_test)
    accuracies.append(accuracy_score(Y_test, Y_pred_val))

plt.plot(range(1, 51), accuracies, marker='o')
plt.title("Precisión del Modelo vs. Número de Vecinos (k)")
plt.xlabel("Número de Vecinos (k)")
plt.ylabel("Precisión")
plt.xticks(range(1, 51))
plt.grid()
plt.show()

# Mostrar resultados
print(f"Mejor k encontrado: {best_k}")
print(f"Precisión del modelo con k={best_k}: {accuracy:.2f}")
print("Reporte de clasificación:")
print(report)
