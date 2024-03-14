import numpy as np 
from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score  

# Generar un conjunto de datos de ejemplo.
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)

# Dividir el conjunto de datos en entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imprimir los conjuntos de datos de entrenamiento y prueba.
print("Conjunto de datos de entrenamiento:")
print(f"X_train:\n{X_train}\n")
print(f"y_train:\n{y_train}\n")

print("Conjunto de datos de prueba:")
print(f"X_test:\n{X_test}\n")
print(f"y_test:\n{y_test}\n")

# Implementar la función de distancia euclidiana.
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Implementar la función de clasificación por distancia euclidiana.
def euclidean_classifier(train_data, train_labels, test_point):
    distances = [euclidean_distance(train_point, test_point) for train_point in train_data]
    nearest_neighbor_index = np.argmin(distances)
    return train_labels[nearest_neighbor_index]

# Clasificar el conjunto de prueba y evaluar la precisión.
predictions = [euclidean_classifier(X_train, y_train, test_point) for test_point in X_test]
accuracy = accuracy_score(y_test, predictions)
print(f"Precisión del clasificador: {accuracy * 100:.2f}%")
