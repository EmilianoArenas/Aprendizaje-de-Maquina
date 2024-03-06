import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# conjuntos de datos en scikit-learn
datasets_list = [
    ('Cáncer de Seno', datasets.load_breast_cancer()),
    ('Iris', datasets.load_iris()),
    ('Diabetes', datasets.load_diabetes()),
    ('Vino', datasets.load_wine())
]

# obtener la descripción y las características del conjunto de datos

def print_dataset_info(name, dataset):
    print(f"Información del conjunto de datos: {name}")
    print("Descripción:")
    print(dataset.DESCR)
    print("Características del conjunto de datos:")
    print(dataset.feature_names)
    print(f"Número de instancias: {dataset.data.shape[0]}")
    print(f"Número de atributos: {dataset.data.shape[1]}")
    try:
        print(f"Nombres de las clases: {dataset.target_names}")
    except AttributeError:
        print("Este conjunto de datos no tiene nombres de clase.")
    print()

# graficar el conjunto de datos y su representación PCA para Iris
def plot_dataset_and_pca(name, dataset):
    if name == 'Iris':
        colors = ['r', 'b', 'y']
    else:
        colors = plt.cm.Set1.colors

    # gráfico de dispersión del conjunto de datos
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset.data[:, 0], dataset.data[:, 1], c=dataset.target, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel(dataset.feature_names[0])
    plt.ylabel(dataset.feature_names[1])
    plt.title(f"Gráfico de dispersión del conjunto de datos: {name}")

    if name == 'Iris':
        for i, color in enumerate(colors):
            plt.scatter(dataset.data[dataset.target == i, 0], dataset.data[dataset.target == i, 1],
                        c=color, edgecolor='k', label=dataset.target_names[i])
        plt.legend(loc='best')
    
    plt.show()

    # Representación PCA
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    X_reduced = PCA(n_components=3).fit_transform(dataset.data)
    
    if name == 'Iris':
        for i, color in enumerate(colors):
            ax.scatter(X_reduced[dataset.target == i, 0], X_reduced[dataset.target == i, 1], X_reduced[dataset.target == i, 2],
                       c=color, edgecolor='k', label=dataset.target_names[i])
        ax.legend(loc='best')
    else:
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=dataset.target, cmap=plt.cm.Set1, edgecolor='k')
    
    ax.set_title(f"Representación PCA del conjunto de datos: {name}")
    ax.set_xlabel("1er Componente Principal")
    ax.set_ylabel("2do Componente Principal")
    ax.set_zlabel("3er Componente Principal")
    plt.show()

# iterar sobre los conjuntos de datos
for name, dataset in datasets_list:
    print_dataset_info(name, dataset)
    plot_dataset_and_pca(name, dataset)