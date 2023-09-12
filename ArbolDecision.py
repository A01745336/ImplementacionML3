import numpy as np
from sklearn.model_selection import train_test_split


# Definir la estructura de un nodo del árbol
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None,
                 right=None, value=None):
        # Índice de la característica a considerar en este nodo
        self.feature_index = feature_index
        # Umbral para tomar la decisión en este nodo
        self.threshold = threshold
        # Subárbol izquierdo (menor o igual al umbral)
        self.left = left
        # Subárbol derecho (mayor que el umbral)
        self.right = right
        # Valor de la clase si este nodo es una hoja
        self.value = value


# Función para calcular la impureza de Gini
def gini_impurity(y):
    m = len(y)
    if m == 0:
        return 0
    p_1 = sum(y) / m
    p_0 = 1 - p_1
    return 1 - (p_0 ** 2 + p_1 ** 2)


# Función para dividir el conjunto de datos en dos grupos según una
# característica y umbral dados
def split_data(X, y, feature_index, threshold):
    left_X, left_y, right_X, right_y = [], [], [], []
    for i in range(len(X)):
        if X[i][feature_index] <= threshold:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])
    return left_X, left_y, right_X, right_y


# Función para encontrar la mejor división para un conjunto de datos y
# etiquetas
def find_best_split(X, y):
    best_gini = 1  # Gini inicial máximo
    best_feature_index = None
    best_threshold = None

    for feature_index in range(len(X[0])):
        unique_values = list(set([x[feature_index] for x in X]))
        unique_values.sort()

        for i in range(1, len(unique_values)):
            threshold = (unique_values[i - 1] + unique_values[i]) / 2
            left_X, left_y, right_X, right_y = split_data(X, y, feature_index,
                                                          threshold)

            gini_left = gini_impurity(left_y)
            gini_right = gini_impurity(right_y)
            gini = ((len(left_y) * gini_left + len(right_y) * gini_right)
                    / len(y))

            if gini < best_gini:
                best_gini = gini
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold


# Función para construir el árbol de decisión recursivamente
def build_tree(X, y, depth=0, max_depth=None, max_features=None):
    if depth == max_depth or len(set(y)) == 1 or max_features == 0:
        value = max(set(y), key=y.count)
        return TreeNode(value=value)

    feature_index, threshold = find_best_split(X, y)
    if feature_index is None or threshold is None:
        value = max(set(y), key=y.count)
        return TreeNode(value=value)

    left_X, left_y, right_X, right_y = split_data(X, y, feature_index, threshold)
    left_subtree = build_tree(left_X, left_y, depth + 1, max_depth, max_features - 1)
    right_subtree = build_tree(right_X, right_y, depth + 1, max_depth, max_features - 1)

    return TreeNode(feature_index=feature_index, threshold=threshold,
                    left=left_subtree, right=right_subtree)

# Función para hacer predicciones con el árbol construido
def predict(tree, x):
    if tree.left is None and tree.right is None:
        return tree.value
    if x[tree.feature_index] <= tree.threshold:
        return predict(tree.left, x)
    else:
        return predict(tree.right, x)




""" def print_tree(node, depth=0):
    if node is None:
        return
    print(" " * depth, "Depth:", depth, "Feature:", node.feature_index,
          "Threshold:", node.threshold, "Value:", node.value)
    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1) """


# Ejemplo de uso
""" if __name__ == "__main__":
    # Ejemplo de datos de entrada (características y etiquetas)
    X = np.array([
                [6.5, 4.5],
                [4.5, 3.5],
                [6.0, 5.0],
                [8.0, 2.5],
                [9.0, 6.0],
                [4.0, 7.0],
                [7.0, 5.5],
                [5.0, 6.5],
                [3.0, 4.0],
                [2.0, 5.0],
                [7.5, 3.0],
                [6.0, 8.0],
                [8.5, 7.0],
                [1.5, 4.5]
                ])
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0])

    # Construir el árbol de decisión
    tree = build_tree(X, y, max_depth=3)

    # Ejemplo de nuevos datos de entrada (características)
    # Si quiere agregar más predicciones, use el formato que se tiene
    X_new = np.array([
        [4.0, 3.5],
        [7.0, 5.5],
        [5.0, 10.0],
        [7.1, 4.0]
    ])

    # Hacer predicciones utilizando el nuevo árbol construido
    for i in range(len(X_new)):
        prediction_new = predict(tree, X_new[i])
        print("Predicción para la entrada", i+1, ":", prediction_new)
 """