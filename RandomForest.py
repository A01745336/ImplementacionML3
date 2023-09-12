from ArbolDecision import TreeNode, build_tree, predict
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter


class RandomForest:
    def __init__(self, num_trees=50, max_depth=None, max_features=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.num_trees):
            # Seleccionar un subconjunto aleatorio de características
            random_feature_indices = np.random.choice(len(X[0]), size=self.max_features, replace=False)
            X_train_subset = X[:, random_feature_indices]

            # Seleccionar un subconjunto aleatorio de datos
            random_indices = np.random.choice(len(X_train_subset), size=len(X_train_subset), replace=True)
            X_train_subset = X_train_subset[random_indices]
            y_train_subset = y[random_indices]

            # Construir un árbol de decisión usando tu implementación
            tree = build_tree(X_train_subset, y_train_subset, max_depth=self.max_depth, max_features=self.max_features)

            self.trees.append(tree)

    def predict(self, X):
        predictions = [self.predict_one(x) for x in X]
        return predictions

    def predict_one(self, x):
        tree_predictions = [predict(tree, x) for tree in self.trees]
        # Realiza un conteo de las predicciones y selecciona la más común como resultado
        most_common_prediction = Counter(tree_predictions).most_common(1)[0][0]
        return most_common_prediction

    def evaluate(self, X, y):
        predictions = [self.predict(x) for x in X]
        accuracy = accuracy_score(y, predictions)
        print(f'Precisión del Random Forest: {accuracy:.4f}')

    def score(self, X, y):
        # Puedes utilizar una métrica como la precisión o cualquier otra adecuada
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)  # Ejemplo de uso de la precisión
        return accuracy
    
    def get_params(self, deep=True):
        return {
            'num_trees': self.num_trees,
            'max_depth': self.max_depth,
            'max_features': self.max_features
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
