import tkinter as tk
from tkinter import ttk
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from RandomForest import RandomForest
from math import sqrt
from sklearn.model_selection import KFold
from matplotlib_venn import venn3


class RandomForestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Random Forest Classifier")

        # Crear widgets
        self.label = ttk.Label(root, text="Resultados de Random Forest")
        self.label.pack()

        self.textbox = tk.Text(root, height=20, width=50)
        self.textbox.pack(expand=True, fill=tk.BOTH)  # Hacer que el widget Text se ajuste automáticamente

        self.run_button = ttk.Button(root, text="Ejecutar Random Forest", command=self.run_random_forest)
        self.run_button.pack()

    def ejecutar_random_forest(self, X_train, y_train):
        # Construir un Random Forest y entrenarlo
        num_trees = 100
        max_depth = 11
        max_features = int(sqrt(X_train.shape[1]))
        random_forest = RandomForest(num_trees=num_trees, max_depth=max_depth, max_features=max_features)
        random_forest.fit(X_train, y_train)

        return random_forest

    def run_random_forest(self):
        self.textbox.delete(1.0, tk.END)  # Limpiar el área de texto

        # Cargar el conjunto de datos Titanic
        titanic = sns.load_dataset("titanic")

        # Preprocesamiento básico del conjunto de datos
        titanic.dropna(inplace=True)  # Eliminar filas con datos faltantes
        titanic = titanic[["pclass", "sex", "age", "sibsp", "parch", "fare", "survived"]]
        titanic["sex"] = titanic["sex"].map({"male": 0, "female": 1})

        # Dividir el conjunto de datos en características (X) y etiquetas (y)
        X = titanic.drop("survived", axis=1).values
        y = titanic["survived"].values

        # Realizar validación cruzada
        num_repeticiones = 5  # Cambia el número de repeticiones según tus necesidades
        kf = KFold(n_splits=num_repeticiones, shuffle=True, random_state=42)
        accuracies = []
        test_accuracies = []
        train_accuracies = []

        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Crear un diagrama de Venn para mostrar la división de datos
            plt.figure(figsize=(8, 6))
            venn_diagram = venn3(subsets=(len(X_train), len(X_test), 0, 
                                        len(X_train) - len(X_test), len(X_test), 0, 0),
                                 set_labels=('Entrenamiento', 'Prueba', 'Validación'))
            plt.title('División de Datos entre Conjuntos de Entrenamiento, Prueba y Validación')
            plt.show()

            print("valores de entrenamiento.  Ejecucion: ", i + 1)
            print(X_train[:10])
            print("clases de los valores de entrenamiento.  Ejecucion: ", i + 1)
            print(y_train[:10], "\n")
            print("valores de prueba.  Ejecucion: ", i + 1)
            print(X_test[:10])
            print("clases de los valores de prueba.  Ejecucion: ", i + 1)
            print(y_test[:10], "\n")

            self.textbox.insert(tk.END, f"Ejecución {i + 1}:\n")

            # Entrenar y evaluar el modelo
            random_forest = self.ejecutar_random_forest(X_train, y_train)

            # Hacer predicciones en el conjunto de prueba
            predictions = random_forest.predict(X_test)

            # Calcular la precisión
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)
            self.textbox.insert(tk.END, f"Precisión del Random Forest: {accuracy:.2f}\n")

            # Calcular la precisión en el conjunto de entrenamiento
            train_predictions = random_forest.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_predictions)
            train_accuracies.append(train_accuracy)

            # Calcular la precisión en el conjunto de prueba
            test_accuracy = accuracy_score(y_test, predictions)
            test_accuracies.append(test_accuracy)

            # Registrar las métricas en el área de texto
            self.textbox.insert(tk.END, f"Precisión en entrenamiento: {train_accuracy:.2f}\n")
            self.textbox.insert(tk.END, f"Precisión en prueba: {test_accuracy:.2f}\n")

            # Evaluar el ajuste del modelo
            if train_accuracy > test_accuracy:
                self.textbox.insert(tk.END, "El modelo podría estar sobreajustado.\n")
            elif train_accuracy < test_accuracy:
                self.textbox.insert(tk.END, "El modelo podría estar subajustado.\n")
            else:
                self.textbox.insert(tk.END, "El modelo parece estar bien ajustado.\n")

            # Mostrar el informe de clasificación
            classification_rep = classification_report(y_test, predictions)
            self.textbox.insert(tk.END, f"Informe de clasificación del Random Forest:\n{classification_rep}\n")

            # Crear la matriz de confusión
            cm = confusion_matrix(y_test, predictions)

            # Crear un DataFrame para la matriz de confusión
            df_cm = pd.DataFrame(cm, index=["0", "1"], columns=["0", "1"])

            # Visualizar la matriz de confusión
            plt.figure(figsize=(5, 5))
            sns.heatmap(df_cm, annot=True, cmap="Greens", fmt='.0f', cbar=False, annot_kws={"size": 14})
            plt.xlabel("Etiqueta Predicha")
            plt.ylabel("Etiqueta Real")
            plt.title("Matriz de Confusión del Random Forest")
            plt.show()

        # Calcular la precisión promedio
        mean_accuracy = np.mean(accuracies)
        self.textbox.insert(tk.END, f"Precisión promedio de todas las ejecuciones: {mean_accuracy:.2f}\n")

        # Crear un DataFrame para la tabla
        df = pd.DataFrame({
            'Ejecución': range(1, 6),
            'Precisión en Entrenamiento': train_accuracies,
            'Precisión en Prueba': test_accuracies
        })

        # Mostrar la tabla
        print(df)

        # Crear una gráfica de barras para mostrar la variación en precisión en prueba
        plt.figure(figsize=(8, 4))
        plt.bar(range(1, 6), test_accuracies, color='blue')
        plt.xlabel('Ejecución de Validación Cruzada')
        plt.ylabel('Precisión en Prueba')
        plt.title('Variación en Precisión en Prueba en Diferentes Ejecuciones')
        plt.xticks(range(1, 6))
        plt.ylim(0, 1)  # Establecer límites en el eje y entre 0 y 1
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = RandomForestApp(root)
    root.mainloop()
