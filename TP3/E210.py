import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression

# Leer los datos
data = pd.read_csv('dataset01.csv')
X = data.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Leer y escalar los datos de y
with open('y1.txt', 'r') as file:
    lines = file.readlines()
y = np.array([float(line.strip()) for line in lines])
y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std

# Visualización de los datos
def visualizacion(X_scaled):
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_scaled, cmap="viridis", center=0)
    plt.show()
    return

def descomposicionSVD(X):
    # Descomposición SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U, S, Vt

def reducirMatriz(U,S,Vt,d):
    U_reduced = U[:, :d]  # Seleccionamos las primeras d columnas de U
    S_reduced = S[:d]  # Seleccionamos los primeros d valores singulares y formamos una matriz diagonal
    Vt_reduced = Vt[:d, :]  # Seleccionamos las primeras d filas de Vt
    X_reduced = np.dot(U_reduced, np.dot(np.diag(S_reduced), Vt_reduced))
    return X_reduced, U_reduced, S_reduced, Vt_reduced

def eucledean(X, d):
    # Calculamos la distancia euclidiana entre cada par de puntos en la matriz reducida
    distances = pdist(X, metric='euclidean')
    dist_matrix = squareform(distances)

    # Visualizamos los datos reducidos usando un heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, cmap="RdBu", center=0)
    plt.title(f'Heatmap de las Distancias Euclidianas entre Vectores Reducidos para d = {d}')
    plt.xlabel('Índice de Vector')
    plt.ylabel('Índice de Vector')
    plt.show()
    return

def valores(S): 
    # Graficamos los valores singulares
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(S)), S, color='skyblue')
    plt.xlabel('Índice del Valor Singular')
    plt.ylabel('Valor Singular')
    plt.show()
    return

def verb(b):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(b)), b, color='skyblue')
    plt.ylabel('Valor de Beta')
    plt.show()
    return

def cuadradosMinimos(X, U, S, Vt, y):

    # Preparo USVt para el cálculo de b
    Ut = np.transpose(U)
    V = np.transpose(Vt)
    S_inv = np.diag(1/S) # Inversa de Sr (convirtiendo Sr en una matriz diagonal y luego invirtiendo)

    # Cálculo de b
    b = V @ S_inv @ Ut @ y
    y_pred = np.dot(X, b)
    return y_pred, y, b

def visualizarValores(y_pred, y):
    # Visualización de los valores reales vs predichos
    plt.figure(figsize=(10, 6))
    plt.title("Visualizacion de valores predichos y valores reales")
    plt.scatter(range(len(y)), y, color='blue', label='y')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='y_pred')
    plt.xlabel('Índice de la Muestra')
    plt.ylabel('Valor de y')
    plt.legend()
    plt.show()
    return

def calcularError(y_pred, y):
    error = np.mean((y - y_pred) ** 2)
    return error

# Grafico la Matriz
visualizacion(X_scaled)
# Grafico X
eucledean(X_scaled, 'p')
# Descompongo
U, S, Vt = descomposicionSVD(X_scaled)
#reduzco y grafico para d = 10
X_reduced, U_reduced, S_reduced, Vt_reduced = reducirMatriz(U,S,Vt,10)
eucledean(X_reduced, 10)

# Reduzco y grafico para d = 6
X_reduced, U_reduced, S_reduced, Vt_reduced = reducirMatriz(U,S,Vt,6)
eucledean(X_reduced, 6)

# Reduzco y grafico para d = 2
X_reduced, U_reduced, S_reduced, Vt_reduced = reducirMatriz(U,S,Vt,2)
eucledean(X_reduced, 2)

# Grafico los valores singulares
valores(S)

# Calculo de b para cuadrados mínimos para X
y_pred, y, b = cuadradosMinimos(X_scaled, U, S, Vt, y)
visualizarValores(y_pred, y)
errorX = calcularError(y_pred, y)
print(f'Error Cuadratico medio para  X: {errorX}')

# Calculo de b para cuadrados mínimos para X reducida
y_pred, y, b = cuadradosMinimos(X_reduced, U_reduced, S_reduced, Vt_reduced, y)
visualizarValores(y_pred, y)
errorXreduced = calcularError(y_pred, y)
print(b)
verb(b)


print(f'Error Cuadratico medio para X reducida con d = 2 es: {errorXreduced}')
plt.figure(figsize=(10, 6))
plt.title("Comparación de errores cuadráticos medios para X y X reducida")
plt.bar(['X', 'X reducida'], [errorX, errorXreduced], color='skyblue')
plt.ylabel('Error Cuadrático Medio')
plt.show()