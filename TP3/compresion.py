
from os import listdir, path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# Ruta de la carpeta donde están las imágenes
ruta_carpeta = 'TP3/TP 03 dataset imagenes'

imagenes_matrices = []  # Lista de matrices de las imágenes

lista_imagenes = []

for archivo in listdir(ruta_carpeta):
        if archivo.endswith('.jpeg'):  # Filtrar imágenes
            ruta_imagen = path.join(ruta_carpeta, archivo)
            imagen = Image.open(ruta_imagen)  # Convertir a escala de grises
            matriz_imagen = np.array(imagen)
            lista_imagenes.append(matriz_imagen)
            imagenes_matrices.append(matriz_imagen.flatten()) 

A = np.array(imagenes_matrices)

U, S, V = np.linalg.svd(A, full_matrices=False)
S = np.diag(S)

imagenes_reconstruidas_por_d = {}

# Reconstruir A_d para cada valor de d y almacenarlo en un diccionario
for d in range(1,17):
    Ud = U[:, :d]
    Sd = S[:d, :d]
    Vtd = V[:d, :]
    
    # Reconstrucción para el valor actual de d
    A_d = Ud @ Sd @ Vtd
    imagenes_reconstruidas = [A_d[i].reshape(lista_imagenes[i].shape) for i in range(len(lista_imagenes))]
    
    # Guardar la reconstrucción para el valor actual de d
    imagenes_reconstruidas_por_d[d] = imagenes_reconstruidas

# Calcula la norma de Frobenius para cada imagen original (esto será el "valor total" de cada imagen)
errores_originales = [np.linalg.norm(imagen, 'fro') for imagen in lista_imagenes]

# Definir el límite de error relativo promedio permitido (10%)
limite_error_promedio = 0.10

# Encontrar el valor mínimo de d que cumpla con el límite de error promedio
d_optimo = None

errores_relativos_promedio = {}

for d in range(1, min(A.shape) + 1):  # d varía desde 1 hasta el rango máximo permitido
    Ud = U[:, :d]
    Sd = S[:d, :d]
    Vtd = V[:d, :]
    
    # Reconstrucción para el valor actual de d
    A_d = Ud @ Sd @ Vtd
    
    # Calcular el error relativo promedio para este valor de d
    errores_relativos = []
    for i in range(len(lista_imagenes)):
        # Reshape para obtener la imagen reconstruida
        imagen_reconstruida = A_d[i].reshape(lista_imagenes[i].shape)
        
        # Calcular el error de Frobenius para esta imagen
        error_frobenius = np.linalg.norm(lista_imagenes[i] - imagen_reconstruida, 'fro')
        
        # Calcular el error relativo
        error_relativo = error_frobenius / errores_originales[i]
        errores_relativos.append(error_relativo)
    
    # Calcular el error promedio de Frobenius para este valor de d
    error_promedio = np.mean(errores_relativos)
    errores_relativos_promedio[d] = error_promedio
    
    # Verificar si el error promedio cumple con el límite de 10%
    if error_promedio <= limite_error_promedio:
        d_optimo = d
        break

print(f"El valor mínimo de d que asegura un error relativo promedio ≤ 10% es: d = {d_optimo}")

# graficar el error relativo promedio vs d

plt.plot(list(errores_relativos_promedio.keys()), list(errores_relativos_promedio.values()), marker='o')
plt.xlabel('d')
plt.ylabel('Error relativo promedio')
plt.title('Error relativo promedio vs d')
plt.grid()
plt.show()

