"""

Generación y Guardado del Informe de Clasificación

Este código guarda el informe de clasificación como un archivo de texto y el modelo entrenado como un archivo binario en 
carpetas específicas,asegurándose de que las carpetas de destino existan antes de intentar guardar los archivos.
Se guarda en archivo binario por una cuestión de tamaño del archivo. 


"""

import os
from sklearn.metrics import classification_report
import joblib
from Proyecto.Modelo.Entrenamiento_Interfaz import model

# Guardar el informe de clasificación en un archivo de texto
informe_clasificacion = classification_report()
carpeta_destino = r"C:\Users\Ornella Gigante\enrutador_preguntas_enfermedades\Informes_Clasificacion"

# Asegurarse de que la carpeta de destino exista
os.makedirs(carpeta_destino, exist_ok=True)

output_path = os.path.join(carpeta_destino, "reporte_modelo_1.txt")
with open(output_path, 'w') as f:
    f.write(informe_clasificacion)

# Guardar el modelo entrenado en la carpeta de salida
carpeta_destino = r"C:\Users\Ornella Gigante\enrutador_preguntas_enfermedades\modelos_listos"

# Asegurarse de que la carpeta de destino exista
os.makedirs(carpeta_destino, exist_ok=True)

model_output_path = os.path.join(carpeta_destino, 'trained_model.pkl')
joblib.dump(model, model_output_path)
