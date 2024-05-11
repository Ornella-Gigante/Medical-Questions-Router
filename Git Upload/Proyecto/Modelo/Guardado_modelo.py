
"""
GUARDADO DEL MODELO ENTRENADO

El siguiente código guarda un modelo de aprendizaje automático entrenado en una carpeta específica 
utilizando la biblioteca joblib. Primero, se importan las bibliotecas necesarias y el modelo entrenado. Luego, se especifica la carpeta de destino donde se guardará el modelo.
Finalmente, se utiliza joblib.dump() para guardar el modelo en el archivo especificado.

"""
import os
import json
from transformers import AutoTokenizer
import sys 
sys.path.append(r"C:\Users\Ornella Gigante\enrutador_preguntas_enfermedades\Proyecto")
from Proyecto.Modelo.Entrenamiento_Interfaz import model


class ModelSaver:
    def __init__(self, model):
        self.model = model

    def save_model(self, destination_folder):
        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Save the model configuration
        model_config = self.model.config.to_dict()
        with open(os.path.join(destination_folder, 'config.json'), 'w') as config_file:
            json.dump(model_config, config_file)

        # Save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        tokenizer.save_pretrained(destination_folder)

        # Save other model-related files or metadata if needed
        # For example, save training logs, evaluation metrics, etc.

        print("Model saved successfully to:", destination_folder)

# Ruta de destino
destination_folder = r"C:\Users\Ornella Gigante\OneDrive\Escritorio\enrutador_preguntas_enfermedades\modelos_creados"

# Crear una instancia del ModelSaver
saver = ModelSaver(model)

# Guardar el modelo
saver.save_model(destination_folder)

