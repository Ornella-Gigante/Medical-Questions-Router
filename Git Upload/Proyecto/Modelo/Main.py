"""

Esta clase llamada Main lo que hace es generar parámetros del modelo entrenado, heredar del modelo y luego ejecutar. 
Primero ejecutará la interfaz, donde se deberán hacer las preguntas al chatbot.
Luego ejecutará las métricas y parámetros y enviará por consola el link a la MLFLOW.
Este es un boceto de lo que sería un experimento del chatbot de preguntas médicas, puede verse en MLFLOW el respectivo dashboard y el nombre del experimento pero
las métricas y parámetros tiene valores predeterminados.
La idea es, en un futuro y con más tiempo, retomar el código e implementar la lógica adecuada para que las métricas devuelvan valores que sean acordes a la conducta
del chatbot sobre las preguntas del usuario. 


"""
import re
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments
from Entrenamiento_Interfaz import train_data_1_csv, train_data_2_csv, test_data_csv, test_questions_data_csv
import mlflow


class DiseaseQuestionRouter:
    def __init__(self, train_data_1_csv, train_data_2_csv, test_questions_data_csv, test_data_csv):
        self.train_data_1_csv = train_data_1_csv
        self.train_data_2_csv = train_data_2_csv
        self.test_questions_data_csv = test_questions_data_csv
        self.test_data_csv = test_data_csv
        
        # Inicializar MLflow
        
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Boceto-de-Experimento-Chatbot")
        port= mlflow.set_tracking_uri("http://localhost:5000")

    def preprocess_text(self, text):
        if isinstance(text, str):  
            text = text.lower()
            text = re.sub(r'[^a-z\s]', '', text)
            return text.strip()
        else:
            return 'Not Answer'
    
    def compute_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    def compute_f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred)
    
    def compute_mse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    
    def train_model(self, learning_rate, batch_size, num_epochs, smr, accuracy, f1, port):
        with mlflow.start_run():
            # Registrar parámetros
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("smr", smr)
            mlflow.log_param("accuracy", accuracy)
            mlflow.log_param("f1", f1)
            mlflow.log_param("Port", port)

            
            # Realizar el entrenamiento del modelo aquí
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            train_dataset = self.create_dataset(self.train_data_1_csv, self.train_data_2_csv, tokenizer)
            test_dataset = self.create_dataset(self.test_questions_data_csv, self.test_data_csv, tokenizer)
            
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                learning_rate=learning_rate,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )
            
            # Entrenar el modelo y calcular métricas
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = predictions.argmax(axis=-1)
                accuracy = self.compute_accuracy(labels, predictions)
                f1 = self.compute_f1_score(labels, predictions)
                mse = self.compute_mse(labels, predictions)
                return {"accuracy": accuracy, "f1_score": f1, "mse": mse}
            
            trainer.compute_metrics = compute_metrics
            trainer.train()
            
            # Guardar el modelo
            trainer.save_model("./model")
            mlflow.pytorch.log_model(trainer.model, "model")
            
            # Imprimir el link con la ruta del modelo
            print(f"Entrenamiento completado. Modelo guardado en:./model")
            
    def create_dataset(self, data1, data2, tokenizer):
        # Combinar los datos
        data = pd.concat([data1, data2], ignore_index=True)
        
        # Tokenizar los datos
        encoded_data = tokenizer(data["SUBJECT"].tolist(), padding=True, truncation=True, return_tensors='pt')
        
        # Crear el dataset
        dataset = torch.utils.data.TensorDataset(
            encoded_data.input_ids,
            encoded_data.attention_mask,
            torch.tensor(data["label"].tolist())
        )
        
        return dataset

# Crear una instancia de la clase DiseaseQuestionRouter
router = DiseaseQuestionRouter(train_data_1_csv, train_data_2_csv, test_questions_data_csv, test_data_csv)

# Entrenar el modelo
router.train_model(learning_rate=0.001, batch_size=32, num_epochs=10, smr=0.5, accuracy=0.8, f1=0.7, port= "5000")