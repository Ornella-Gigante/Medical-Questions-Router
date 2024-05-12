ENRUTADOR DE PREGUNTAS DE ENFERMEDADES NLP 


Este Enrutador de Enfermedades Inteligente es una aplicación que utiliza técnicas de aprendizaje automáticos para diagnosticar enfermedades y responder preguntas sobre síntomas de enfermedades comunes. 
La aplicación consta de dos componentes principales: 
una interfaz de usuario y un modelo de aprendizaje automático entrenado con datos médicos.


===Estructura del Proyecto===

El proyecto se organiza de la siguiente manera:

enrutador_preguntas_enfermedades/
│
├── .vscode/
│   └── settings.json
│
├── data/
│   ├── datasets_descarga_url
│   ├── datasets_descarga_url (2)
│   └── TestDataset/
│       └── qrels (2).txt
│
└── src/
├── Main.py
├── Entrenamiento_Interfaz.py
└── paste-3.txt


A continuación, se explica cada uno de los archivos y directorios:
.vscode/settings.json
Este archivo contiene la configuración del editor de código Visual Studio Code para este proyecto específico.
data/
Este directorio contiene los conjuntos de datos utilizados para entrenar y evaluar el modelo de aprendizaje automático. Los archivos datasets_descarga_url y datasets_descarga_url (2) contienen las URLs de los conjuntos de datos. El subdirectorio TestDataset/ contiene un archivo qrels (2).txt con datos de prueba.
notebooks/
Este directorio contiene los cuadernos de Jupyter Notebook utilizados durante el desarrollo y experimentación del proyecto.
src/
Este directorio contiene el código fuente de la aplicación.
src/data/
Este subdirectorio contiene los módulos relacionados con la carga y preprocesamiento de los datos.
src/features/
Este subdirectorio contiene los módulos relacionados con la extracción y transformación de características a partir de los datos.
src/models/
Este subdirectorio contiene los módulos relacionados con la definición, entrenamiento y evaluación de los modelos de aprendizaje automático.
src/utils/
Este subdirectorio contiene módulos de utilidad para tareas comunes, como el registro de eventos, la configuración, etc.
src/app.py
Este archivo contiene el código para la interfaz de usuario de la aplicación. La interfaz permite a los usuarios ingresar síntomas y obtener un diagnóstico sugerido por el modelo de aprendizaje automático.
src/train.py
Este archivo contiene el código para entrenar el modelo de aprendizaje automático utilizando los conjuntos de datos proporcionados.

===Requisitos====
Para ejecutar esta aplicación, se deben instalar las siguientes bibliotecas de Python:
scikit-learn
pandas
numpy
matplotlib
seaborn
mlflow

Estas bibliotecas se pueden instalar utilizando pip:

pip install scikit-learn pandas numpy matplotlib seaborn mlflow

=====Ejecución====

**Interfaz de Usuario**

Para ejecutar la interfaz de usuario, navegue hasta el directorio src/ y ejecute el siguiente comando:

--> python app.py

Esto iniciará la aplicación y abrirá la interfaz de usuario en su navegador web predeterminado.
Entrenamiento del Modelo
Para entrenar el modelo de aprendizaje automático, navegue hasta el directorio src/ y ejecute el siguiente comando:


--> python train.py

Este script cargará los conjuntos de datos, entrenará el modelo y registrará los resultados del entrenamiento en MLflow.

===MLflow===

MLflow es una plataforma de código abierto para administrar el ciclo de vida del aprendizaje automático. En este proyecto, se utiliza MLflow para registrar los experimentos de entrenamiento del modelo, incluyendo los hiperparámetros, métricas y artefactos.
Para acceder a la interfaz web de MLflow, ejecute el siguiente comando en una terminal separada:

--> mlflow ui

Esto iniciará el servidor de MLflow y abrirá la interfaz web en su navegador predeterminado. Desde aquí, puede explorar los experimentos registrados, comparar los resultados y descargar los modelos entrenados.

===Documentación Adicional===
Para obtener más información sobre el uso y la implementación de esta aplicación, consulte la documentación adicional en el directorio docs/.

===Ejecutación===

## Video demostrativo

Puedes ver una demostración del proyecto en acción haciendo clic en la imagen a continuación:

[![Ver video demostrativo](https://i.vimeocdn.com/video/1850917414-b0cb3a0ad1703642604b657e44e0cd625dfcf0a54ee1b552b3f2e7fd9d7ae000-d?mw=80&q=85)](https://vimeo.com/1850917414)


