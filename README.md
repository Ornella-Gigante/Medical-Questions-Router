# 🏥 Enrutador Inteligente de Preguntas Médicas (NLP)

## 🎯 Descripción
Sistema inteligente que utiliza NLP para diagnosticar enfermedades y responder preguntas sobre síntomas comunes, combinando una interfaz intuitiva con modelos de machine learning entrenados con datos médicos.

## 📁 Estructura del Proyecto
```
enrutador_preguntas_enfermedades/
├── .vscode/
│   └── settings.json
├── data/
│   ├── datasets_descarga_url
│   ├── datasets_descarga_url (2)
│   └── TestDataset/
│       └── qrels (2).txt
└── src/
    ├── Main.py
    ├── Entrenamiento_Interfaz.py
    └── paste-3.txt
```

## ⚙️ Requisitos
```bash
pip install scikit-learn pandas numpy matplotlib seaborn mlflow
```

## 🚀 Ejecución

### 💻 Interfaz de Usuario
```bash
cd src
python app.py
```

### 🔬 Entrenamiento del Modelo
```bash
cd src
python train.py
```

## 📊 MLflow Dashboard
```bash
mlflow ui
```

## 🔑 Componentes Clave

### 📊 Directorios
- `.vscode/`: Configuración de Visual Studio Code
- `data/`: Datasets de entrenamiento y evaluación
- `src/`: Código fuente principal
  - `data/`: Procesamiento de datos
  - `features/`: Extracción de características
  - `models/`: Modelos ML
  - `utils/`: Utilidades generales

### 📜 Archivos Principales
- `src/app.py`: Interfaz de usuario
- `src/train.py`: Entrenamiento del modelo

## 📈 MLflow
Plataforma integrada para gestionar:
- Experimentos de entrenamiento
- Hiperparámetros
- Métricas
- Artefactos del modelo

## 📚 Más Información
Consulte la carpeta `docs/` para documentación detallada.

## Video demostrativo

Puedes ver una demostración del proyecto en acción haciendo clic en la imagen a continuación:

[![Ejecución](https://i.vimeocdn.com/video/1850926811-9c914608397b06ba206f6cbd3c9a67fab371c385b8ad1ac21a46f8bf38c27a9e-d?mw=1200&mh=844&q=70)](https://vimeo.com/945483706?share=copy)





