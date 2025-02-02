# 🏥 Intelligent Medical Questions Router (NLP)

## 🎯 Description
Intelligent system that uses NLP to diagnose diseases and answer questions about common symptoms, combining an intuitive interface with machine learning models trained on medical data.

## 📁 Project Structure
```
disease_question_router/
├── .vscode/
│   └── settings.json
├── data/
│   ├── datasets_download_url
│   ├── datasets_download_url (2)
│   └── TestDataset/
│       └── qrels (2).txt
└── src/
    ├── Main.py
    ├── Training_Interface.py
    └── paste-3.txt
```

## ⚙️ Requirements
```bash
pip install scikit-learn pandas numpy matplotlib seaborn mlflow
```

## 🚀 Execution

### 💻 User Interface
```bash
cd src
python app.py
```

### 🔬 Model Training
```bash
cd src
python train.py
```

## 📊 MLflow Dashboard
```bash
mlflow ui
```

## 🔑 Key Components

### 📊 Directories
- `.vscode/`: Visual Studio Code configuration
- `data/`: Training and evaluation datasets
- `src/`: Main source code
  - `data/`: Data processing
  - `features/`: Feature extraction
  - `models/`: ML models
  - `utils/`: General utilities

### 📜 Main Files
- `src/app.py`: User interface
- `src/train.py`: Model training

## 📈 MLflow
Integrated platform for managing:
- Training experiments
- Hyperparameters
- Metrics
- Model artifacts

## 📚 More Information
See the `docs/` folder for detailed documentation.

## Demo Video

You can watch a demonstration of the project in action by clicking on the image below:

[![Ejecución](https://i.vimeocdn.com/video/1850926811-9c914608397b06ba206f6cbd3c9a67fab371c385b8ad1ac21a46f8bf38c27a9e-d?mw=1200&mh=844&q=70)](https://vimeo.com/945483706?share=copy)





