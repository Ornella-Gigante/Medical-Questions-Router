
"""
UTILIZACIÓN DE MODELO-PRE ENTRENADO EN DATASETS Y CONFIGURACIÓN DE INTERFAZ

Este script implementa un enrutador de preguntas sobre enfermedades utilizando modelos de procesamiento de lenguaje natural (NLP) basados en la arquitectura BERT (Bidirectional Encoder Representations from Transformers). El enrutador responde a preguntas relacionadas con enfermedades proporcionando respuestas relevantes obtenidas de un conjunto de datos de entrenamiento previamente etiquetado.

El flujo de trabajo del script es el siguiente:

1.Importación de bibliotecas: Importa las bibliotecas necesarias, como Tkinter para la GUI, pandas para el manejo de datos, y nltk para procesamiento de lenguaje natural, entre otras.

2.Carga de datos: Lee conjuntos de datos que contienen preguntas y respuestas sobre enfermedades desde archivos CSV.

3.Preprocesamiento de datos: Aplica algunas funciones de preprocesamiento a los datos, como convertir el texto a minúsculas y eliminar caracteres no deseados.

4.Mapeo de preguntas a respuestas: Crea un diccionario que mapea preguntas en inglés y español a sus respectivas respuestas.

5.Funciones de manejo de preguntas y respuestas: Define funciones para manejar preguntas ingresadas por el usuario. Estas funciones determinan el idioma de la pregunta, generan respuestas coherentes basadas en el modelo DialoGPT, y buscan respuestas similares en el conjunto de datos si no se encuentra una coincidencia exacta.

6.Interfaz gráfica de usuario (GUI): Crea una ventana de GUI utilizando Tkinter con campos de entrada para preguntas, botones para enviar preguntas, y áreas de texto para mostrar respuestas. También hay un botón para generar y documentar respuestas médicas como archivos JSON.

7.Generación de prompts médicos: Define una función para generar prompts médicos basados en las preguntas y respuestas proporcionadas.

"""

import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import re
import torch
import nltk
from transformers import BertTokenizer, BertForSequenceClassification
import langdetect
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import json


# NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Cargando los datasets transformados
train_data_1_csv = pd.read_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\training1.csv")
train_data_2_csv = pd.read_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\training2.csv")
test_questions_data_csv = pd.read_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\test1.csv")
test_data_csv = pd.read_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\test2.csv")
qrels_data_csv = pd.read_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\test3.csv")


# Combinando el data
all_data = pd.concat([train_data_1_csv, train_data_2_csv, test_questions_data_csv, test_data_csv], ignore_index=True)

# Pre-procesamiento
def preprocess_text(text):
    if isinstance(text, str):  
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.strip()
    else:
        return 'Not Answer'

all_data["SUBJECT"] = all_data["SUBJECT"].apply(preprocess_text)


# Mapeo 
question_to_answer_en = {}
question_to_answer_es = {}

# Incorporando nuevas preguntas y respuestas en inglés

question_to_answer_en.update({
    "What are the symptoms of COVID-19?": "Common symptoms of COVID-19 include fever, cough, and shortness of breath.",
    "How is diabetes diagnosed?": "Diabetes is typically diagnosed through blood tests that measure blood glucose levels.",
    "What are the symptoms of the flu?": "Symptoms of the flu include fever, chills, sore throat, muscle aches, and fatigue.",
    "How is hypertension treated?": "Hypertension can be treated with lifestyle changes and medications.",
    "What are the symptoms of asthma?": "Symptoms of asthma include wheezing, coughing, chest tightness, and shortness of breath.",
    "How is arthritis managed?": "Arthritis can be managed with medication, physical therapy, and lifestyle changes.",
    "What are the symptoms of depression?": "Symptoms of depression include persistent sadness, loss of interest or pleasure in activities, and changes in sleep or appetite.",
    "How is anxiety treated?": "Anxiety can be treated with therapy, medication, or a combination of both.",
    "What are the symptoms of Alzheimer's disease?": "Symptoms of Alzheimer's disease include memory loss, confusion, and difficulty with language and reasoning.",
    "How is cancer diagnosed?": "Cancer can be diagnosed through imaging tests, biopsies, and blood tests.",
    "What are the symptoms of Parkinson's disease?": "Symptoms of Parkinson's disease include tremors, stiffness, and slowness of movement.",
    "How is multiple sclerosis managed?": "Multiple sclerosis can be managed with medications to control symptoms and therapies to improve function.",
    "What are the symptoms of heart disease?": "Symptoms of heart disease can include chest pain, shortness of breath, and fatigue.",
    "How is celiac disease diagnosed?": "Celiac disease is diagnosed through blood tests and a biopsy of the small intestine.",
    "What are the symptoms of migraines?": "Symptoms of migraines include severe headache, nausea, and sensitivity to light and sound.",
    "How is epilepsy treated?": "Epilepsy can be treated with medications, surgery, or other therapies.",
    "What are the symptoms of fibromyalgia?": "Symptoms of fibromyalgia include widespread pain, fatigue, and cognitive difficulties.",
    "How is chronic obstructive pulmonary disease managed?": "COPD can be managed with medications, pulmonary rehabilitation, and lifestyle changes.",
    "What are the symptoms of irritable bowel syndrome?": "Symptoms of irritable bowel syndrome include abdominal pain, bloating, and changes in bowel habits.",
    "How is rheumatoid arthritis treated?": "Rheumatoid arthritis can be treated with medications to reduce inflammation and slow joint damage.",
    "What are the symptoms of lupus?": "Symptoms of lupus include joint pain, fatigue, skin rashes, and fever.",
    "How is Crohn's disease managed?": "Crohn's disease can be managed with medications, lifestyle changes, and surgery in some cases.",
    "What are the symptoms of kidney stones?": "Symptoms of kidney stones include severe pain in the back or side, blood in the urine, and nausea.",
    "How is osteoporosis diagnosed?": "Osteoporosis is diagnosed through bone density tests, such as DXA scans.",
    "What are the symptoms of endometriosis?": "Symptoms of endometriosis include pelvic pain, painful periods, and infertility.",
    "How is sleep apnea treated?": "Sleep apnea can be treated with CPAP therapy, oral appliances, or surgery.",
    "What are the symptoms of irritable bladder syndrome?": "Symptoms of irritable bladder syndrome include urinary urgency, frequency, and bladder pain.",
    "How is glaucoma diagnosed?": "Glaucoma is diagnosed through a comprehensive eye exam that includes measuring eye pressure.",
    "What are the symptoms of hypothyroidism?": "Symptoms of hypothyroidism include fatigue, weight gain, and sensitivity to cold.",
    "How is fibroids treated?": "Fibroids can be treated with medication, non-invasive procedures, or surgery.",
    "What are the symptoms of allergies?": "Common symptoms of allergies include sneezing, runny or stuffy nose, and itchy eyes.",
    "How is anxiety diagnosed?": "Anxiety is diagnosed through a combination of physical exams, psychological evaluations, and discussions of symptoms.",
    "What are the symptoms of anxiety attacks?": "Symptoms of anxiety attacks include sudden feelings of intense fear or panic, rapid heartbeat, and difficulty breathing.",
    "How is depression treated?": "Depression can be treated with therapy, medication, or a combination of both.",
    "What are the symptoms of bipolar disorder?": "Symptoms of bipolar disorder include extreme mood swings, changes in energy levels, and difficulty concentrating.",
    "How is bipolar disorder diagnosed?": "Bipolar disorder is diagnosed through a comprehensive psychiatric evaluation, including a discussion of symptoms and medical history.",
    "What are the symptoms of schizophrenia?": "Symptoms of schizophrenia include hallucinations, delusions, disorganized thinking, and social withdrawal.",
    "How is schizophrenia treated?": "Schizophrenia is typically treated with antipsychotic medications, therapy, and support services.",
    "What are the symptoms of attention deficit hyperactivity disorder (ADHD)?": "Symptoms of ADHD include inattention, hyperactivity, and impulsivity.",
    "How is ADHD diagnosed?": "ADHD is diagnosed through a thorough evaluation that includes gathering information from parents, teachers, and other caregivers.",
    "What are the symptoms of post-traumatic stress disorder (PTSD)?": "Symptoms of PTSD include flashbacks, nightmares, severe anxiety, and uncontrollable thoughts about the traumatic event.",
    "How is PTSD treated?": "PTSD can be treated with therapy, medication, or a combination of both.",
    "What are the symptoms of obsessive-compulsive disorder (OCD)?": "Symptoms of OCD include repetitive thoughts or behaviors, such as excessive cleaning or checking.",
    "How is OCD diagnosed?": "OCD is diagnosed based on the presence of obsessions, compulsions, or both, which interfere with daily life.",
    "What are the symptoms of panic disorder?": "Symptoms of panic disorder include sudden attacks of fear or panic, along with physical symptoms like sweating and heart palpitations.",
    "How is panic disorder treated?": "Panic disorder can be treated with therapy, medication, or a combination of both.",
    "What are the symptoms of social anxiety disorder?": "Symptoms of social anxiety disorder include intense fear of social situations, avoidance of social interactions, and physical symptoms like sweating and trembling.",
    "How is social anxiety disorder diagnosed?": "Social anxiety disorder is diagnosed based on the presence of persistent and excessive fear or anxiety about social situations.",
    "What are the symptoms of borderline personality disorder?": "Symptoms of borderline personality disorder include unstable relationships, impulsive behavior, and intense mood swings.",
    "How is borderline personality disorder treated?": "Borderline personality disorder is typically treated with therapy, such as dialectical behavior therapy (DBT), and sometimes medication.",

})


# Incorporando nuevas preguntas y respuestas en español
question_to_answer_es.update({
    "¿Cuáles son los síntomas del COVID-19?": "Los síntomas comunes del COVID-19 incluyen fiebre, tos y dificultad para respirar.",
    "¿Cómo se diagnostica la diabetes?": "La diabetes suele diagnosticarse a través de análisis de sangre que miden los niveles de glucosa en sangre.",
    "¿Cuáles son los síntomas de la gripe?": "Los síntomas de la gripe incluyen fiebre, escalofríos, dolor de garganta, dolores musculares y fatiga.",
    "¿Cómo se trata la hipertensión?": "La hipertensión se puede tratar con cambios en el estilo de vida y medicamentos.",
    "¿Cuáles son los síntomas del asma?": "Los síntomas del asma incluyen sibilancias, tos, opresión en el pecho y dificultad para respirar.",
    "¿Cómo se maneja la artritis?": "La artritis se puede manejar con medicamentos, fisioterapia y cambios en el estilo de vida.",
    "¿Cuáles son los síntomas de la depresión?": "Los síntomas de la depresión incluyen tristeza persistente, pérdida de interés o placer en las actividades y cambios en el sueño o el apetito.",
    "¿Cómo se trata la ansiedad?": "La ansiedad se puede tratar con terapia, medicamentos o una combinación de ambos.",
    "¿Cuáles son los síntomas de la enfermedad de Alzheimer?": "Los síntomas de la enfermedad de Alzheimer incluyen pérdida de memoria, confusión y dificultad con el lenguaje y el razonamiento.",
    "¿Cómo se diagnostica el cáncer?": "El cáncer se puede diagnosticar mediante pruebas de imagen, biopsias y análisis de sangre.",
    "¿Cuáles son los síntomas de la enfermedad de Parkinson?": "Los síntomas de la enfermedad de Parkinson incluyen temblores, rigidez y lentitud de movimiento.",
    "¿Cómo se maneja la esclerosis múltiple?": "La esclerosis múltiple se puede manejar con medicamentos para controlar los síntomas y terapias para mejorar la función.",
    "¿Cuáles son los síntomas de la enfermedad cardíaca?": "Los síntomas de la enfermedad cardíaca pueden incluir dolor en el pecho, dificultad para respirar y fatiga.",
    "¿Cómo se diagnostica la enfermedad celíaca?": "La enfermedad celíaca se diagnostica mediante análisis de sangre y una biopsia del intestino delgado.",
    "¿Cuáles son los síntomas de las migrañas?": "Los síntomas de las migrañas incluyen dolor de cabeza intenso, náuseas y sensibilidad a la luz y al sonido.",
    "¿Cómo se trata la epilepsia?": "La epilepsia se puede tratar con medicamentos, cirugía u otras terapias.",
    "¿Cuáles son los síntomas de la fibromialgia?": "Los síntomas de la fibromialgia incluyen dolor generalizado, fatiga y dificultades cognitivas.",
    "¿Cómo se maneja la enfermedad pulmonar obstructiva crónica?": "La EPOC se puede manejar con medicamentos, rehabilitación pulmonar y cambios en el estilo de vida.",
    "¿Cuáles son los síntomas del síndrome del intestino irritable?": "Los síntomas del síndrome del intestino irritable incluyen dolor abdominal, distensión y cambios en los hábitos intestinales.",
    "¿Cómo se trata la artritis reumatoide?": "La artritis reumatoide se puede tratar con medicamentos para reducir la inflamación y ralentizar el daño articular.",
    "¿Cuáles son los síntomas del lupus?": "Los síntomas del lupus incluyen dolor en las articulaciones, fatiga, erupciones cutáneas y fiebre.",
    "¿Cómo se maneja la enfermedad de Crohn?": "La enfermedad de Crohn se puede manejar con medicamentos, cambios en el estilo de vida y cirugía en algunos casos.",
    "¿Cuáles son los síntomas de los cálculos renales?": "Los síntomas de los cálculos renales incluyen dolor intenso en la espalda o el costado, sangre en la orina y náuseas.",
    "¿Cómo se diagnostica la osteoporosis?": "La osteoporosis se diagnostica mediante pruebas de densidad ósea, como las exploraciones DXA.",
    "¿Cuáles son los síntomas de la endometriosis?": "Los síntomas de la endometriosis incluyen dolor pélvico, menstruaciones dolorosas e infertilidad.",
    "¿Cómo se trata la apnea del sueño?": "La apnea del sueño se puede tratar con terapia CPAP, dispositivos bucales o cirugía.",
    "¿Cuáles son los síntomas del síndrome de la vejiga hiperactiva?": "Los síntomas del síndrome de la vejiga hiperactiva incluyen urgencia urinaria, frecuencia y dolor vesical.",
    "¿Cómo se diagnostica el glaucoma?": "El glaucoma se diagnostica mediante un examen ocular completo que incluye la medición de la presión ocular.",
    "¿Cuáles son los síntomas del hipotiroidismo?": "Los síntomas del hipotiroidismo incluyen fatiga, aumento de peso y sensibilidad al frío.",
    "¿Cómo se tratan los miomas?": "Los miomas se pueden tratar con medicamentos, procedimientos no invasivos o cirugía.",
    "¿Cuáles son los síntomas de las alergias?": "Los síntomas comunes de las alergias incluyen estornudos, nariz congestionada o con mucosidad y picazón en los ojos.",
    "¿Cómo se diagnostica la ansiedad?": "La ansiedad se diagnostica mediante una combinación de exámenes físicos, evaluaciones psicológicas y discusiones sobre los síntomas.",
    "¿Cuáles son los síntomas de los ataques de ansiedad?": "Los síntomas de los ataques de ansiedad incluyen sentimientos repentinos de miedo o pánico intenso, taquicardia y dificultad para respirar.",
    "¿Cómo se trata la depresión?": "La depresión se puede tratar con terapia, medicamentos o una combinación de ambos.",
    "¿Cuáles son los síntomas del trastorno bipolar?": "Los síntomas del trastorno bipolar incluyen cambios extremos en el estado de ánimo, cambios en los niveles de energía y dificultad para concentrarse.",
    "¿Cómo se diagnostica el trastorno bipolar?": "El trastorno bipolar se diagnostica mediante una evaluación psiquiátrica completa, que incluye una discusión de los síntomas y la historia médica.",
    "¿Cuáles son los síntomas de la esquizofrenia?": "Los síntomas de la esquizofrenia incluyen alucinaciones, delirios, pensamiento desorganizado y retraimiento social.",
    "¿Cómo se trata la esquizofrenia?": "La esquizofrenia se trata típicamente con medicamentos antipsicóticos, terapia y servicios de apoyo.",
    "¿Cuáles son los síntomas del trastorno por déficit de atención e hiperactividad (TDAH)?": "Los síntomas del TDAH incluyen falta de atención, hiperactividad e impulsividad.",
    "¿Cómo se diagnostica el TDAH?": "El TDAH se diagnostica mediante una evaluación exhaustiva que incluye recopilación de información de padres, maestros y otros cuidadores.",
    "¿Cuáles son los síntomas del trastorno de estrés postraumático (TEPT)?": "Los síntomas del TEPT incluyen flashbacks, pesadillas, ansiedad severa y pensamientos incontrolables sobre el evento traumático.",
    "¿Cómo se trata el TEPT?": "El TEPT se puede tratar con terapia, medicamentos o una combinación de ambos.",
    "¿Cuáles son los síntomas del trastorno obsesivo-compulsivo (TOC)?": "Los síntomas del TOC incluyen pensamientos o comportamientos repetitivos, como limpieza excesiva o revisión.",
    "¿Cómo se diagnostica el TOC?": "El TOC se diagnostica en función de la presencia de obsesiones, compulsiones o ambas, que interfieren con la vida diaria.",
    "¿Cuáles son los síntomas del trastorno de pánico?": "Los síntomas del trastorno de pánico incluyen ataques repentinos de miedo o pánico, junto con síntomas físicos como sudoración y palpitaciones.",
    "¿Cómo se trata el trastorno de pánico?": "El trastorno de pánico se puede tratar con terapia, medicamentos o una combinación de ambos.",
    "¿Cuáles son los síntomas del trastorno de ansiedad social?": "Los síntomas del trastorno de ansiedad social incluyen miedo intenso a situaciones sociales, evitación de interacciones sociales y síntomas físicos como sudoración y temblores.",
    "¿Cómo se diagnostica el trastorno de ansiedad social?": "El trastorno de ansiedad social se diagnostica en función de la presencia de miedo o ansiedad persistente y excesiva sobre situaciones sociales.",
    "¿Cuáles son los síntomas del trastorno límite de la personalidad?": "Los síntomas del trastorno límite de la personalidad incluyen relaciones inestables, comportamiento impulsivo y cambios de humor intensos.",
    "¿Cómo se trata el trastorno límite de la personalidad?": "El trastorno límite de la personalidad se trata típicamente con terapia, como la terapia dialéctica conductual (TDC), y a veces medicación.",
})
    


for index, row in train_data_1_csv.iterrows():
    question_to_answer_en[row['SUBJECT']] = row['MESSAGE']
    question_to_answer_es[row['SUBJECT']] = row['MESSAGE']
for index, row in train_data_2_csv.iterrows():
    question_to_answer_en[row['SUBJECT']] = row['MESSAGE']
    question_to_answer_es[row['SUBJECT']] = row['MESSAGE']
for index, row in test_questions_data_csv.iterrows():
    question_to_answer_en[row['Original-Question']] = row['NIST-PARAPHRASE']
    question_to_answer_es[row['Original-Question']] = row['NIST-PARAPHRASE']
for index, row in test_data_csv.iterrows():
    question_to_answer_en[row['Original-Question']] = row['NIST-PARAPHRASE']
    question_to_answer_es[row['Original-Question']] = row['NIST-PARAPHRASE']

# Minimo de longitud de la pregunta, de lo contrario, pedirá más información 
LENGTH_THRESHOLD = 5

# Tokenizador DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForQuestionAnswering.from_pretrained("microsoft/DialoGPT-large")

# Procesamiento de la pregunta y luego respuesta 
# Pasa todo a formato JSON y luego devuelve un documento con los inputs y outputs en dicho formato
def handle_question():
    question = question_entry.get("1.0", "end-1c")
    if question.strip() == "":
        response_text.insert("end", "Please enter a question.\n")
    else:
        language = detect_language(question)
        if language == "es":
            predicted_text = generate_coherent_response_es(question)
        elif language == "en":
            predicted_text = generate_coherent_response_en(question)
        else:
            predicted_text = "Sorry, I couldn't understand the language of the question or I need more information."
        response_text.insert("end", f"Question: {question}\nAnswer: {predicted_text}\n\n")
        # Convert question and answer to JSON format
        question_json = {'question': question, 'language': language}
        response_json = {'answer': predicted_text}
        # Save data to JSON files
        with open('input.json', 'w') as f:
            json.dump(question_json, f)
        with open('output.json', 'w') as f:
            json.dump(response_json, f)
        question_entry.delete("1.0", "end")


#Respuestas en ESPAÑOL y en INGLÉS

def generate_coherent_response_en(question):
    preprocessed_question = preprocess_text(question)
    if len(preprocessed_question.split()) < LENGTH_THRESHOLD:
        return "Please provide more information."
    else:
        if preprocessed_question in question_to_answer_en:
            return question_to_answer_en[preprocessed_question]
        else:
            most_similar_question = find_most_similar_question(preprocessed_question, question_to_answer_en.keys())
            if most_similar_question:
                return question_to_answer_en[most_similar_question]
            else:
                return "Sorry, I couldn't find a suitable answer for your question."
         

def generate_coherent_response_es(question):
    preprocessed_question = preprocess_text(question)
    if len(preprocessed_question.split()) < LENGTH_THRESHOLD:
        return "Por favor, dame más información."
    else:
        most_similar_question = find_most_similar_question(preprocessed_question, question_to_answer_es.keys())
        if most_similar_question:
            return question_to_answer_es[most_similar_question]
        else:
            return "Lo siento, no he encontrado información suficiente para tu respuesta."

# Con esta función se busca similitud entre la posible respuesta y la info en el dataset 
def find_most_similar_question(question, question_set):
    max_similarity = 0
    most_similar_question = None
    for q in question_set:
        sim_score = similarity_score(question, q)
        if sim_score > max_similarity:
            max_similarity = sim_score
            most_similar_question = q
    return most_similar_question

# Con esta función se detectaría la similitud entre 2 strings
def similarity_score(text1, text2):
    if isinstance(text1, str) and isinstance(text2, str):  
        words_text1 = set(text1.split())
        words_text2 = set(text2.split())
        intersection = len(words_text1.intersection(words_text2))
        union = len(words_text1.union(words_text2))
        return intersection / union if union != 0 else 0.0
    else:
        return 0.0  

# Con esta función se detecta el idioma del texto
def detect_language(text):
    try:
        lang = langdetect.detect(text)
        return lang
    except:
        return "unknown"
    

    # Función para generar prompts médicos
def generate_medical_prompts():
    prompt = """ 
  Eres un médico especialista en diagnóstico de enfermedades y estás revisando un sistema de inteligencia artificial para ayudar en diagnósticos.
Se te ha pedido que generes una lista de preguntas que podrían hacer los pacientes sobre diferentes enfermedades y sus respuestas.
Cada pregunta debe ser una frase o párrafo de entre 10 y 50 palabras.
Por favor, genera las preguntas y respuestas como un JSON para cada enfermedad.

Por ejemplo:
# Enfermedad
{"disease": "COVID-19"}
# Pregunta y respuesta
{
    "question": "¿Cuáles son los síntomas del COVID-19?",
    "answer": "Los síntomas comunes del COVID-19 incluyen fiebre, tos y dificultad para respirar."
}

# Enfermedad: Diabetes Tipo 2
{
    "disease": "Diabetes Tipo 2"
}
{
    "question": "¿Cuáles son los síntomas de la diabetes tipo 2?",
    "answer": "Los síntomas de la diabetes tipo 2 incluyen sed excesiva, aumento del hambre, fatiga y visión borrosa."
}

# Enfermedad: Hipertensión
{
    "disease": "Hipertensión"
}
{
    "question": "¿Qué factores pueden aumentar el riesgo de hipertensión?",
    "answer": "Factores como la obesidad, el consumo excesivo de sal, la falta de actividad física y el consumo de alcohol pueden aumentar el riesgo de hipertensión."
}

# Enfermedad: Asma
{
    "disease": "Asma"
}
{
    "question": "¿Cuáles son los desencadenantes comunes del asma?",
    "answer": "Los desencadenantes comunes del asma incluyen alérgenos como el polen, el pelo de mascotas, el humo del tabaco, el aire frío y el ejercicio físico intenso."
}

# Enfermedad: Artritis
{
    "disease": "Artritis"
}
{
    "question": "¿Cómo se puede manejar el dolor asociado con la artritis?",
    "answer": "El dolor asociado con la artritis se puede manejar mediante medicamentos antiinflamatorios, terapia física, ejercicio regular y técnicas de relajación."
}

# Enfermedad: Depresión
{
    "disease": "Depresión"
}
{
    "question": "¿Qué opciones de tratamiento están disponibles para la depresión?",
    "answer": "Las opciones de tratamiento para la depresión incluyen terapia psicológica, medicamentos antidepresivos, cambios en el estilo de vida y terapia de electroconvulsión en casos graves."
}

    \n
    """

    prompts = []

    for question, answer in question_to_answer_en.items():
        prompts.append({"question": question, "answer": answer})

    with open('medical_prompts.json', 'w') as f:
        json.dump(prompts, f)

    response_text.insert("end", "¡Prompts médicos generados y guardados exitosamente!\n")



# Crear la ventana de la interfaz gráfica de usuario (GUI)
root = tk.Tk()
root.title("👨‍⚕️Enrutador de preguntas y respuestas sobre enfermedades👩‍⚕️")
root.geometry("800x600")
root.configure(bg='purple')

# Etiqueta para la pregunta
question_label = tk.Label(root, text="INGRESE SU PREGUNTA:", bg='purple', fg='black', font=("Arial", 14))
question_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

# Entrada de texto para la pregunta
question_entry = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=5)
question_entry.grid(row=1, column=0, padx=10, pady=10, sticky='we')

# Botón para enviar la pregunta
ask_button = tk.Button(root, text="Enviar pregunta", command=handle_question, bg='purple', fg='white', font=("Arial", 14))
ask_button.grid(row=2, column=0, padx=10, pady=10, sticky='w')

# Etiqueta para la respuesta
response_label = tk.Label(root, text="RESPUESTA:", bg='purple', fg='black', font=("Arial", 14))
response_label.grid(row=3, column=0, padx=10, pady=10, sticky='w')

# Área de texto para la respuesta
response_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=10)
response_text.grid(row=4, column=0, padx=10, pady=10, sticky='we')

# Botón para generar prompts médicos
generate_prompts_button = tk.Button(root, text="Documenta rus respuestas", command=generate_medical_prompts, bg='purple', fg='white', font=("Arial", 14))
generate_prompts_button.grid(row=5, column=0, padx=10, pady=10, sticky='w')

# Emoticones de doctores
doctor_label = tk.Label(root, text="👩‍⚕️👨‍⚕️", bg='purple', fg='white', font=("Arial", 24))
doctor_label.grid(row=0, column=1, rowspan=6, padx=10, pady=10, sticky='ns')

# Centrar la ventana
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

root.mainloop()
