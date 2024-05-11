
"""

EXPLORACIÓN DE LOS DATOS 


Este código utiliza la biblioteca Pandas de Python para leer datos desde archivos XML y CSV, guardarlos como archivos CSV y luego leer esos archivos CSV recién creados.
El código se divide en dos partes: 

1)Lectura de datos desde archivos XML y CSV:

*Utiliza pd.read_xml() para leer datos desde archivos XML.
*Utiliza pd.read_csv() para leer datos desde archivos CSV.
*Los datos de entrenamiento se leen desde archivos XML (training1.xml y training2.xml).
*Los datos de prueba se leen desde archivos XML (test1.xml y test2.xml) y un archivo CSV (test3.txt).
*En el caso del archivo CSV (test3.txt), se especifica el delimitador como \t (tabulación) ya que es un archivo de valores separados por tabulaciones.


2)Guardado de datos como archivos CSV:

Utiliza el método to_csv() para guardar los DataFrames como archivos CSV en la ubicación especificada.
Cada DataFrame se guarda en un archivo CSV correspondiente, con el mismo nombre que el archivo XML de origen pero con extensión .csv.
Lectura de los archivos CSV recién creados:

Después de guardar los datos como archivos CSV, los lee nuevamente utilizando pd.read_csv().
Esto asegura que los datos guardados se han creado correctamente y se pueden leer correctamente desde los archivos CSV.
Mostrar los primeros registros de cada DataFrame:

Utiliza el método head() para mostrar los primeros registros de cada DataFrame.
Esto sirve como una verificación rápida para asegurarse de que los datos se han leído correctamente y se almacenan correctamente en los DataFrames.
Al final, el código imprime un mensaje indicando que los archivos CSV se han guardado en la ubicación especificada y luego muestra los primeros registros de cada DataFrame para verificar el proceso de lectura y guardado.



"""



import pandas as pd

# Leer los datos de entrenamiento desde archivos XML
train_data_1 = pd.read_xml(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\training1.xml")
train_data_2 = pd.read_xml(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\training2.xml")

# Leer los datos de prueba desde archivos XML y CSV
test_questions_data = pd.read_xml(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\test1.xml")
test_data = pd.read_xml(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\test2.xml")
qrels_data = pd.read_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\test3.txt", delimiter="\t", header=None)

# Guardar los DataFrames como archivos CSV
train_data_1.to_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\training1.csv", index=False)
train_data_2.to_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\training2.csv", index=False)
test_questions_data.to_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\test1.csv", index=False)
test_data.to_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\test2.csv", index=False)
qrels_data.to_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\test3.csv", index=False)

# Mostrar la ubicación de los archivos CSV guardados
print("Los archivos CSV se han guardado en la ubicación especificada.")


# Leer los archivos CSV recién creados
train_data_1_csv = pd.read_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\training1.csv")
train_data_2_csv = pd.read_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\training2.csv")
test_questions_data_csv = pd.read_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\test1.csv")
test_data_csv = pd.read_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\test2.csv")
qrels_data_csv = pd.read_csv(r"C:\Users\Ornella Gigante\OneDrive\Escritorio\test3.csv")

# Mostrar los primeros registros de cada DataFrame
print("Primeros registros de train_data_1:")
print(train_data_1_csv.head())
print("===================================")
print("Primeros registros de train_data_2:")
print(train_data_2_csv.head())
print("===================================")
print("Primeros registros de test_questions_data:")
print(test_questions_data_csv.head())
print("===================================")
print("Primeros registros de test_data:")
print(test_data_csv.head())
print("===================================")
print("Primeros registros de qrels_data:")
print(qrels_data_csv.head())









