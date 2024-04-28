# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:01:36 2024

@author: migui
"""
modelo= "RED NEURONAL"
modelo= "RNN"
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

def load_text(filename):
    """Función para cargar el texto de un archivo."""
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def load_texts_from_folder(folder_path,archivo=None):
    if archivo:
        file_text = load_text(folder_path+archivo)
        return file_text
    else:
        """Función para cargar todos los archivos de texto en una carpeta."""
        full_texts = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                file_text = load_text(file_path)
                full_texts.append(file_text)
        return ' '.join(full_texts)


def process_text(text):
    import string
    import re
    # Caracteres de puntuación
    punctuation = string.punctuation
    # Eliminar stop words
    words = text.split()
    #filtered_text = ' '.join([word for word in words if word.lower() not in stop_words])
    
    # Eliminar puntuación y números
    cleaned_text = re.sub(r'[{}0-9]'.format(punctuation), '', text)
    
    return cleaned_text


# Ruta de la carpeta que contiene los archivos de texto
#carpeta = "D:/miguel/data_train/"
carpeta = "data_train/"


# Cargar todos los textos de la carpeta en un único texto de entrada
text = load_texts_from_folder(carpeta,"enchiridion.txt")
text = load_texts_from_folder(carpeta)
# Convertir a minúsculas
text = text.lower()

text = process_text(text)
# Lista de formas del verbo "to be" para predecir
target_words = ['am', 'are', 'is']
target_words = ["is", 'are',"am","were","was","have","has","had"]



# Filtrar el conteo para incluir solo las palabras objetivo

# Tokenizar el texto y contar las palabras
from collections import Counter
words = text.split()  # Convertimos a minúscula y separamos por espacios
word_count = Counter(words)
target_counts = {word: word_count[word] for word in target_words}


import matplotlib.pyplot as plt

# Etiquetas para el gráfico, que son nuestras palabras objetivo
labels = target_counts.keys()

# Valores, que son las frecuencias de las palabras objetivo
sizes = target_counts.values()

# Verificar que el tamaño total no es cero para evitar errores de división por cero en gráficos vacíos
if sum(sizes) > 0:
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Esto asegura que el gráfico de pastel sea un círculo
    plt.title('Frecuencia de aparición de las palabras ')
    plt.show()
else:
    print("Las palabras objetivo no aparecen en el texto.")






# Tokenización y creación de secuencias
numero_palabras=200
tokenizer = Tokenizer(num_words=numero_palabras, oov_token='<OOV>')
tokenizer.fit_on_texts([text])

sequence = tokenizer.texts_to_sequences([text])[0]





word_index = tokenizer.word_index
word_counts = tokenizer.word_counts
# Ordenar las palabras por frecuencia y seleccionar las 20 más frecuentes
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

top_100_words = sorted_word_counts[:numero_palabras]



import matplotlib.pyplot as plt

# Separar las palabras y sus frecuencias
words, frequencies = zip(*top_100_words)

# Crear un gráfico de barras
plt.figure(figsize=(10, 8))
plt.bar(words, frequencies, color='blue')
plt.xlabel('Palabras')
plt.ylabel('Frecuencia')
plt.title('Top 20 Tokens Más Utilizados')
plt.xticks(rotation=45, ha='right')
plt.show()


# Lista de palabras a verificar
words_to_check = target_words
words_to_check = ["i", "you","he","was","were"]

# Extraer solo las palabras de las tuplas de las top 20
top_words = [word for word, count in top_100_words]

# Verificar si cada palabra en words_to_check está en top_words
check_results = {word: word in top_words for word in words_to_check}

# Imprimir los resultados
for word, is_top in check_results.items():
    print(f"La palabra '{word}' {'está' if is_top else 'no está'} entre las 100 más utilizadas.")










# Crear X e y
X = []
y = []
diccionario={}
# Lista de ejemplo con elementos duplicados
mi_lista = sequence

# Convertir la lista a un conjunto para eliminar duplicados
elementos_unicos = set(mi_lista)

# Si necesitas que el resultado sea una lista
elementos_unicos_lista = list(elementos_unicos)

#print(elementos_unicos_lista)

for h in range(len(sequence)):
    aaaaaaaaaa=tokenizer.index_word[sequence[h]]
    diccionario[aaaaaaaaaa]=sequence[h]
"""   
for i in range(2, len(sequence)):
    if tokenizer.index_word[sequence[i]] in target_words:
        X.append(sequence[i-2:i])  # Tomar las dos últimas palabras antes del verbo
        y.append(sequence[i])
"""

"""
for i in range(2, len(sequence) - 2):  # Cambiar para evitar el desbordamiento de índice
    if tokenizer.index_word[sequence[i]] in target_words:
        # Tomar las dos últimas palabras antes y las dos primeras palabras después del verbo
        context_before = sequence[i-2:i]  # Las dos palabras antes del verbo
        context_after = sequence[i+1:i+3]  # Las dos palabras después del verbo
        
        # Combina ambos contextos en uno solo
        context = context_before + context_after
        
        # Añadir el contexto combinado y el verbo objetivo a las listas X e y
        X.append(context)
        y.append(sequence[i])

"""


def extract_context(sequence, target_words, num_words_before, num_words_after):
    X = []
    y = []
    # Asegúrate de que el rango en el bucle no cause desbordamiento de índice
    for i in range(num_words_before, len(sequence) - num_words_after):
        if tokenizer.index_word[sequence[i]] in target_words:
            # Tomar las palabras antes del verbo objetivo
            context_before = sequence[max(0, i-num_words_before):i]  # Ajuste para no salir del límite
            
            # Tomar las palabras después del verbo objetivo
            context_after = sequence[i+1:min(len(sequence), i+1+num_words_after)]  # Ajuste para no salir del límite
            
            # Combina ambos contextos en uno solo
            context = context_before + context_after
            
            # Añadir el contexto combinado y el verbo objetivo a las listas X e y
            X.append(context)
            y.append(sequence[i])
    
    return X, y

# Ejemplo de uso
#sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#target_words = {tokenizer.index_word[4], tokenizer.index_word[7]}  # Suponiendo que los números representan palabras
num_words_before = 8  # Número de palabras antes de la palabra objetivo
num_words_after = 8  # Número de palabras después de la palabra objetivo

X, y = extract_context(sequence, target_words, num_words_before, num_words_after)
#plitting the dataset into train and test
contexts_train, contexts_test = train_test_split(contexts, test_size=0.2, random_state=42)

print("Training Data:", contexts_train)
print("Test Data:", contexts_test)
X_ANT=X







# Asegurar que todas las secuencias en X tengan longitud 2


maxlen= num_words_after+num_words_before
X = pad_sequences(X, maxlen=maxlen, padding='pre')

# Convertir etiquetas a números
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(target_words)
y = np.array(label_tokenizer.texts_to_sequences([' '.join([tokenizer.index_word[i] for i in y])])).flatten() - 1

# Definir la arquitectura del modelo
if modelo=="RED NEURONAL":
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=numero_palabras+1, output_dim=80, input_length=maxlen),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='tanh'),
    #tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(len(target_words), activation='softmax')
    ])
else:
    """
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=101, output_dim=8, input_length=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(len(target_words), activation='softmax')
    ])
    """
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=8, input_length=maxlen),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(len(target_words), activation='softmax')
    ])


# Compilar el 
"""
def find_preceding_words(text, word_list):
    import re
    from collections import Counter

    # Tokenizar el texto conservando sólo palabras
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Diccionario para mantener el conteo de las palabras que preceden a las palabras objetivo
    preceding_words = Counter()
    
    # Buscar cada palabra en la lista y registrar la palabra que la precede
    for i in range(1, len(words)):
        if words[i] in word_list:
            preceding_words[words[i-1]] += 1
    
    # Retornar las palabras más frecuentes que preceden a las palabras en la lista
    return preceding_words.most_common()

# Ejemplo de uso
text = text
word_list = target_words

# Llamar a la función y imprimir el resultado
result = find_preceding_words(text, word_list)
print(result)

"""
def find_preceding_words(text, word_list):
    import re
    from collections import defaultdict, Counter

    # Tokenizar el texto conservando sólo palabras
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Diccionario para mantener contadores separados para cada palabra objetivo
    preceding_words = defaultdict(Counter)
    
    # Buscar cada palabra en la lista y registrar la palabra que la precede
    for i in range(1, len(words)):
        if words[i] in word_list:
            preceding_words[words[i]][words[i-1]] += 1
    
    # Convertir cada contador en una lista de las palabras más comunes y sus conteos
    most_common_preceding = {word: counter.most_common() for word, counter in preceding_words.items()}
    return most_common_preceding

# Ejemplo de uso
text = text
word_list = target_words

# Llamar a la función y imprimir el resultado
result = find_preceding_words(text, word_list)
#print(result)



results = find_preceding_words(text, word_list)
"""
# Prepar""ar un gráfico de pastel para cada palabra en la lista de palabras
for target_word in word_list:
    data = results[target_word]
    labels = [item[0] for item in data]
    sizes = [item[1] for item in data]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'Palabras que más frecuentemente preceden a "{target_word}"')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
"""

# Preparar un gráfico de pastel para cada palabra en la lista de palabras
for target_word in word_list:
    data = results[target_word]
    if len(data) > 12:
        top_data = data[:12]
        others_count = sum([item[1] for item in data[12:]])
        top_data.append(('Otros', others_count))
    else:
        top_data = data
    
    labels = [item[0] for item in top_data]
    sizes = [item[1] for item in top_data]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'Palabras que más frecuentemente preceden a "{target_word}"')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

if True:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Entrenar el modelo
    model.fit(X, y, epochs=25)
    if True:
        # Probar con un fragmento nuevo del texto
        test_text = "you are"
        test_texts = [
            "I think you are",
            "They were going to",
            "She always is",
            "We are not sure if",
        "It is believed that",
        "Do you think I am"
        ]
    
        for test_text in test_texts:

            test_seq = tokenizer.texts_to_sequences([test_text.split()[-2:]])  # Tomar las dos últimas palabras
            test_seq_padded = pad_sequences(test_seq, maxlen=maxlen)
            prediction = model.predict(test_seq_padded)
            predicted_index = np.argmax(prediction)
            predicted_word = label_tokenizer.index_word[predicted_index + 1]  # +1 porque los índices en Tokenizer empiezan en 1

            print(test_text, predicted_word)
