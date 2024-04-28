# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:01:36 2024

@author: migui
"""
modelo= "RED NEURONAL"
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

def load_text(filename):
    """Función para cargar el texto de un archivo."""
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def load_texts_from_folder(folder_path):
    """Función para cargar todos los archivos de texto en una carpeta."""
    full_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            file_text = load_text(file_path)
            full_texts.append(file_text)
    return ' '.join(full_texts)

# Ruta de la carpeta que contiene los archivos de texto
carpeta = "C:/Users/migui/Downloads/miguel/data_train/"

# Cargar todos los textos de la carpeta en un único texto de entrada
text = load_texts_from_folder(carpeta)

# Convertir a minúsculas
text = text.lower()

# Lista de formas del verbo "to be" para predecir
target_words = ['am', 'are', 'is']

# Tokenización y creación de secuencias
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts([text])
sequence = tokenizer.texts_to_sequences([text])[0]

# Crear X e y
X = []
y = []

for i in range(2, len(sequence)):
    if tokenizer.index_word[sequence[i]] in target_words:
        X.append(sequence[i-2:i])  # Tomar las dos últimas palabras antes del verbo
        y.append(sequence[i])

# Asegurar que todas las secuencias en X tengan longitud 2
X = pad_sequences(X, maxlen=2, padding='pre')

# Convertir etiquetas a números
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(target_words)
y = np.array(label_tokenizer.texts_to_sequences([' '.join([tokenizer.index_word[i] for i in y])])).flatten() - 1

# Definir la arquitectura del modelo
if modelo=="RED NEURONAL":
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=101, output_dim=8, input_length=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(len(target_words), activation='softmax')
    ])
else:
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=101, output_dim=8, input_length=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(len(target_words), activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrenar el modelo
model.fit(X, y, epochs=10)

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
    test_seq_padded = pad_sequences(test_seq, maxlen=2)
    prediction = model.predict(test_seq_padded)
    predicted_index = np.argmax(prediction)
    predicted_word = label_tokenizer.index_word[predicted_index + 1]  # +1 porque los índices en Tokenizer empiezan en 1

    print(test_text, predicted_word)
