import tensorflow as tf
import numpy as np
import functions

# Carico il modello gia addestrato
model = tf.keras.models.load_model("model_save")
# Nuovo layer per il modello che serve a fare delle previsioni
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

def indovina():
    path_immagine = input("Inserisci il path dell'immagine:\n")
    test_image = functions.img_to_array(path_immagine)
    labels = ["una banana", "un cocomero", "un kiwi", "un limone", "un mandarino", "una mela", "una pera"]

    # Faccio una previsione con l'immagine caricata
    prediction = probability_model.predict(test_image)
    if np.max(prediction[0]) < 0.25:
        print("Non sono in grado di identificare questo frutto")
    else:
        print(f"L'immagine che hai caricato è {labels[np.argmax(prediction[0])]}\n")

if __name__ == "__main__":
    while True:
        indovina()

