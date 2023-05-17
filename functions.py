import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



def imgs_to_array(path):
    data = []
    # Corro con un loop tutti i file nella cartella
    for filename in os.scandir(path):
        try:
            # Apro l'immagine
            img = Image.open(filename.path)
            img = img.convert('RGB')
            # Converto l'immagine in un array di pixel
            temp = np.asarray(img)
            # Aggiungo l'immagine a un array con tutte le altre immagini
            data.append(temp)
        except:
            print("Error")
    # Converto l'array in un Numpy array
    data = np.array(data)
    # Ritorno l'array
    return data

def img_to_array(path):
    # Funzione per rimpicciolire l'immagine a 50 per 50 pixel
    larghezza, altezza = 64, 64
    try:
        img = Image.open(path)
        img = img.resize((larghezza, altezza))
        # Converti l'immagine in formato RGB
        img = img.convert('RGB')
    except:
        print("Errore!")
    data = []
    # Converto l'array in un Numpy array
    temp = np.asarray(img)
    data.append(temp)
    data = np.array(data)
    # Ritorno l'array
    return data



if __name__ == "__main__":
    img = img_to_array("C:\\Users\\alexi\\Desktop\\prova\\banana_prova.png")
    plt.figure()
    plt.imshow(img)
    plt.show()
    print(img.shape)
