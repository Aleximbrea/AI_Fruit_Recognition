import tensorflow as tf
import numpy as np
import functions
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import kerastuner as kt
import model

if __name__ == "__main__":
    # Creo degli array per ogni tipo di frutta
    banane = functions.imgs_to_array("c:\\users\\alexi\\desktop\\frutta\\banane_converted")
    cocomeri = functions.imgs_to_array("c:\\users\\alexi\\desktop\\frutta\\cocomeri_converted")
    kiwi = functions.imgs_to_array("c:\\users\\alexi\\desktop\\frutta\\kiwi_converted")
    limoni = functions.imgs_to_array("c:\\users\\alexi\\desktop\\frutta\\limoni_converted")
    mandarini = functions.imgs_to_array("c:\\users\\alexi\\desktop\\frutta\\mandarini_converted")
    mele = functions.imgs_to_array("c:\\users\\alexi\\desktop\\frutta\\mele_converted")
    pere = functions.imgs_to_array("c:\\users\\alexi\\desktop\\frutta\\pere_converted")

    labels = ["banane", "cocomeri", "kiwi", "limoni", "mandarini", "mele", "pere"]

    # Creo un unico array con tutti i frutti
    frutta = [banane, cocomeri, kiwi, limoni, mandarini, mele, pere]

    # Questo array non è nella forma corretta quindi dobbiamo ridimensionarlo
    temp = []
    for i in frutta:
        for j in i:
            temp.append(j)
    train_images = np.array(temp)

    # Riempio l'array con dei numeri che indicano il tipo di frutta
    train_labels = []
    k = 0
    for image in train_images:
        train_labels.append(int(k/60))
        k = k + 1
    train_labels = np.array(train_labels)

    # train_images= train_images.reshape(-1,64,64, 3)
    
    # Definisco il callback EarlyStopping monitorando la perdita per non andare in overfitting
    early_stopping_callback = EarlyStopping(monitor='loss', patience=5)

    
    # Inizia l'addestramento
    #model = functions.create_model()
    #model.fit(train_images, train_labels, epochs=50, callbacks=[early_stopping_callback])
    #model.save("model_save")

    # Addestramento con ricerca di iperparametri ideali
    hypermodel = model.MyHyperModel2(input_shape=(64, 64, 3), num_classes=7)
    tuner = kt.Hyperband(hypermodel,
                     objective='val_accuracy',
                     max_epochs=100,
                     factor=3,
                     project_name = "tuner")
    #tuner.search(train_images, train_labels, epochs=50, validation_split=0.2, callbacks=[early_stopping_callback])
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    # Addestro il modello con gli iperparametri migliori
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_images, train_labels, epochs=50, validation_split=0.2)
    # Trovo il numero di epoch con la val_accuracy più alta
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    #print('Best epoch: %d' % (best_epoch,))
    # Ri addestro il modello utilizzando il numero di epoch migliore
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(train_images, train_labels, epochs=best_epoch, validation_split=0.2)
    hypermodel.save("model_save")


    


    
    
