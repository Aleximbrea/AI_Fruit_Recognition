import tensorflow as tf
from tensorflow import keras
from keras import layers
from kerastuner import HyperModel

class MyHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)
        ])
        
        model = keras.Sequential()
        model.add(data_augmentation)
        model.add(layers.Conv2D(filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=32),
                                kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
                                activation='relu',
                                input_shape=self.input_shape))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
        model.add(layers.Conv2D(filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=32),
                                kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
                                activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
        model.add(layers.Conv2D(filters=hp.Int('conv_3_filter', min_value=32, max_value=64, step=32),
                                kernel_size=hp.Choice('conv_3_kernel', values=[3, 5]),
                                activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(units=hp.Int('dense_1_units', min_value=32, max_value=128, step=32),
                               activation='relu'))
        model.add(layers.Dense(units=self.num_classes, activation='softmax'))

        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

class MyHyperModel2(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)
        ])

        model = keras.Sequential()
        model.add(data_augmentation)
        model.add(layers.Conv2D(filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=32),
                                kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
                                activation='relu',
                                input_shape=self.input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=32),
                                kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
                                activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=hp.Int('conv_3_filter', min_value=32, max_value=64, step=32),
                                kernel_size=hp.Choice('conv_3_kernel', values=[3, 5]),
                                activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))

        model.add(layers.Dense(units=hp.Int('dense_1_units', min_value=32, max_value=256, step=32),
                               activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))

        model.add(layers.Dense(units=hp.Int('dense_2_units', min_value=32, max_value=128, step=32),
                               activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.Dense(units=self.num_classes, activation='softmax'))

        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
