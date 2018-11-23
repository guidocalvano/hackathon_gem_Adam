from Configuration import Configuration
from DataImporter import DataImporter

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Reshape

from keras.callbacks import EarlyStopping
from keras.models import Model
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import numpy as np


class Learning:

    @staticmethod
    def simple_binary_classification(data_set, epochs=10, batch_size=320):

        inputs = Input(shape=data_set["training"][0].shape[1:])
        layer_1 = Conv2D(32, 5, padding='valid', activation='relu', kernel_initializer='glorot_normal')(inputs)
        layer_2 = MaxPooling2D(pool_size=(2, 2))(layer_1)
        layer_3 = Conv2D(32, 5, padding='valid', activation='relu', kernel_initializer='glorot_normal')(layer_2)
        layer_4 = MaxPooling2D(pool_size=(2, 2))(layer_3)
        layer_5 = Flatten()(layer_4)
        outputs = Dense(units=1, activation='sigmoid', name='prediction', kernel_initializer='glorot_normal')(layer_5)

        model = Model(inputs=inputs, outputs=[outputs])

        model.compile(optimizer='adam', loss=['binary_crossentropy'], metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_acc',
                                       patience=0,
                                       verbose=0, mode='auto')

        results = model.fit(data_set["training"][0],
                            data_set["training"][1].label_clean_int,
                            validation_data=(data_set["validation"][0], data_set["validation"][1].label_clean_int),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping])

        validation_accuracy = results.history["val_acc"][-1]

        evaluation = model.evaluate(data_set["test"][0], data_set["test"][1].label_clean_int)

        return {
            "validation": validation_accuracy,
            "test": evaluation[1]
        }

    @staticmethod
    def simple_categorical_classification(data_set, epochs = 10, batch_size = 320):

        inputs = Input(shape=data_set["training"][0].shape[1:])
        layer_1 = Conv2D(32, 5, padding='valid', activation='relu', kernel_initializer='glorot_normal')(inputs)
        layer_2 = MaxPooling2D(pool_size=(2, 2))(layer_1)
        layer_3 = Conv2D(32, 5, padding='valid', activation='relu', kernel_initializer='glorot_normal')(layer_2)
        layer_4 = MaxPooling2D(pool_size=(2, 2))(layer_3)
        layer_5 = Flatten()(layer_4)
        outputs = Dense(units=(np.max(data_set["training"][1].label_type_int) + 1), activation='softmax', name='prediction', kernel_initializer='glorot_normal')(layer_5)

        model = Model(inputs=inputs, outputs=[outputs])

        model.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_acc',
        patience = 0,
        verbose = 0, mode = 'auto')

        results = model.fit(data_set["training"][0],
            to_categorical(data_set["training"][1].label_type_int),
            validation_data = (data_set["validation"][0], to_categorical(data_set["validation"][1].label_type_int)),
            epochs = epochs,
            batch_size = batch_size,
            callbacks = [early_stopping])

        validation_accuracy = results.history["val_acc"][-1]

        evaluation = model.evaluate(data_set["test"][0], to_categorical(data_set["test"][1].label_type_int))

        return {
            "validation": validation_accuracy,
            "test": evaluation[1]
        }

    @staticmethod
    def simple_crow_score_regression(data_set, epochs=10, batch_size=320):
        inputs = Input(shape=data_set["training"][0].shape[1:])
        layer_1 = Conv2D(32, 5, padding='valid', activation='relu', kernel_initializer='glorot_normal')(inputs)
        layer_2 = MaxPooling2D(pool_size=(2, 2))(layer_1)
        layer_3 = Conv2D(32, 5, padding='valid', activation='relu', kernel_initializer='glorot_normal')(layer_2)
        layer_4 = MaxPooling2D(pool_size=(2, 2))(layer_3)
        layer_5 = Flatten()(layer_4)
        outputs = Dense(units=1, activation='relu', name='prediction', kernel_initializer='glorot_normal')(layer_5)

        model = Model(inputs=inputs, outputs=[outputs])

        model.compile(optimizer='adam', loss=['mean_squared_error'], metrics=['mse'])

        early_stopping = EarlyStopping(monitor='val_acc',
                                       patience=0,
                                       verbose=0, mode='auto')

        results = model.fit(data_set["training"][0],
                            data_set["training"][1].label_crow_score_int,
                            validation_data=(data_set["validation"][0], data_set["validation"][1].label_crow_score_int),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping])

        validation_accuracy = results.history["val_mean_squared_error"][-1]

        evaluation = model.evaluate(data_set["test"][0], data_set["test"][1].label_crow_score_int)

        return {
            "validation": validation_accuracy,
            "test": evaluation[1]
        }
