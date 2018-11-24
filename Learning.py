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

        training_metrics = model.evaluate(data_set["training"][0], data_set["training"][1].label_clean_int)
        training_prediction = np.round(model.predict(data_set["training"][0], batch_size=batch_size))

        validation_metrics = model.evaluate(data_set["validation"][0], data_set["validation"][1].label_clean_int)
        validation_prediction = np.round(model.predict(data_set["validation"][0], batch_size=batch_size))

        test_metrics = model.evaluate(data_set["test"][0], data_set["test"][1].label_clean_int)
        test_prediction = np.round(model.predict(data_set["test"][0], batch_size=batch_size))

        return {
            "stats": {
                "training": {
                    "history": results.history,
                    "metrics": dict(zip(model.metrics_names, training_metrics)),
                    "correct": list(data_set["training"][1].label_clean_int),
                    "predicted": training_prediction.reshape(-1).tolist()
                },
                "validation": {
                    "metrics": dict(zip(model.metrics_names, validation_metrics)),
                    "correct": list(data_set["validation"][1].label_clean_int),
                    "predicted": validation_prediction.reshape(-1).tolist()
                },
                "test": {
                    "metrics": dict(zip(model.metrics_names, test_metrics)),
                    "correct": list(data_set["test"][1].label_clean_int),
                    "predicted": test_prediction.reshape(-1).tolist()
                },
                "meta": {
                    "type": "binary",
                    "labels": ['dirty', 'clean']
                }
            },
            "model": model
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

        training_metrics = model.evaluate(data_set["training"][0], to_categorical(data_set["training"][1].label_type_int))
        training_prediction = np.argmax(model.predict(data_set["training"][0], batch_size=batch_size), -1)

        validation_metrics = model.evaluate(data_set["validation"][0], to_categorical(data_set["validation"][1].label_type_int))
        validation_prediction = np.argmax(model.predict(data_set["validation"][0], batch_size=batch_size), -1)

        test_metrics = model.evaluate(data_set["test"][0], to_categorical(data_set["test"][1].label_type_int))
        test_prediction = np.argmax(model.predict(data_set["test"][0], batch_size=batch_size), -1)

        int_label = list(data_set["training"][1].label_type_int)
        label = list(data_set["meta"]["training"].label_type_str)
        int_label_hash = dict(zip(int_label, label))
        labels = list(map(lambda i: int_label_hash[i], np.arange(np.max(int_label) + 1)))

        return {
            "stats": {
                "training": {
                    "history": results.history,
                    "metrics": dict(zip(model.metrics_names, training_metrics)),
                    "correct": list(data_set["training"][1].label_type_int),
                    "predicted": training_prediction.reshape(-1).tolist()
                },
                "validation": {
                    "metrics": dict(zip(model.metrics_names, validation_metrics)),
                    "correct": list(data_set["validation"][1].label_type_int),
                    "predicted": validation_prediction.reshape(-1).tolist()
                },
                "test": {
                    "metrics": dict(zip(model.metrics_names, test_metrics)),
                    "correct": list(data_set["test"][1].label_type_int),
                    "predicted": test_prediction.reshape(-1).tolist()
                },
                "meta": {
                    "type": "categorical",
                    "labels": labels
                }
            },
            "model": model
        }

    @staticmethod
    def simple_crow_score_regression(data_set, epochs=10, batch_size=320):
        inputs = Input(shape=data_set["training"][0].shape[1:])
        layer_1 = Conv2D(32, 5, padding='valid', activation='relu', kernel_initializer='glorot_normal')(inputs)
        layer_2 = MaxPooling2D(pool_size=(2, 2))(layer_1)
        layer_3 = Conv2D(32, 5, padding='valid', activation='relu', kernel_initializer='glorot_normal')(layer_2)
        layer_4 = MaxPooling2D(pool_size=(2, 2))(layer_3)
        layer_5 = Flatten()(layer_4)
        outputs = Dense(units=1, activation='sigmoid', name='prediction', kernel_initializer='glorot_normal')(layer_5)

        model = Model(inputs=inputs, outputs=[outputs])

        model.compile(optimizer='adam', loss=['mean_squared_error'], metrics=['mse'])

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=0,
                                       verbose=0, mode='auto')

        results = model.fit(data_set["training"][0],
                            data_set["training"][1].label_crow_score_int / 3,
                            validation_data=(data_set["validation"][0], data_set["validation"][1].label_crow_score_int / 3),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping])

        # validation_accuracy = results.history["val_mean_squared_error"][-1]

        training_metrics = model.evaluate(data_set["training"][0], data_set["training"][1].label_crow_score_int / 3)
        training_prediction = model.predict(data_set["training"][0], batch_size=batch_size) * 3

        validation_metrics = model.evaluate(data_set["validation"][0], data_set["validation"][1].label_crow_score_int / 3)
        validation_prediction = model.predict(data_set["validation"][0], batch_size=batch_size) * 3

        test_metrics = model.evaluate(data_set["test"][0], data_set["test"][1].label_crow_score_int / 3)
        test_prediction = model.predict(data_set["test"][0], batch_size=batch_size) * 3

        return {
            "stats": {
                "training": {
                    "history": results.history,
                    "metrics": dict(zip(model.metrics_names, training_metrics)),
                    "correct": list(data_set["training"][1].label_crow_score_int),
                    "predicted": training_prediction.reshape(-1).tolist()
                },
                "validation": {
                    "metrics": dict(zip(model.metrics_names, validation_metrics)),
                    "correct": list(data_set["validation"][1].label_crow_score_int),
                    "predicted": validation_prediction.reshape(-1).tolist()
                },
                "test": {
                    "metrics": dict(zip(model.metrics_names, test_metrics)),
                    "correct": list(data_set["test"][1].label_crow_score_int),
                    "predicted": test_prediction.reshape(-1).tolist()
                },
                "meta": {
                    "type": "regression",
                    "labels": None
                }
            },
            "model": model
        }
