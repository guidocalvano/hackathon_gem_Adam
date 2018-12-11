from DataImporter import DataImporter
import keras
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard as TensorBoardCallback
from keras.models import Model
from keras.utils import to_categorical
import numpy as np



class SimpleLearning:

    @staticmethod
    def simple_binary_classification(data_set,
                                     epochs=10, batch_size=320):

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

#
# class ValidationLossGuidedLearningRate(keras.callbacks.History):
#
#     def __init__(self, base_learning_rate, lookback_epochs, decay_threshold=0.002,
#                  decay_rate=0.5, loss_type='val_loss'):
#
#         super(ValidationLossGuidedLearningRate, self).__init__()
#
#         self.base_learning_rate = base_learning_rate
#         self.lookback_epochs = int(lookback_epochs)
#         self.decay_threshold = decay_threshold
#         self.decay_rate = decay_rate
#         self.loss_type = loss_type
#
#     def on_epoch_begin(self, epoch, logs=None):
#         relevant_loss_history = self.history[self.loss_type][-self.lookback_epochs:]
#
#         current_learning_rate = K.get_value(self.model.optimizer.lr)
#         if(self.must_decrease_learning_rate(relevant_loss_history)):
#             current_learning_rate = current_learning_rate * self.decay_rate
#
#             K.set_value(self.model.optimizer.lr, current_learning_rate)
#
#         return K.get_value(self.model.optimizer.lr)
#
#     def loss_stability_probability(self, loss_history):
#         if len(loss_history) < self.lookback_epochs:
#             # if loss history too short assume the loss is not stable
#             return 0
#
#         lookback_loss_mean = np.mean(loss_history)
#         lookback_loss_std = np.std(loss_history)
#
#         # The probability of improvement
#         current_loss_z_score = (loss_history[-1] - lookback_loss_mean) / lookback_loss_std
#         p = scipy.stats.norm.cdf(current_loss_z_score)
#         return p
#
#     def must_decrease_learning_rate(self, loss_history):
#         # if loss stability probability fairly great, reduce the learning rate
#         return self.loss_stability_probability(loss_history) > self.stability_probability_threshold