# -*- coding: utf-8 -*-

#################################################################################
# training the model
#################################################################################
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
import dill


import config
import data_import
import os
import datetime
import numpy as np
import logging


def add_standard_conv_layer(model):
    model.add(Conv2D(32, 5, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    return model

def create_model(input_shape, output_count, layer_count=2):
    model = Sequential()

    model.add(Conv2D(32, 5, padding='valid', activation='relu', input_shape=np.array(input_shape)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(layer_count - 1):
        add_standard_conv_layer(model)

    # # Step 1 = Convolution
    # model.add(Conv2D(32, 5, padding='valid', activation='relu', input_shape=np.array(input_shape)))
    #
    # # Step 2 Max Pooling
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # Adding second convolutional layer
    # model.add(Conv2D(32, 5, border_mode='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # Adding third convolutional layer
    # model.add(Conv2D(32, 5, border_mode='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 3 Flattening
    model.add(Flatten())
    # Step 4 Full Connection
    model.add(Dense(units=output_count, activation='softmax'))

    # Compiling the CNN
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_2_layer_model_with_input_reconstruction(input_shape, output_count):
    inputs = Input(shape=input_shape)
    layer_1 = Conv2D(32, 5, padding='valid', activation='relu')(inputs)
    layer_2 = MaxPooling2D(pool_size=(2, 2))(layer_1)
    layer_3 = Conv2D(32, 5, border_mode='valid', activation='relu')(layer_2)
    layer_4 = MaxPooling2D(pool_size=(2, 2))(layer_3)
    layer_5 = Flatten()(layer_4)
    outputs = Dense(units=output_count, activation='softmax', name='prediction')(layer_5)

    reconstruction_1 = Dense(32, activation='relu')(layer_5)
    reconstruction_2 = Dense(np.product(input_shape), activation='sigmoid')(reconstruction_1)
    final_reconstruction = Reshape(input_shape)(reconstruction_2)

    model = Model(inputs=inputs, outputs=[outputs, final_reconstruction])

    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])

    return model

def create_5_layer_model_with_input_reconstruction(input_shape, output_count):
    inputs = Input(shape=input_shape)
    layer = Conv2D(32, 5, padding='valid', activation='relu')(inputs)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(32, 5, border_mode='valid', activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(32, 5, border_mode='valid', activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(32, 5, border_mode='valid', activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(32, 5, border_mode='valid', activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Flatten()(layer)
    outputs = Dense(units=output_count, activation='softmax', name='prediction')(layer)

    reconstruction_1 = Dense(32, activation='relu')(layer)
    reconstruction_2 = Dense(32, activation='relu')(reconstruction_1)
    reconstruction_3 = Dense(np.product(input_shape), activation='sigmoid')(reconstruction_2)

    final_reconstruction = Reshape(input_shape)(reconstruction_3)

    model = Model(inputs=inputs, outputs=[outputs, final_reconstruction])

    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])

    return model

def run_n_layers(ds, layer_count):

    model = create_model(ds["training"][0].shape[1:], config.CLASS_COUNT)

    early_stopping = EarlyStopping(monitor='val_acc',
                                  patience=0,
                                  verbose=0, mode='auto')

    results = model.fit(ds["training"][0],
                        ds["training"][1],
                        validation_data=ds["validation"],
                        epochs=1000,
                        batch_size=1000,
                        callbacks=[early_stopping])

    accuracy = get_accuracy_from_results(results)
    one_hot_predictions = model.predict(ds["validation"][0])
    predictions = np.argmax(one_hot_predictions, axis=1)

    cm = confusion_matrix(np.argmax(ds["validation"][1], axis=1), predictions)

    results = {
        "validation_confusion": cm,
        "validation_accuracy": accuracy,
        "history": results.history
    }

    return results

def run_2_layers_with_reconstruction(ds):
    model = create_2_layer_model_with_input_reconstruction(ds["training"][0].shape[1:], config.CLASS_COUNT)

    early_stopping = EarlyStopping(monitor='val_prediction_acc',
                                  patience=0,
                                  verbose=0, mode='auto')

    results = model.fit(ds["training"][0],
                        [ds["training"][1], ds["training"][0]],
                        validation_data=(ds["validation"][0], [ds["validation"][1], ds["validation"][0]]),
                        epochs=1000,
                        batch_size=1000,
                        callbacks=[early_stopping])

    accuracy = results.history["val_prediction_acc"][-1]

    one_hot_predictions = model.predict(ds["validation"][0])[0]
    predictions = np.argmax(one_hot_predictions, axis=1)

    cm = confusion_matrix(np.argmax(ds["validation"][1], axis=1), predictions)

    results = {
        "validation_confusion": cm,
        "validation_accuracy": accuracy,
        "history": results.history
    }

    return results

def run_5_layers_with_reconstruction(ds):
    model = create_2_layer_model_with_input_reconstruction(ds["training"][0].shape[1:], config.CLASS_COUNT)

    early_stopping = EarlyStopping(monitor='val_prediction_acc',
                                  patience=0,
                                  verbose=0, mode='auto')

    results = model.fit(ds["training"][0],
                        [ds["training"][1], ds["training"][0]],
                        validation_data=(ds["validation"][0], [ds["validation"][1], ds["validation"][0]]),
                        epochs=1000,
                        batch_size=1000,
                        callbacks=[early_stopping])

    accuracy = results.history["val_prediction_acc"][-1]

    one_hot_predictions = model.predict(ds["validation"][0])[0]
    predictions = np.argmax(one_hot_predictions, axis=1)

    cm = confusion_matrix(np.argmax(ds["validation"][1], axis=1), predictions)

    results = {
        "validation_confusion": cm,
        "validation_accuracy": accuracy,
        "history": results.history
    }

    return results


def run_experiments():
    ds = data_import.default_cache_load()

    try:
        two_layer_reconstruction = run_2_layers_with_reconstruction(ds)
    except Exception as e:
        two_layer_reconstruction = str(e)
        logging.exception(e)

    try:
        five_layer_reconstruction = run_5_layers_with_reconstruction(ds)
    except Exception as e:
        five_layer_reconstruction = str(e)
        logging.exception(e)

    try:
        two_layer = run_n_layers(ds, 2)
    except Exception as e:
        two_layer = str(e)
        logging.exception(e)

    try:
        three_layer = run_n_layers(ds, 3)
    except Exception as e:
        three_layer = str(e)
        logging.exception(e)

    try:
        four_layer = run_n_layers(ds, 4)
    except Exception as e:
        four_layer = str(e)
        logging.exception(e)

    try:
        five_layer = run_n_layers(ds, 5)
    except Exception as e:
        five_layer = str(e)
        logging.exception(e)

    results = {
        "two_layer": two_layer,
        "three_layer": three_layer,
        "four_layer": four_layer,
        "five_layer": five_layer,
        "two_layer_reconstruction": two_layer_reconstruction,
        "five_layer_reconstruction": five_layer_reconstruction
    }

    dill.dump(results, open(config.RESULTS_FILEPATH, 'wb'))


def run():
    ds = data_import.default_cache_load()

    model = create_model(ds["training"][0].shape[1:], config.CLASS_COUNT)

    early_stopping = EarlyStopping(monitor='val_acc',
                                  patience=0,
                                  verbose=0, mode='auto')

    results = model.fit(ds["training"][0],
                        ds["training"][1],
                        validation_data=ds["validation"],
                        epochs=10,
                        batch_size=320,
                        callbacks=[early_stopping])

    accuracy = get_accuracy_from_results(results)

    evaluation = model.evaluate(ds["test"][0], ds["test"][1])

    predictions = model.predict(ds["test"][0])

    # output_file_path = config.PREDICTION_PATH + 'accuracy_' + str(accuracy) + '_date_time_' + str(datetime.datetime.now()) + '_git_commit_' + git_commit + '.csv'

    # presentation.save_predictions(config.TEST_DATA_FILE_PATH, output_file_path, predictions)


def run2():
    ds = data_import.default_cache_load()

    model = create_2_layer_model_with_input_reconstruction(ds["training"][0].shape[1:], config.CLASS_COUNT)

    early_stopping = EarlyStopping(monitor='val_dense_1_acc',
                                  patience=0,
                                  verbose=0, mode='auto')

    results = model.fit(ds["training"][0],
                        [ds["training"][1], ds["training"][0]],
                        validation_data=(ds["validation"][0], [ds["validation"][1], ds["validation"][0]]),
                        epochs=10,
                        batch_size=32,
                        callbacks=[early_stopping])

    accuracy = results.history["dense_1_acc"][-1]

    evaluation = model.evaluate(ds["test"][0], [ds["test"][1], ds["test"][0]])

    predictions = model.predict(ds["test"][0])

    # output_file_path = config.PREDICTION_PATH + 'accuracy_' + str(accuracy) + '_date_time_' + str(datetime.datetime.now()) + '_git_commit_' + git_commit + '.csv'

    # presentation.save_predictions(config.TEST_DATA_FILE_PATH, output_file_path, predictions)




def get_accuracy_from_results(results):
    accuracy = results.history["val_acc"][-1]

    return accuracy

run_experiments()
