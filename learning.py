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
from keras.callbacks import EarlyStopping
import config
import data_import
import os
import datetime
import numpy as np

def create_model(input_shape, output_count):
    classifier = Sequential()

    # Step 0 downsampling

    # Step 1 = Convolution
    classifier.add(Conv2D(32, 5, padding='same', activation='relu', input_shape=np.array(input_shape)))

    # Step 2 Max Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding second convolutional layer
    classifier.add(Conv2D(32, 5, border_mode='valid', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 3 Flattening
    classifier.add(Flatten())
    # Step 4 Full Connection
    classifier.add(Dense(units=output_count, activation='softmax'))

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier


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
                        batch_size=32,
                        callbacks=[early_stopping])

    accuracy = get_accuracy_from_results(results)

    predictions = model.predict(ds["test"][0])

    # output_file_path = config.PREDICTION_PATH + 'accuracy_' + str(accuracy) + '_date_time_' + str(datetime.datetime.now()) + '_git_commit_' + git_commit + '.csv'

    # presentation.save_predictions(config.TEST_DATA_FILE_PATH, output_file_path, predictions)

def get_accuracy_from_results(results):
    accuracy = results.history["val_acc"][-1]

    return accuracy

run()