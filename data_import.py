import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sys

import config

def load_image_set(image_description_file_path, image_path):
    # not finished
    image_description = pd.read_csv(image_description_file_path)

    image_file_paths = []

    images = []

    return image

def load_image_tensor(image_file_paths):

    image_array_list = []

    for i in range(image_file_paths.shape[0]):
        next_image = load_img(image_file_paths[i])
        image_array = np.array(next_image)

        standardized_image_array = standardize_image_size(image_array)

        downsampled_image_array = resize_images(standardized_image_array)

        image_array_list.append(downsampled_image_array)

    image_tensor = np.concatenate(image_array_list)

    return image_tensor

def standardize_image_size(image_array):
    (width, height) = image_array.shape
    ratio = height / width

    target_ratio = config.COMMON_RATIO

    target_height = width * target_ratio
    target_width = height / target_ratio

    if target_height < height:
        excess_height = height - target_height
        slice_start = int(excess_height / 2)
        slice_end = slice_start + height

        cropped_image = image_array[:, slice_start:slice_end]

        return cropped_image

    if target_width < width:
        excess_width = width - target_width
        slice_start = int(excess_width / 2)
        slice_end = slice_start + width

        cropped_image = image_array[slice_start:slice_end, :]

        return cropped_image

    return image_array


def resize_images(image_arrays, target_size):
    sess = tf.Sesssion()

    image_tf = tf.placeholder()

    resized_image_tf = tf.image.resize_images(image_tf, size=target_size)

    resized_images = []

    with sess.as_default():
        for i in len(image_arrays):
            original = image_arrays[i]
            resized = resized_image_tf.eval({image_tf: original})

            resized_images.append(resized)

    return resized_images


def normalize_data(data):
    mean = np.mean(data)
    centered_data = data - mean
    standard_deviation = np.std(centered_data) + sys.float_info.epsilon

    standardized_data = centered_data / standard_deviation

    return standardized_data, mean, standard_deviation

def normalize_test_images(data, mean, standard_deviation):
    centered_data = data - mean

    standardized_data = centered_data / standard_deviation

    return standardized_data

def import_all_data():


    return
