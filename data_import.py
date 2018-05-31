import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sys
import config

def load_data_set(image_description_file_path, image_path, target_size):
    # not finished
    image_description = pd.read_csv(image_description_file_path).iloc[:10] # this will reduce your number of examples to 10 .iloc[:10]

    image_file_paths = get_image_file_paths(image_description.foto, image_path)

    images, is_successful = load_image_tensor(image_file_paths, target_size)

    successful_description = image_description.iloc[is_successful]

    return images, None # none will be replaced with labels

def get_image_file_paths(image_series, image_path):
    return image_path + image_series.str.slice(2, -2)

def load_image_tensor(image_file_paths, target_size):

    target_ratio = target_size[1] / target_size[0]

    standardized_ratio_array = []

    is_successful = np.zeros([len(image_file_paths)]).astype(bool)

    for i in range(image_file_paths.shape[0]):
        try:
            next_image = load_img(image_file_paths[i])
            image_array = np.array(next_image)

            standardized_ratio_image = standardize_image_ratio(image_array, target_ratio)

            standardized_ratio_array.append(standardized_ratio_image)
            is_successful[i] = True
        except: # this should match specific errors but if I did the code would be clean and this IS a hackathon
            pass


    target_ratio_height_per_width = target_size[1] / target_size[0]
    downsampled_image_arrays = standardize_resolution(standardized_ratio_array, target_size)

    image_tensor = np.concatenate(downsampled_image_arrays)

    # return array containing ONLY successful images (without holes) and a boolean indexing vector with successful loads
    return image_tensor, is_successful


def standardize_image_ratio(image_array, target_ratio_height_per_width):
    (width, height, _) = image_array.shape

    target_height = width * target_ratio_height_per_width
    target_width = height / target_ratio_height_per_width

    if target_height < height:
        excess_height = height - target_height
        slice_start = int(excess_height / 2)
        slice_end = slice_start + int(target_height)

        cropped_image = image_array[:, slice_start:slice_end]

        return cropped_image

    if target_width < width:
        excess_width = width - target_width
        slice_start = int(excess_width / 2)
        slice_end = slice_start + int(target_width)

        cropped_image = image_array[slice_start:slice_end, :]

        return cropped_image

    return image_array


def standardize_resolution(image_arrays, target_size):
    sess = tf.Session()

    resized_images = []

    with sess.as_default():
        for i in range(len(image_arrays)):

            original = image_arrays[i].astype('float')
            original = np.expand_dims(original, axis=0)
            image_tf = tf.placeholder(tf.float32, shape=original.shape)

            resized_image_tf = tf.image.resize_images(image_tf, size=target_size)

            resized = resized_image_tf.eval({image_tf: original})

            resized_images.append(resized)

    return resized_images


def normalize_data(data): # does not take into account night or day
    mean = np.mean(data)
    centered_data = data - mean
    standard_deviation = np.std(centered_data) + sys.float_info.epsilon

    standardized_data = centered_data / standard_deviation

    return standardized_data, mean, standard_deviation

def normalize_test_images(data, mean, standard_deviation):
    centered_data = data - mean

    standardized_data = centered_data / standard_deviation

    return standardized_data

def split_data_set(full_set):

    examples, labels = full_set

    label_types = np.sort(np.unique(labels))

    split_indices_per_label_type = []

    for label_type in label_types:
        lable_type_indices = np.where(labels == label_type)



def import_all_data():
    image_tensor, labels = load_data_set(config.DATA_DESCRIPTION_FILE, config.IMAGE_PATH, [100, 10])

    training, validation, test = split_data_set((image_tensor, labels))

    normalized_image_tensor, mean, standard_deviation = normalize_data(image_tensor)

    return
