import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sys
import config
import math
import os
from pathos.multiprocessing import ProcessingPool
from keras.utils import to_categorical
import dill
import logging


def load_data_set(image_description_file_path, image_path, target_size):
    # not finished
    image_description = pd.read_csv(image_description_file_path) # this will reduce your number of examples to 10 .iloc[:10]

    image_file_paths = get_image_file_paths(image_description.filename, image_path)

    # images, is_successful = load_image_tensor(image_file_paths, target_size)
    images, is_successful = parallel_load_image_tensor(image_file_paths, target_size, config.IMAGE_PREPROCESSING_BATCH_COUNT)

    successful_description = image_description.iloc[is_successful]

    successful_description.index = range(successful_description.shape[0])

    labels = successful_description.label_clean_int

    return images, labels # none will be replaced with labels

def get_image_file_paths(image_series, image_path):

    return list(image_path + image_series)

def load_image_tensor(image_file_paths, target_size):

    target_ratio = target_size[1] / target_size[0]

    standardized_ratio_array = []

    is_successful = np.zeros([len(image_file_paths)]).astype(bool)

    for i in range(len(image_file_paths)):
        if i % 150 == 0:
            print("progress load image tensor ratio standardization: " + str(float(i) / float(len(image_file_paths))))
        try:
            next_image = load_img(image_file_paths[i])
            image_array = np.array(next_image)

            standardized_ratio_image = standardize_image_ratio(image_array, target_ratio)

            standardized_ratio_array.append(standardized_ratio_image)
            is_successful[i] = True
        except FileNotFoundError as f:
            logging.error(f)
        except Exception as e: # this should match specific errors but if I did the code would be clean and this IS a hackathon
            print("ExceptionClass: " + e.__class__.__name__)
            print("info: " + e.__str__())
            print("str(): " + str(e))
            pass  # so I can set a break point here


    target_ratio_height_per_width = target_size[1] / target_size[0]

    downsampled_image_arrays = standardize_resolution(standardized_ratio_array, target_size)

    image_tensor = np.array([]).reshape([0, target_size[0], target_size[1], 3])

    if len(downsampled_image_arrays) > 0:
        image_tensor = np.concatenate(downsampled_image_arrays)

    # return array containing ONLY successful images (without holes) and a boolean indexing vector with successful loads
    return image_tensor, is_successful


def chunk_df(l, chunk_count):
    chunk_size = int(len(l) / chunk_count)

    chunk_size = max(1, chunk_size)

    return [l[i:i+chunk_size] for i in range(0, len(l), chunk_size)]


def construct_batch_arguments(image_file_paths, target_size, batch_count):
    batched_image_file_paths = chunk_df(image_file_paths, batch_count)

    return [(image_file_path_batch, target_size) for image_file_path_batch in batched_image_file_paths]


def load_image_tensor_task(arg_tuple):
    return load_image_tensor(arg_tuple[0], arg_tuple[1])


def parallel_load_image_tensor(image_file_paths, target_size, batch_count):
    batched_image_file_paths = construct_batch_arguments(image_file_paths, target_size, batch_count)

    standarized_images = []
    is_successful = []

    for standardized_image_batch, is_successful_batch \
            in ProcessingPool(processes=config.MULTI_CORE_COUNT).map(load_image_tensor_task, batched_image_file_paths):
        print("finished mapping")
        standarized_images.append(standardized_image_batch)
        is_successful.append(is_successful_batch)

    standarized_images = np.concatenate(standarized_images)
    is_successful = np.concatenate(is_successful)

    return standarized_images, is_successful

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
            print("progress standardize resolution: " + str(float(i) / float(len(image_arrays))))

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

def split_data_set(full_set, split_ratios):

    examples, labels = full_set

    split_indices = stratified_split_indices_from(labels, split_ratios)

    split_data_set = {
        "training": (examples[split_indices["training"]], labels[split_indices["training"]]),
        "validation": (examples[split_indices["validation"]], labels[split_indices["validation"]]),
        "test": (examples[split_indices["test"]], labels[split_indices["test"]])
    }

    return split_data_set

def stratified_split_indices_from(labels, split_ratios):
    training_ratio = split_ratios["training"]
    validation_ratio = split_ratios["validation"]
    test_ratio = split_ratios["test"]

    label_types = np.sort(np.unique(labels))
    assert abs(1 - (split_ratios["training"] + split_ratios["validation"] + split_ratios["test"])) < .01

    training_indices = []
    validation_indices = []
    test_indices = []

    for label_type in label_types:
        label_type_indices = np.where(labels == label_type)[0].copy()
        np.random.shuffle(label_type_indices)

        training_offset = 0
        validation_offset = int(label_type_indices.shape[0] * training_ratio)
        test_offset = int(label_type_indices.shape[0] * (training_ratio + validation_ratio))

        label_type_training_indices = label_type_indices[training_offset: validation_offset]
        label_type_validation_indices = label_type_indices[validation_offset:test_offset]
        label_type_test_indices = label_type_indices[test_offset:]

        training_indices.append(label_type_training_indices)
        validation_indices.append(label_type_validation_indices)
        test_indices.append(label_type_test_indices)

    training_indices = np.concatenate(training_indices)
    validation_indices = np.concatenate(validation_indices)
    test_indices = np.concatenate(test_indices)

    return {
        "training": training_indices,
        "validation": validation_indices,
        "test": test_indices
    }

def import_all_data(data_description_file, image_path, target_size):

    image_tensor, labels = load_data_set(data_description_file, image_path, target_size)

    split = split_data_set((image_tensor, labels), {
        "training": .5,
        "validation": .25,
        "test": .25
    })

    normalized_training_examples, mean, standard_deviation = normalize_data(split["training"][0])
    normalized_validation_examples = normalize_test_images(split["validation"][0], mean, standard_deviation)
    normalized_test_examples = normalize_test_images(split["test"][0], mean, standard_deviation)

    return {
        "training": (normalized_training_examples, split["training"][1]),
        "validation": (normalized_validation_examples, split["validation"][1]),
        "test": (normalized_test_examples, split["test"][1])
    }

def convert_to_categorical(ds, label_type_count):
    result = {}
    for key, data_set in ds.items():
        result[key] = (data_set[0], to_categorical(data_set[1], num_classes=label_type_count))

    return result

def default_cache_load():
    cache_file_path = config.DEFAULT_CACHE_FILE_PATH

    if os.path.isfile(cache_file_path):
        res = dill.load(open(cache_file_path, 'rb'))
        return res

    ds = convert_to_categorical(import_all_data(config.DATA_DESCRIPTION_FILE, config.IMAGE_PATH, config.INPUT_SIZE), config.CLASS_COUNT)

    dill.dump(ds, open(cache_file_path, 'wb'))

    return ds

if __name__ == "__main__":
    ds = default_cache_load()
