import os
import pickle
import pandas as pd
import numpy as np
from pebble import ProcessPool
# from keras.preprocessing.image import load_img
import imageio
import tensorflow as tf
import sys
import scipy
import scipy.misc
from itertools import compress
import json


class DataImporter:

    # file management

    def convert_to_standard_resolution(self, source_path, sink_path, output_image_dimensions, loading_config):
        if not os.path.isdir(sink_path): os.makedirs(sink_path)

        target_images = list(filter(lambda name: name.endswith('.jpg') and (not name.startswith('.')), os.listdir(source_path)))

        source_image_file_paths = list(map(lambda file_name: os.path.join(source_path, file_name), target_images))

        standardized_images, is_successful = self.parallel_load_image_tensor(source_image_file_paths, output_image_dimensions, loading_config)

        successful_target_images = list(compress(target_images, is_successful))

        sink_image_file_paths = list(map(lambda file_name: os.path.join(sink_path, file_name), successful_target_images))

        self.save_image_tensors(sink_image_file_paths, standardized_images)

    @staticmethod
    def load_from_cache(cache_file_path, data_description_file_path, standardized_photos_file_path):
        if os.path.isfile(cache_file_path):
            all_data = pickle.load(open(cache_file_path, 'rb'))
            return all_data
        di = DataImporter()

        all_data = di.import_all_data(data_description_file_path, standardized_photos_file_path)

        pickle.dump(all_data, open(cache_file_path, 'wb'), protocol=-1)

        return all_data

    # data splitting

    def import_all_data(self, data_description_file, image_path):

        image_tensor, labels, meta = self.load_data_set(data_description_file, image_path)

        split, split_indices = self.split_data_set((image_tensor, labels), {
            "training": .5,
            "validation": .25,
            "test": .25
        })

        normalized_training_examples, mean, standard_deviation = self.normalize_data(split["training"][0])
        normalized_validation_examples = self.normalize_test_images(split["validation"][0], mean, standard_deviation)
        normalized_test_examples = self.normalize_test_images(split["test"][0], mean, standard_deviation)

        split_meta = {
            "training": self.index_pandas_df(meta, split_indices["training"]),
            "validation": self.index_pandas_df(meta, split_indices["validation"]),
            "test": self.index_pandas_df(meta, split_indices["test"])
        }

        return {
            "training": (normalized_training_examples, split["training"][1]),
            "validation": (normalized_validation_examples, split["validation"][1]),
            "test": (normalized_test_examples, split["test"][1]),
            "meta": split_meta
        }

    def split_data_set(self, full_set, split_ratios):

        examples, labels = full_set

        split_indices = self.stratified_split_indices_from(labels.label_type_int, split_ratios)

        split_data_set = {
            "training": (examples[split_indices["training"]], self.index_pandas_df(labels, split_indices["training"])),
            "validation": (examples[split_indices["validation"]], self.index_pandas_df(labels, split_indices["validation"])),
            "test": (examples[split_indices["test"]], self.index_pandas_df(labels, split_indices["test"]))
        }

        return split_data_set, split_indices

    def stratified_split_indices_from(self, labels, split_ratios):

        training_ratio = split_ratios["training"]
        validation_ratio = split_ratios["validation"]
        test_ratio = split_ratios["test"]

        assert abs(1 - (split_ratios["training"] + split_ratios["validation"] + split_ratios["test"])) < .01

        training_indices = []
        validation_indices = []
        test_indices = []

        label_types = np.sort(np.unique(labels))

        for label_type in label_types:
            #  filter out all indices of labels of a specific type
            label_type_indices = np.where(labels == label_type)[0].copy()

            #  shuffle those indices
            np.random.shuffle(label_type_indices)

            #  split the shuffled indices into three sections according to the split ratios
            #    first compute the boundries of the sections
            training_offset = 0
            validation_offset = int(label_type_indices.shape[0] * training_ratio)
            test_offset = int(label_type_indices.shape[0] * (training_ratio + validation_ratio))

            #    slice according to the sections
            label_type_training_indices = label_type_indices[training_offset: validation_offset]
            label_type_validation_indices = label_type_indices[validation_offset:test_offset]
            label_type_test_indices = label_type_indices[test_offset:]

            #    add the extracted random indices to the full training, validation and test sets
            training_indices.append(label_type_training_indices)
            validation_indices.append(label_type_validation_indices)
            test_indices.append(label_type_test_indices)

        #  merge everything cleanly into numpy arrays
        #  AND RESHUFFLE THOSE !#$!#@$ %INDICES
        training_indices = np.concatenate(training_indices)
        validation_indices = np.concatenate(validation_indices)
        test_indices = np.concatenate(test_indices)

        np.random.shuffle(training_indices)
        np.random.shuffle(validation_indices)
        np.random.shuffle(test_indices)

        #  return the stratitified random indices in a fancy dictionary
        return {
            "training": training_indices,
            "validation": validation_indices,
            "test": test_indices
        }

    # data loading and paralelization

    def load_data_set(self, image_description_file_path, image_path):
        # not finished
        image_description = pd.read_csv(
            image_description_file_path)  # this will reduce your number of examples to 10 .iloc[:10]

        image_file_paths = self.get_image_file_paths(image_description.filename, image_path)

        # images, is_successful = load_image_tensor(image_file_paths)
        images, is_successful = self.load_image_tensors(image_file_paths)

        successful_image_description = image_description.iloc[is_successful]

        successful_image_description.index = range(successful_image_description.shape[0])

        label_columns = ["label_clean_int", "label_type_int", "label_crow_score_int"]
        labels = successful_image_description[label_columns]

        meta = successful_image_description.loc[:,
               np.logical_not(np.isin(successful_image_description.columns, label_columns))]

        return images, labels, meta

    def load_image_tensors(self, image_file_paths):

        image_array_list = []
        is_successful_list = []

        for image_file_path in image_file_paths:
            try:
                next_image = imageio.imread(image_file_path)
                image_array = np.expand_dims(next_image, 0)

                image_array_list.append(image_array)
                is_successful_list.append(True)
            except FileNotFoundError:
                is_successful_list.append(False)

        return np.concatenate(image_array_list), is_successful_list

    def save_image_tensors(self, image_file_paths, image_array_list):
        for i in range(len(image_array_list)):
            scipy.misc.toimage(image_array_list[i], cmin=0.0, cmax=255).save(image_file_paths[i])

    def parallel_load_image_tensor(self, image_file_paths, output_image_dimensions, loading_config):

        standarized_images = []  # np.array([]).reshape([0, output_image_dimensions[0], output_image_dimensions[1], 3])
        is_successful_array = []

        tasks = self.construct_parallel_task_arguments(image_file_paths, output_image_dimensions)

        with ProcessPool(max_workers=loading_config["multi_core_count"]) as pool:
            future = pool.map(parallel_load_and_standardize_image, tasks, timeout=loading_config["timeout_secs"],
                              chunksize=loading_config["chunk_size"])

            iterator = future.result()
            i = 0
            while True:
                try:
                    standardized_image = None
                    standardized_image, is_successful = next(iterator)

                    if not is_successful:
                        print("returned unsuccessfully")
                        raise Exception("result returned false")

                    if standardized_image is None:
                        raise Exception("No image returned")

                    if standardized_image.shape[1:3] != (output_image_dimensions[0], output_image_dimensions[1]):
                         raise Exception("dimension mismatch")

                    if standardized_image.shape[3] != 3:
                         raise Exception("not RGB image")

                    standarized_images.append(standardized_image)
                    is_successful_array.append(is_successful)
                except StopIteration:
                    break
                except Exception as e:
                    print("exception")
                    print(e)
                    print(image_file_paths[i])
                    if (standardized_image is not None) and hasattr(standardized_image, 'shape'):
                        print("target dimensions: " + json.stringify(output_image_dimensions))
                        print("actual dimensions: " + json.stringify(standardized_image.shape))
                    is_successful_array.append(False)

                print("RESULTS PROCESSED = " + str(i + 1) + " / " + str(len(list(image_file_paths))))

                i = i + 1

        print("finished standardizing images")

        print("total images retrieved = " + str(len(standarized_images)))

        standarized_images = np.concatenate(standarized_images)
        is_successful_array = np.array(is_successful_array)

        return standarized_images, is_successful_array

    def construct_parallel_task_arguments(self, image_file_paths, output_image_dimensions):

        return [(self, image_file_path, output_image_dimensions) for image_file_path in image_file_paths]

    # image processing

    def load_and_standardize_image(self, image_file_path, output_image_dimensions):

        target_ratio = output_image_dimensions[1] / output_image_dimensions[0]

        standardized_ratio_array = []

        try:
            image_array = imageio.imread(image_file_path)

            standardized_ratio_image = self.standardize_image_ratio(image_array, target_ratio)

            standardized_ratio_array.append(standardized_ratio_image)

            downsampled_image = self.standardize_resolution(standardized_ratio_image, output_image_dimensions)

        except Exception as e:
            print(e.__class__.__name__)
            print(str(e))

            return None, False
        # successful image
        return downsampled_image, True

    def standardize_image_ratio(self, image_array, target_ratio_height_per_width):
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

    def standardize_resolution(self, image_array, target_size):
        sess = tf.Session()

        resized_image = None

        with sess.as_default():
                original = image_array.astype('float')
                original = np.expand_dims(original, axis=0)
                image_tf = tf.placeholder(tf.float32, shape=original.shape)

                resized_image_tf = tf.image.resize_images(image_tf, size=target_size)

                resized_image = resized_image_tf.eval({image_tf: original})

        return resized_image

    def normalize_data(self, data):  # does not take into account night or day
        mean = np.mean(data)
        centered_data = data - mean
        standard_deviation = np.std(centered_data) + sys.float_info.epsilon

        standardized_data = centered_data / standard_deviation

        return standardized_data, mean, standard_deviation

    def normalize_test_images(self, data, mean, standard_deviation):
        centered_data = data - mean

        standardized_data = centered_data / standard_deviation

        return standardized_data

    # util functions

    def index_pandas_df(self, df, indices):
        df = df.iloc[indices]
        df.index = range(len(df))

        return df

    def get_image_file_paths(self, image_series, image_path):
        return [os.path.join(image_path, image_name) for image_name in list(image_series) ]

def parallel_load_and_standardize_image(args):
    di, image_file_path, output_image_dimensions = args
    return di.load_and_standardize_image(image_file_path, output_image_dimensions)