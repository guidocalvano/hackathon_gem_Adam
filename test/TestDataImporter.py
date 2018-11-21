import unittest
import os
from Configuration import Configuration
from DataImporter import DataImporter
import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img
import shutil
import scipy.stats


class TestDataImporter(unittest.TestCase):

    def setUp(self):

        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'dataImport', 'test_config.json')
        self.photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'dataImport', 'photos')
        self.unstandardized_photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'dataImport', 'unstandardized_photos')

        self.edge_case_photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'dataImport', 'edgeCasePhotos')
        self.fixed_size_edge_case_photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'dataImport', 'fixed_size_edge_case_photos')

        self.fixed_size_images_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'dataImport', 'fixed_size_images')

        self.temp_reduced_edge_case_photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'dataImport', 'temp_reduced')


        self.test_photos_csv_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'dataImport', 'test_photos.csv')
        self.reduced_test_photos_csv_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'dataImport', 'reduced_test_photos.csv')

        self.config = Configuration.load(self.config_path)

        self.dataImporter = DataImporter()

        self.test_photos_description = pd.read_csv(self.test_photos_csv_file_path)

        self.reduced_test_photos_description = pd.read_csv(self.reduced_test_photos_csv_file_path)

        self.cache_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'dataImport', 'cache.dill')

    def tearDown(self):

        self.config = None
        self.dataImporter = None

        if os.path.isdir(self.temp_reduced_edge_case_photos_path):
            shutil.rmtree(self.temp_reduced_edge_case_photos_path)

        if os.path.isdir(self.photos_path):
            shutil.rmtree(self.photos_path)

        if os.path.isfile(self.cache_file_path):
            os.remove(self.cache_file_path)

    # file management
    def test_load_from_cache(self):
        self.dataImporter.convert_to_standard_resolution(
            self.unstandardized_photos_path,
            self.photos_path,
            [152, 114],
            {
                "multi_core_count": 2,
                "timeout_secs": 100,
                "chunk_size": 2
            }
        )

        self.assertTrue(not os.path.isfile(self.cache_file_path))

        data_set_dictionary = self.dataImporter.load_from_cache({
            "cache_file_path": self.cache_file_path,
            "data_description_file_path": self.test_photos_csv_file_path,
            "image_path": self.photos_path
        })

        first_crow_labels = np.array(data_set_dictionary["training"][1].label_crow_score_int)

        self.assertTrue(os.path.isfile(self.cache_file_path))

        modification_time = os.path.getmtime(self.cache_file_path)
        creation_time = os.path.getctime(self.cache_file_path)

        data_set_dictionary2 = self.dataImporter.load_from_cache({
            "cache_file_path": self.cache_file_path,
            "data_description_file_path": self.test_photos_csv_file_path,
            "image_path": self.photos_path
        })

        second_crow_labels = np.array(data_set_dictionary2["training"][1].label_crow_score_int)

        self.assertTrue(np.all(first_crow_labels == second_crow_labels))

        self.assertTrue(modification_time == os.path.getmtime(self.cache_file_path))
        self.assertTrue(creation_time == os.path.getctime(self.cache_file_path))

    # data splitting

    def test_import_all_data(self):

        self.dataImporter.convert_to_standard_resolution(
            self.unstandardized_photos_path,
            self.photos_path,
            [152, 114],
            {
                "multi_core_count": 2,
                "timeout_secs": 100,
                "chunk_size": 2
            }
        )

        data_set_dictionary = self.dataImporter.import_all_data(self.test_photos_csv_file_path, self.photos_path)

        self.assertTrue(np.all(data_set_dictionary["training"][0].shape == (8, 152, 114, 3)))
        self.assertTrue(np.all(data_set_dictionary["validation"][0].shape == (4, 152, 114, 3)))
        self.assertTrue(np.all(data_set_dictionary["test"][0].shape == (4, 152, 114, 3)))

        #  all data must be uniquely assigned to either training validation or test
        self.assertTrue(np.unique(np.concatenate([np.array(data_set_dictionary["test"][1].label_crow_score_int), np.array(data_set_dictionary["validation"][1].label_crow_score_int), np.array(data_set_dictionary["training"][1].label_crow_score_int )])).shape[0] == 16)
        self.assertTrue(len(set(list(data_set_dictionary["meta"]["test"].label_crow_score_str) + list(data_set_dictionary["meta"]["validation"].label_crow_score_str) + list(data_set_dictionary["meta"]["training"].label_crow_score_str) )) == 16)


    def test_split_data_set(self):

        label_type_int_labels = np.tile(np.array([0, 1, 2, 2, 3, 3, 3, 3]).repeat(5), [4])

        labels_df = pd.DataFrame({
            'label_type_int': label_type_int_labels,
            'label_id': np.arange(label_type_int_labels.shape[0])
        })

        mock_image_tensors = np.random.random([label_type_int_labels.shape[0], 4, 4, 3])

        data_set = (mock_image_tensors, labels_df)

        split_data_set, split_indices = self.dataImporter.split_data_set(data_set, {"training": .5, "validation": .25, "test": .25})

        self.assertTrue(
            np.sum(split_data_set["training"][1].label_type_int == 0) == (np.sum(label_type_int_labels == 1) / 2))

        self.assertTrue(
            np.sum(split_data_set["validation"][1].label_type_int == 2) == (np.sum(label_type_int_labels == 2) / 4))

        self.assertTrue(
            np.sum(split_data_set["test"][1].label_type_int == 3) == (np.sum(label_type_int_labels == 3) / 4))

        #  split indices are all disjoint subsets of the original index set
        self.assertTrue(np.unique(np.concatenate([split_indices["training"], split_indices["validation"], split_indices["test"]])).shape[0] == label_type_int_labels.shape[0])

        #  images are indexed correctly
        self.assertTrue(np.all(mock_image_tensors[split_indices["training"], :, :, :] == split_data_set["training"][0]))
        self.assertTrue(np.all(mock_image_tensors[split_indices["validation"], :, :, :] == split_data_set["validation"][0]))
        self.assertTrue(np.all(mock_image_tensors[split_indices["test"], :, :, :] == split_data_set["test"][0]))

        self.assertTrue(np.all(np.array(labels_df.label_id[split_indices["training"]]) == np.array(split_data_set["training"][1].label_id)))
        self.assertTrue(np.all(np.array(labels_df.label_id[split_indices["validation"]]) == np.array(split_data_set["validation"][1].label_id)))
        self.assertTrue(np.all(np.array(labels_df.label_id[split_indices["test"]]) == np.array(split_data_set["test"][1].label_id)))

    def test_stratified_split_indices_from(self):

        # three labels, where 1/8 is 0, 1/8 is 1 and 1/4 is 2 and 1/2 is 3
        labels = np.tile(np.array([0, 1, 2, 2, 3, 3, 3, 3]).repeat(5), [4])

        split = self.dataImporter.stratified_split_indices_from(labels, {"training": .5, "validation": .25, "test": .25})

        training_labels = labels[split["training"]]
        validation_labels = labels[split["validation"]]
        test_labels = labels[split["test"]]

        self.assertTrue(training_labels.shape[0] == (labels.shape[0] / 2))
        self.assertTrue(validation_labels.shape[0] == (labels.shape[0] / 4))
        self.assertTrue(test_labels.shape[0] == (labels.shape[0] / 4))

        self.assertTrue(np.sum(training_labels == 0) == (training_labels.shape[0] / 8))
        self.assertTrue(np.sum(training_labels == 1) == (training_labels.shape[0] / 8))
        self.assertTrue(np.sum(training_labels == 2) == (training_labels.shape[0] / 4))
        self.assertTrue(np.sum(training_labels == 3) == (training_labels.shape[0] / 2))

        self.assertTrue(np.sum(validation_labels == 0) == (validation_labels.shape[0] / 8))
        self.assertTrue(np.sum(validation_labels == 1) == (validation_labels.shape[0] / 8))
        self.assertTrue(np.sum(validation_labels == 2) == (validation_labels.shape[0] / 4))
        self.assertTrue(np.sum(validation_labels == 3) == (validation_labels.shape[0] / 2))

        self.assertTrue(np.sum(test_labels == 0) == (test_labels.shape[0] / 8))
        self.assertTrue(np.sum(test_labels == 1) == (test_labels.shape[0] / 8))
        self.assertTrue(np.sum(test_labels == 2) == (test_labels.shape[0] / 4))
        self.assertTrue(np.sum(test_labels == 3) == (test_labels.shape[0] / 2))

        #  note that due to the small sample size it might accidentally think there is a spearman correlation
        self.assertTrue(scipy.stats.spearmanr(np.arange(training_labels.shape[0]), training_labels).pvalue > .1)
        self.assertTrue(scipy.stats.spearmanr(np.arange(validation_labels.shape[0]), validation_labels).pvalue > .1)
        self.assertTrue(scipy.stats.spearmanr(np.arange(test_labels.shape[0]), test_labels).pvalue > .1)

    # data loading, saving and parallelization
    def test_load_data_set(self):
        # image_description_file_path, image_path

        self.dataImporter.convert_to_standard_resolution(
            self.edge_case_photos_path,
            self.fixed_size_edge_case_photos_path,
            [152, 114],
            {
                "multi_core_count": 2,
                "timeout_secs": 100,
                "chunk_size": 2
            }
        )

        images, labels, meta = self.dataImporter.load_data_set(self.reduced_test_photos_csv_file_path, self.fixed_size_edge_case_photos_path)

        self.assertTrue(np.all(meta.filename == ['high.jpg', 'wide.jpg', 'bigSquare.jpg', 'smallSquare.jpg', 'exact.jpg']))
        self.assertTrue(np.all(meta.label_crow_score_str == ['A', 'B', 'C', 'D', 'E']))
        self.assertTrue(np.all(meta.label_type_str == ['gemengd', 'plastic', 'gemengd', 'gemengd', 'gemengd']))

        self.assertTrue(np.all(labels.label_type_int == [7, 1, 7, 7, 7]), 'label type should be loaded correctly')
        self.assertTrue(np.all(labels.label_crow_score_int == [0, 1, 2, 3, 5]), 'label crow score should be loaded correctly')
        self.assertTrue(np.all(labels.label_clean_int == [0, 1, 0, 1, 0]), 'label clean int should be loaded correctly')
        self.assertTrue(images.shape == (5, 152, 114, 3))

    def test_convert_to_standard_resolution(self):
        self.dataImporter.convert_to_standard_resolution(
            self.edge_case_photos_path,
            self.temp_reduced_edge_case_photos_path,
            [152, 114],
            {
                "multi_core_count": 2,
                "timeout_secs": 100,
                "chunk_size": 2
            }
        )

        source_path = self.temp_reduced_edge_case_photos_path
        target_images = list(filter(lambda name: name.endswith('.jpg') and (not name.startswith('.')), os.listdir(source_path)))

        source_image_file_paths = list(map(lambda file_name: os.path.join(source_path, file_name), target_images))

        image_arrays, is_successful = self.dataImporter.load_image_tensors(source_image_file_paths)

        self.assertTrue(image_arrays.shape == (5, 152, 114, 3))

    def test_parallel_load_image_tensor(self):
        source_path = self.edge_case_photos_path
        target_images = list(filter(lambda name: name.endswith('.jpg') and (not name.startswith('.')), os.listdir(source_path)))

        source_image_file_paths = list(map(lambda file_name: os.path.join(source_path, file_name), target_images))

        image_arrays, is_successful = \
            self.dataImporter.parallel_load_image_tensor(
                source_image_file_paths,
                [152, 114],
                {
                    "multi_core_count": 2,
                    "timeout_secs": 100,
                    "chunk_size": 2
                }
            )

        self.assertTrue(image_arrays.shape == (5, 152, 114, 3))

    def test_load_image_tensors(self):
        source_path = self.fixed_size_images_path
        target_images = list(filter(lambda name: name.endswith('.jpg') and (not name.startswith('.')), os.listdir(source_path)))

        source_image_file_paths = map(lambda file_name: os.path.join(source_path, file_name), target_images)

        image_arrays, is_successful = self.dataImporter.load_image_tensors(source_image_file_paths)

        self.assertTrue(image_arrays.shape == (3, 152, 114, 3))

    # image processing

    def test_load_and_standardize_image(self):
        big_standard_resolution_image, is_successful =  self.dataImporter.load_and_standardize_image(os.path.join(self.edge_case_photos_path, 'bigSquare.jpg'), [114, 152])

        self.assertTrue(big_standard_resolution_image.shape == (1, 114, 152, 3), 'files must be loaded and standardized correctly')
        self.assertTrue(is_successful)

        missing_image, is_successful =  self.dataImporter.load_and_standardize_image(os.path.join(self.edge_case_photos_path, 'asdf.jpg'), [114, 152])

        self.assertTrue(missing_image is None, 'files must be loaded and standardized correctly')
        self.assertTrue(not is_successful)

    def test_standardize_image_ratio(self):

        target_ratio = 114.0 / 152.0

        big_image =  np.array(load_img(os.path.join(self.edge_case_photos_path, 'bigSquare.jpg')))
        small_image =  np.array(load_img(os.path.join(self.edge_case_photos_path, 'smallSquare.jpg')))
        wide_image =  np.array(load_img(os.path.join(self.edge_case_photos_path, 'wide.jpg')))
        high_image =  np.array(load_img(os.path.join(self.edge_case_photos_path, 'high.jpg')))
        exact_image =  np.array(load_img(os.path.join(self.edge_case_photos_path, 'exact.jpg')))

        big_standard_ratio_image = self.dataImporter.standardize_image_ratio(big_image, target_ratio)
        small_standard_ratio_image = self.dataImporter.standardize_image_ratio(small_image, target_ratio)
        wide_standard_ratio_image = self.dataImporter.standardize_image_ratio(wide_image, target_ratio)
        high_standard_ratio_image = self.dataImporter.standardize_image_ratio(high_image, target_ratio)
        exact_standard_ratio_image = self.dataImporter.standardize_image_ratio(exact_image, target_ratio)

        big_image_standard_ratio = big_standard_ratio_image.shape[1] / big_standard_ratio_image.shape[0]
        small_image_standard_ratio = small_standard_ratio_image.shape[1] / small_standard_ratio_image.shape[0]
        wide_image_standard_ratio = wide_standard_ratio_image.shape[1] / wide_standard_ratio_image.shape[0]
        high_image_standard_ratio = high_standard_ratio_image.shape[1] / high_standard_ratio_image.shape[0]
        exact_image_standard_ratio = exact_standard_ratio_image.shape[1] / exact_standard_ratio_image.shape[0]

        self.assertTrue(abs(big_image_standard_ratio - target_ratio) < .05, 'edge case ratio must be close to target ratio')
        self.assertTrue(abs(small_image_standard_ratio - target_ratio) < .1, 'edge case ratio must be close to target ratio')
        self.assertTrue(abs(wide_image_standard_ratio - target_ratio) < .1, 'edge case ratio must be close to target ratio')
        self.assertTrue(abs(high_image_standard_ratio - target_ratio) < .1, 'edge case ratio must be close to target ratio')
        self.assertTrue(abs(exact_image_standard_ratio - target_ratio) <= .0, 'edge case ratio must be close to target ratio')

    def test_standardize_resolution(self):

        target_ratio = 114.0 / 152.0

        big_image = np.array(load_img(os.path.join(self.edge_case_photos_path, 'bigSquare.jpg')))
        small_image = np.array(load_img(os.path.join(self.edge_case_photos_path, 'smallSquare.jpg')))
        wide_image = np.array(load_img(os.path.join(self.edge_case_photos_path, 'wide.jpg')))
        high_image = np.array(load_img(os.path.join(self.edge_case_photos_path, 'high.jpg')))
        exact_image = np.array(load_img(os.path.join(self.edge_case_photos_path, 'exact.jpg')))

        big_standard_resolution_image = self.dataImporter.standardize_resolution(self.dataImporter.standardize_image_ratio(big_image, target_ratio), [114, 152])
        small_standard_resolution_image = self.dataImporter.standardize_resolution(self.dataImporter.standardize_image_ratio(small_image, target_ratio), [114, 152])
        wide_standard_resolution_image = self.dataImporter.standardize_resolution(self.dataImporter.standardize_image_ratio(wide_image, target_ratio), [114, 152])
        high_standard_resolution_image = self.dataImporter.standardize_resolution(self.dataImporter.standardize_image_ratio(high_image, target_ratio), [114, 152])
        exact_standard_resolution_image = self.dataImporter.standardize_resolution(self.dataImporter.standardize_image_ratio(exact_image, target_ratio), [114, 152])

        self.assertTrue(big_standard_resolution_image.shape == (1, 114, 152, 3), 'resolution must be changed correctly')
        self.assertTrue(small_standard_resolution_image.shape == (1, 114, 152, 3), 'resolution must be changed correctly')
        self.assertTrue(wide_standard_resolution_image.shape == (1, 114, 152, 3), 'resolution must be changed correctly')
        self.assertTrue(high_standard_resolution_image.shape == (1, 114, 152, 3), 'resolution must be changed correctly')
        self.assertTrue(exact_standard_resolution_image.shape == (1, 114, 152, 3), 'resolution must be changed correctly')

    def test_normalize_data(self):
        normalized_data, mean, std = self.dataImporter.normalize_data(np.random.normal(3, 7, [1000, 100, 200, 3]))

        normalized_data_mean = np.mean(normalized_data)
        normalized_data_std = np.std(normalized_data)

        self.assertTrue(np.abs(normalized_data_mean) < .001, 'mean must be normalized')
        self.assertTrue(np.abs(normalized_data_std - 1.0) < .001, 'std must be normalized')
        self.assertTrue(np.abs(mean - 3.0) < .001, 'mean must be correct')
        self.assertTrue(np.abs(std - 7) < .001, 'std must be correct')

    def test_normalize_test_images(self):
        normalized_data, mean, std = self.dataImporter.normalize_data(np.random.normal(3, 7, [1000, 100, 200, 3]))

        normalized_test_images = self.dataImporter.normalize_test_images(np.random.normal(3, 7, [500, 100, 200, 3]), mean, std)

        normalized_test_data_mean = np.mean(normalized_test_images)
        normalized_test_data_std = np.std(normalized_test_images)

        self.assertTrue(np.abs(normalized_test_data_mean) < .01, 'mean must be normalized')
        self.assertTrue(np.abs(normalized_test_data_std - 1.0) < .01, 'std must be normalized')


    # util functions

    def test_index_pandas_df(self):
        pass

    def test_get_image_file_paths(self):
        series = self.reduced_test_photos_description.label_crow_score_str

        file_paths = self.dataImporter.get_image_file_paths(series, 'x')

        self.assertTrue(file_paths == ['x/A', 'x/B', 'x/C', 'x/D', 'x/E'])

        file_paths = self.dataImporter.get_image_file_paths(series, 'y/')

        self.assertTrue(file_paths == ['y/A', 'y/B', 'y/C', 'y/D', 'y/E'])
