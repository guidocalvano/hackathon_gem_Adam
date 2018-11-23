import unittest
from Learning import Learning
import os
import os.path
from DataImporter import DataImporter
import pandas as pd
import numpy as np
import shutil


class test_Learning(unittest.TestCase):

    def setUp(self):
        self.unstandardized_photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'learning', 'unstandardized_photos')
        self.photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'learning', 'photos')
        self.test_photos_csv_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'learning', 'test_photos.csv')

        self.dataImporter = DataImporter()

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

    def tearDown(self):
        pass

    def test_simple_binary_classification(self):

        data_set = self.dataImporter.import_all_data(self.test_photos_csv_file_path, self.photos_path)

        results = Learning.simple_binary_classification(data_set, epochs=1, batch_size=2)

        self.assertTrue(np.isfinite(results["validation"]))
        self.assertTrue(np.isfinite(results["test"]))

    def test_simple_categorical_classification(self):

        data_set = self.dataImporter.import_all_data(self.test_photos_csv_file_path, self.photos_path)

        results = Learning.simple_categorical_classification(data_set, epochs=1, batch_size=2)

        self.assertTrue(np.isfinite(results["validation"]))
        self.assertTrue(np.isfinite(results["test"]))

    def test_simple_crow_score_regression(self):

        data_set = self.dataImporter.import_all_data(self.test_photos_csv_file_path, self.photos_path)

        results = Learning.simple_crow_score_regression(data_set, epochs=1, batch_size=2)

        self.assertTrue(np.isfinite(results["validation"]))
        self.assertTrue(np.isfinite(results["test"]))
