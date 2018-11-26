import unittest
from Learning import Learning
from Analysis import Analysis
import os
import os.path
from DataImporter import DataImporter
import pandas as pd
import numpy as np
import shutil
from keras.engine.training import Model


class test_Analysis(unittest.TestCase):

    def setUp(self):
        self.result_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'analysis', 'result')

        self.unstandardized_photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'analysis', 'unstandardized_photos')
        self.photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'analysis', 'photos')
        self.test_photos_csv_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'analysis', 'test_photos.csv')

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
        shutil.rmtree(self.result_path)

    def test_simple_binary_classification(self):

        data_set = self.dataImporter.import_all_data(self.test_photos_csv_file_path, self.photos_path)

        results = Learning.simple_binary_classification(data_set, epochs=1, batch_size=2)

        Analysis.store_raw_result(self.result_path, results)
        Analysis.process_result(self.result_path)
        pass

    def test_simple_categorical_classification(self):

        data_set = self.dataImporter.import_all_data(self.test_photos_csv_file_path, self.photos_path)

        results = Learning.simple_categorical_classification(data_set, epochs=1, batch_size=2)

        Analysis.store_raw_result(self.result_path, results)
        Analysis.process_result(self.result_path)
        pass

    def test_simple_crow_score_regression(self):

        data_set = self.dataImporter.import_all_data(self.test_photos_csv_file_path, self.photos_path)

        results = Learning.simple_crow_score_regression(data_set, epochs=1, batch_size=2)

        Analysis.store_raw_result(self.result_path, results)
        Analysis.process_result(self.result_path)
