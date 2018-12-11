import unittest
from SimpleLearning import SimpleLearning
import os
import os.path
from DataImporter import DataImporter
import pandas as pd
import numpy as np
import shutil
from keras.engine.training import Model


class test_SimpleLearning(unittest.TestCase):

    def setUp(self):
        self.unstandardized_photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'simple_learning', 'unstandardized_photos')
        self.photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'simple_learning', 'photos')
        self.test_photos_csv_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'simple_learning', 'test_photos.csv')

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

        results = SimpleLearning.simple_binary_classification(data_set, epochs=1, batch_size=2)

        self.assertTrue(type(results["model"]) == Model)

        self.assertTrue(np.isfinite(results["stats"]["training"]["history"]["val_acc"][-1]))
        self.assertTrue(len(results["stats"]["validation"]["predicted"]) == 4)
        self.assertTrue(len(results["stats"]["test"]["predicted"]) == 4)
        self.assertTrue(len(results["stats"]["validation"]["correct"]) == 4)
        self.assertTrue(len(results["stats"]["test"]["correct"]) == 4)
        self.assertTrue(np.all(np.isfinite(results["stats"]["validation"]["predicted"])))
        self.assertTrue(np.all(np.isfinite(results["stats"]["test"]["predicted"])))

        # could fail due to chance, just rerun
        self.assertTrue(np.max(results["stats"]["validation"]["predicted"]) == 1)

        self.assertTrue(np.isfinite(results["stats"]["validation"]["metrics"]["loss"]))
        self.assertTrue(np.isfinite(results["stats"]["validation"]["metrics"]["acc"]))
        self.assertTrue(np.isfinite(results["stats"]["test"]["metrics"]["loss"]))
        self.assertTrue(np.isfinite(results["stats"]["test"]["metrics"]["acc"]))
        self.assertTrue(results["stats"]["meta"]["labels"] == ["dirty", "clean"])


    def test_simple_categorical_classification(self):

        data_set = self.dataImporter.import_all_data(self.test_photos_csv_file_path, self.photos_path)

        results = SimpleLearning.simple_categorical_classification(data_set, epochs=1, batch_size=2)

        self.assertTrue(type(results["model"]) == Model)

        self.assertTrue(np.isfinite(results["stats"]["training"]["history"]["val_acc"][-1]))
        self.assertTrue(len(results["stats"]["validation"]["predicted"]) == 4)
        self.assertTrue(len(results["stats"]["test"]["predicted"]) == 4)
        self.assertTrue(len(results["stats"]["validation"]["correct"]) == 4)
        self.assertTrue(len(results["stats"]["test"]["correct"]) == 4)
        self.assertTrue(np.all(np.isfinite(results["stats"]["validation"]["predicted"])))
        self.assertTrue(np.all(np.isfinite(results["stats"]["test"]["predicted"])))

        # could fail due to chance, just rerun
        self.assertTrue(np.max(results["stats"]["validation"]["predicted"]) > 1)

        self.assertTrue(np.isfinite(results["stats"]["validation"]["metrics"]["loss"]))
        self.assertTrue(np.isfinite(results["stats"]["validation"]["metrics"]["acc"]))
        self.assertTrue(np.isfinite(results["stats"]["test"]["metrics"]["loss"]))
        self.assertTrue(np.isfinite(results["stats"]["test"]["metrics"]["acc"]))

        self.assertTrue(results["stats"]["meta"]["labels"] == ["w", "x", "y", "z"])


    def test_simple_crow_score_regression(self):

        data_set = self.dataImporter.import_all_data(self.test_photos_csv_file_path, self.photos_path)

        results = SimpleLearning.simple_crow_score_regression(data_set, epochs=1, batch_size=2)

        self.assertTrue(type(results["model"]) == Model)

        self.assertTrue(np.isfinite(results["stats"]["training"]["history"]["val_mean_squared_error"][-1]))
        self.assertTrue(len(results["stats"]["validation"]["predicted"]) == 4)
        self.assertTrue(len(results["stats"]["test"]["predicted"]) == 4)
        self.assertTrue(len(results["stats"]["validation"]["correct"]) == 4)
        self.assertTrue(len(results["stats"]["test"]["correct"]) == 4)
        self.assertTrue(np.all(np.isfinite(results["stats"]["validation"]["predicted"])))
        self.assertTrue(np.all(np.isfinite(results["stats"]["test"]["predicted"])))

        self.assertTrue(np.max(results["stats"]["validation"]["predicted"]) > 1)
        self.assertTrue(not np.all(np.round(results["stats"]["validation"]["predicted"]) == results["stats"]["validation"]["predicted"]))

        self.assertTrue(np.isfinite(results["stats"]["validation"]["metrics"]["loss"]))
        self.assertTrue(np.isfinite(results["stats"]["validation"]["metrics"]["mean_squared_error"]))
        self.assertTrue(np.isfinite(results["stats"]["test"]["metrics"]["loss"]))
        self.assertTrue(np.isfinite(results["stats"]["test"]["metrics"]["mean_squared_error"]))

        self.assertTrue(results["stats"]["meta"]["labels"] is None)
