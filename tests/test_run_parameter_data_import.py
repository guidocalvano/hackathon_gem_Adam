import unittest
from run_data_parameter_import import run
import json
import os
import shutil


class test_run_parameter_data_import(unittest.TestCase):

    def setUp(self):
        self.parameter_space_file_path = './resources/parameter_data_import/parameter_space.json'

        with open(self.parameter_space_file_path) as f:
            self.standardized_photo_path = json.load(f)['data_import']['standardized_photos']

    def tearDown(self):

        if os.path.isdir(self.standardized_photo_path ):
            shutil.rmtree(self.standardized_photo_path)

    def test_everything(self):

        run(['not applicable', self.parameter_space_file_path])

        self.assertTrue(os.path.isdir('./resources/parameter_data_import/standardized_photos/-output_image_dimensions@[114, 152]'))
        self.assertTrue(os.path.isdir(
            './resources/parameter_data_import/standardized_photos/-output_image_dimensions@[228, 304]'))

        self.assertTrue(os.path.isdir(
            './resources/parameter_data_import/standardized_photos/-output_image_dimensions@[456, 608]'))
