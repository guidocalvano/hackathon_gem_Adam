import unittest
from run import run
from run_data_parameter_import import run as run_data_parameter_inport


class test_run(unittest.TestCase):
    def setUp(self):
        # reduced training data and parameter space
        self.parameter_space_file_path = './resources/run/parameter_space.json'

        pass

    def tearDown(self):
        pass

    def test_run(self):
        run_data_parameter_inport([None, self.parameter_space_file_path])

        run([None, self.parameter_space_file_path, 10])
