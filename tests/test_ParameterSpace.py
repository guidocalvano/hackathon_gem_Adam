import unittest
import os
import shutil
from ParameterSpace import ParameterSpace

class test_ParameterSpace(unittest.TestCase):

    # does flatten parameter space work?
    # does apply_instantiation work? does it not permanently alter the parameter space?
    # are parameters parsed correctly


    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'parameter_space')
        self.temp_path = os.path.join(self.base_path, 'temp')

        self.tearDown()
        os.makedirs(self.temp_path)


        self.parameter_space_file_path = os.path.join(self.temp_path, 'test_parameter_space.json')

        shutil.copyfile(os.path.join(self.base_path, 'test_parameter_space.json'), self.parameter_space_file_path)

        self.parameter_space = ParameterSpace.load(self.parameter_space_file_path)

    def tearDown(self):
        if os.path.isdir(self.temp_path):
            shutil.rmtree(self.temp_path)

    def test_vanilla_load(self):

        self.assertTrue(self.parameter_space.config == {
                "constant_number": 1,
                "constant_string": "asdf",
                "constant_array": [[1, 2, 3]],
                "constant_cell_string": ["asdf"],
                "constant_cell_number": [123],
                "deep_constant": {"constant":  1},
                "deep_cell_constant": {"constant":  [1]},

                "shallow_parameter": [1, 2, 3],
                "deep_parameter": {"parameter":  ["a", "b", "c"]},
                "deep_complex_parameter": {"parameter":  ["e", "f", "g"], "parameter2": [3, 2, 1]}
            }, 'vanilla load of config must result in correct configuration')

    def test_override_shallow_var(self):
        with self.assertRaises(SystemExit):
            self.parameter_space.setPath("constant_number", 2)

    def test_override_deep_var(self):
        with self.assertRaises(SystemExit):
            self.parameter_space.setPath("deep_cell_constant/constant", 4)

    def test_parse_multiple_args(self):
        config = self.parameter_space.parseArguments(
            ["--shallow_parameter", "2",
             "--deep_parameter/parameter", "b",
             "--deep_complex_parameter/parameter", "e",
             "--deep_complex_parameter/parameter2", "3"
             ])

        self.assertTrue(config == {
            "constant_number": 1,
            "constant_string": "asdf",
            "constant_array": [1, 2, 3],
            "constant_cell_string": "asdf",
            "constant_cell_number": 123,
            "deep_constant": {"constant": 1},
            "deep_cell_constant": {"constant": 1},

            "shallow_parameter": 2,
            "deep_parameter": {"parameter": "b"},
            "deep_complex_parameter": {"parameter":  "e", "parameter2": 3}

        }, 'multiple args must be assigned correctly')

    def test_apply_instantiation(self):
        config = self.parameter_space.apply_instantiation(
            {
                "shallow_parameter": 1,
                "deep_parameter/parameter": "b",
                "deep_complex_parameter/parameter": "e",
                "deep_complex_parameter/parameter2": 2
            })
        self.assertTrue(config == {
            "constant_number": 1,
            "constant_string": "asdf",
            "constant_array": [1, 2, 3],
            "constant_cell_string": "asdf",
            "constant_cell_number": 123,
            "deep_constant": {"constant": 1},
            "deep_cell_constant": {"constant": 1},

            "shallow_parameter": 1,
            "deep_parameter": {"parameter": "b"},
            "deep_complex_parameter": {"parameter":  "e", "parameter2": 2}

        }, 'instantiation must be assigned correctly')

    def test_flatten_paramter_space(self):
        flattened = self.parameter_space.flatten_parameters()

        self.assertTrue(flattened == {
            "shallow_parameter": [1, 2, 3],
            "deep_parameter/parameter": ["a", "b", "c"],
            "deep_complex_parameter/parameter": ["e", "f", "g"],
            "deep_complex_parameter/parameter2": [3, 2, 1]
        }, 'parameter space must be flattened correctly')
