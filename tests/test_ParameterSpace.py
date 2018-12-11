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
            "configuration_name": "-deep_complex_parameter@parameter@e-deep_complex_parameter@parameter2@2-deep_parameter@parameter@b-shallow_parameter@1",
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

    def test_flatten_parameter_space(self):
        flattened = self.parameter_space.flatten_parameters()

        self.assertTrue(flattened == {
            "shallow_parameter": [1, 2, 3],
            "deep_parameter/parameter": ["a", "b", "c"],
            "deep_complex_parameter/parameter": ["e", "f", "g"],
            "deep_complex_parameter/parameter2": [3, 2, 1]
        }, 'parameter space must be flattened correctly')

    def test__flattened_space_unit_sizes(self):
        unit_sizes, space_size = self.parameter_space._flattened_space_unit_sizes({
            "a": [4, 5, 6],
            "b": [2, 3],
            "c": ["x", "y", "z", "d"]
        })

        self.assertTrue(space_size == 24)

        self.assertTrue(unit_sizes["a"] == 8)
        self.assertTrue(unit_sizes["b"] == 4)
        self.assertTrue(unit_sizes["c"] == 1)

    def test__configuration_index_to_parameter_indices(self):
        unit_sizes, space_size = self.parameter_space._flattened_space_unit_sizes({
            "a": [4, 5, 6],
            "b": [2, 3],
            "c": ["x", "y", "z", "d"]
        })

        index_map = self.parameter_space._configuration_index_to_parameter_indices(unit_sizes, 0)

        self.assertTrue(index_map["a"] == 0)
        self.assertTrue(index_map["b"] == 0)
        self.assertTrue(index_map["c"] == 0)

        index_map = self.parameter_space._configuration_index_to_parameter_indices(unit_sizes, 1)

        self.assertTrue(index_map["a"] == 0)
        self.assertTrue(index_map["b"] == 0)
        self.assertTrue(index_map["c"] == 1)

        index_map = self.parameter_space._configuration_index_to_parameter_indices(unit_sizes, 4)

        self.assertTrue(index_map["a"] == 0)
        self.assertTrue(index_map["b"] == 1)
        self.assertTrue(index_map["c"] == 0)

        index_map = self.parameter_space._configuration_index_to_parameter_indices(unit_sizes, 8)

        self.assertTrue(index_map["a"] == 1)
        self.assertTrue(index_map["b"] == 0)
        self.assertTrue(index_map["c"] == 0)

        index_map = self.parameter_space._configuration_index_to_parameter_indices(unit_sizes, 8 + 4 + 1)

        self.assertTrue(index_map["a"] == 1)
        self.assertTrue(index_map["b"] == 1)
        self.assertTrue(index_map["c"] == 1)

        index_map = self.parameter_space._configuration_index_to_parameter_indices(unit_sizes, 23)

        self.assertTrue(index_map["a"] == 2)
        self.assertTrue(index_map["b"] == 1)
        self.assertTrue(index_map["c"] == 3)

    def test___flat_space_index_to_configuration(self):

        flattened_space = {
            "a": [4, 5, 6],
            "b": [2, 3],
            "c": ["x", "y", "z", "d"]
        }

        unit_sizes, space_size = self.parameter_space._flattened_space_unit_sizes(flattened_space)

        config_map = self.parameter_space._flat_space_index_to_configuration(
            flattened_space,
            self.parameter_space._configuration_index_to_parameter_indices(unit_sizes, 0))


        self.assertTrue(config_map["a"] == 4)
        self.assertTrue(config_map["b"] == 2)
        self.assertTrue(config_map["c"] == "x")

        config_map = self.parameter_space._flat_space_index_to_configuration(
            flattened_space,
            self.parameter_space._configuration_index_to_parameter_indices(unit_sizes, 1))

        self.assertTrue(config_map["a"] == 4)
        self.assertTrue(config_map["b"] == 2)
        self.assertTrue(config_map["c"] == "y")

        config_map = self.parameter_space._flat_space_index_to_configuration(
            flattened_space,
            self.parameter_space._configuration_index_to_parameter_indices(unit_sizes, 4))

        self.assertTrue(config_map["a"] == 4)
        self.assertTrue(config_map["b"] == 3)
        self.assertTrue(config_map["c"] == "x")

        config_map = self.parameter_space._flat_space_index_to_configuration(
            flattened_space,
            self.parameter_space._configuration_index_to_parameter_indices(unit_sizes, 8))

        self.assertTrue(config_map["a"] == 5)
        self.assertTrue(config_map["b"] == 2)
        self.assertTrue(config_map["c"] == "x")

        config_map = self.parameter_space._flat_space_index_to_configuration(
            flattened_space,
            self.parameter_space._configuration_index_to_parameter_indices(unit_sizes, 8 + 4 + 1))

        self.assertTrue(config_map["a"] == 5)
        self.assertTrue(config_map["b"] == 3)
        self.assertTrue(config_map["c"] == "y")

        config_map = self.parameter_space._flat_space_index_to_configuration(
            flattened_space,
            self.parameter_space._configuration_index_to_parameter_indices(unit_sizes, 23))

        self.assertTrue(config_map["a"] == 6)
        self.assertTrue(config_map["b"] == 3)
        self.assertTrue(config_map["c"] == "d")

    def test_list_all_flattened_parameter_configurations(self):
        flattened_space = {
            "a": [4, 5, 6],
            "b": [2, 3],
            "c": ["x", "y", "z", "d"]
        }

        configs = self.parameter_space.list_all_flattened_parameter_configurations(flattened_space)

        config_map = configs[0]

        self.assertTrue(config_map["a"] == 4)
        self.assertTrue(config_map["b"] == 2)
        self.assertTrue(config_map["c"] == "x")

        config_map = configs[1]

        self.assertTrue(config_map["a"] == 4)
        self.assertTrue(config_map["b"] == 2)
        self.assertTrue(config_map["c"] == "y")

        config_map = configs[4]

        self.assertTrue(config_map["a"] == 4)
        self.assertTrue(config_map["b"] == 3)
        self.assertTrue(config_map["c"] == "x")

        config_map = configs[23]

        self.assertTrue(config_map["a"] == 6)
        self.assertTrue(config_map["b"] == 3)
        self.assertTrue(config_map["c"] == "d")

    def test_get_configuration_grid(self):

        config_grid = self.parameter_space.get_configuration_grid()

        self.assertTrue(len(config_grid) == 81)

        self.assertTrue(config_grid[0]["shallow_parameter"] == 1)
        self.assertTrue(config_grid[0]["deep_parameter"]["parameter"] == "a")

        self.assertTrue(config_grid[0]["deep_complex_parameter"]["parameter"] == "e")
        self.assertTrue(config_grid[0]["deep_complex_parameter"]["parameter2"] == 3)

        self.assertTrue(config_grid[80]["shallow_parameter"] == 3)
        self.assertTrue(config_grid[80]["deep_parameter"]["parameter"] == "c")

        self.assertTrue(config_grid[80]["deep_complex_parameter"]["parameter"] == "g")
        self.assertTrue(config_grid[80]["deep_complex_parameter"]["parameter2"] == 1)

    def test_configuration_name(self):
        rfp = self.parameter_space.configuration_name({
                "shallow_parameter": 1,
                "deep_parameter/parameter": "b",
                "deep_complex_parameter/parameter": "e",
                "deep_complex_parameter/parameter2": 2
            })

        self.assertTrue(rfp == os.path.join('-deep_complex_parameter@parameter@e-deep_complex_parameter@parameter2@2-deep_parameter@parameter@b-shallow_parameter@1'))
