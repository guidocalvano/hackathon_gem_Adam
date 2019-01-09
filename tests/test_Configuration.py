import unittest
import os
import shutil
from Configuration import Configuration

class TestConfiguration(unittest.TestCase):

    def setUp(self):
        self.base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'configuration')
        self.temp_path = os.path.join(self.base_path, 'temp')

        self.tearDown()
        os.makedirs(self.temp_path)


        self.target_config = os.path.join(self.temp_path, 'test_config.json')

        shutil.copyfile(os.path.join(self.base_path, 'test_config.json'), self.target_config)

    def tearDown(self):
        if os.path.isdir(self.temp_path):
            shutil.rmtree(self.temp_path)


    def test_vanilla_load(self):
        config = Configuration.load(self.target_config)

        self.assertTrue(config.config == {
          "a": 1,
          "b": {"c": 2},
          "d": "iets"
        }, 'vanilla load of config must result in correct configuration')

    def test_override_shallow_var(self):
        config = Configuration.load(self.target_config)

        config.setPath("a", 2)

        self.assertTrue(config.config == {
          "a": 2,
          "b": {"c": 2},
          "d": "iets"
        }, 'shallow int assignment must result in correct configuration')

        config.setPath("d", "nogiets")

        self.assertTrue(config.config == {
          "a": 2,
          "b": {"c": 2},
          "d": "nogiets"
        }, 'shallow string assignment must result in correct configuration')

    def test_override_deep_var(self):
        config = Configuration.load(self.target_config)

        config.setPath("b/c", 4)

        self.assertTrue(config.config == {
          "a": 1,
          "b": {"c": 4},
          "d": "iets"
        }, 'deep assignment must result in correct configuration')


    def test_parse_shallow_arg(self):
        config = Configuration.load(self.target_config)
        config_name = config.parseArguments(["--a", "4"])

        self.assertTrue(config.config == {
          "a": 4,
          "b": {"c": 2},
          "d": "iets"
        }, 'shallow arg must be assigned correctly')

        self.assertTrue(config_name == '--a#4')

    def test_parse_deep_arg(self):
        config = Configuration.load(self.target_config)
        config_name = config.parseArguments(["--b/c", "5"])

        self.assertTrue(config.config == {
          "a": 1,
          "b": {"c": 5},
          "d": "iets"
        }, 'deep arg must be assigned correctly')

        self.assertTrue(config_name == '--b/c#5')

    def test_parse_multiple_args(self):
        config = Configuration.load(self.target_config)
        config_name = config.parseArguments(["--a", "4", "--b/c", "5", "--d", "asdf"])

        self.assertTrue(config.config == {
          "a": 4,
          "b": {"c": 5},
          "d": "asdf"
        }, 'multiple args must be assigned correctly')

        self.assertTrue(config_name == '--a#4#--b/c#5#--d#asdf')
