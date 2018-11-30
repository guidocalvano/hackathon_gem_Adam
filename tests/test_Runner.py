import unittest
import os
import json
from Runner import Runner
from Learning import Learning
from DataImporter import DataImporter
import shutil
from ParameterSpace import ParameterSpace
from BayesianOptimizer import BayesianOptimizer


class test_Runner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.unstandardized_photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'runner', 'unstandardized_photos')
        cls.photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'runner', 'photos')
        cls.test_photos_csv_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'runner', 'test_photos.csv')

        cls.dataImporter = DataImporter()

        cls.dataImporter.convert_to_standard_resolution(
            cls.unstandardized_photos_path,
            cls.photos_path,
            [152, 114],
            {
                "multi_core_count": 2,
                "timeout_secs": 100,
                "chunk_size": 2
            }
        )

        cls.data_set = cls.dataImporter.import_all_data(cls.test_photos_csv_file_path, cls.photos_path)

        data_set = test_Runner.dataImporter.import_all_data(cls.test_photos_csv_file_path, cls.photos_path)

        cls.mock_results = Learning.simple_binary_classification(data_set, epochs=1, batch_size=2)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):

        # mock parameter space
        self.parameter_space_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'runner', 'test_parameter_space.json')
        self.result_store_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'runner', 'test_parameter_space')
        self.parameter_space_result_path = os.path.splitext(self.parameter_space_file_path)[0]


        self.tearDown()

        # mock argvs

        self.argv_search = ["theprogram", self.parameter_space_file_path, "2"]

        self.argv_probe = ["theprogram", self.parameter_space_file_path,
                                    "--shallow_parameter", "1",
                                    "--deep_parameter/parameter", "4",
                                    "--deep_complex_parameter/parameter", "7",
                                    "--deep_complex_parameter/parameter2", "10"]

        # mock learning_dictionary

        self.param_trace = []

        trace_counter = 0
        # constructing like this will capture self in closure
        def trace_params(params):
            self.param_trace.append(params)
            nonlocal trace_counter
            trace_counter = trace_counter + 1
            return trace_counter, test_Runner.mock_results

        self.trace_params = trace_params

        self.learning_dict = {
            "test_learning": lambda params: trace_params(params)
        }

        # runner

        self.run_probe = Runner(self.argv_probe, self.learning_dict)
        self.run_search = Runner(self.argv_search, self.learning_dict)

    def tearDown(self):

        # if os.path.isfile(self.result_store_file_path):
        #     os.remove(self.result_store_file_path)

        if os.path.isdir(self.parameter_space_result_path):
            shutil.rmtree(self.parameter_space_result_path)

    def test_init(self):
        # are initial parameters configured correctly? Assume all initial parameters are tested by the other tests
        # except init_points

        self.assertTrue(self.run_probe.argv == self.argv_probe)
        self.assertTrue(self.run_probe.parameter_space_file_path == self.parameter_space_file_path)
        self.assertTrue((self.run_probe is not None) and isinstance(self.run_probe.parameter_space, ParameterSpace))
        self.assertTrue(isinstance(self.run_probe.results_log_file_path, str))
        self.assertTrue(self.run_probe.flattened_parameter_space == {'shallow_parameter': [1, 2, 3], 'deep_complex_parameter/parameter': [7, 8, 9], 'deep_complex_parameter/parameter2': [10, 11, 12], 'deep_parameter/parameter': [4, 5, 6]})
        self.assertTrue(isinstance(self.run_probe.bayesian_optimizer, BayesianOptimizer))

    def test_run_search(self):
        # are results tracked?
        # does it run the right number of iterations?
        # does pause continue work?

        parameter_space_path = os.path.splitext(self.parameter_space_file_path)[0]

        results_log_file_path = os.path.join(parameter_space_path, 'result_log.json')

        self.assertTrue(not os.path.isfile(results_log_file_path))

        self.run_search.run()
        self.assertTrue(os.path.isfile(results_log_file_path))

        with open(results_log_file_path) as f:
            probe_data = json.load(f)

        self.assertTrue(probe_data["random_count"] == 3)
        self.assertTrue(len(probe_data["probe_set"]) == 2)

        argv_search2 = ["theprogram", self.parameter_space_file_path, "6"]

        self.run_search = Runner(argv_search2, self.learning_dict)
        self.run_search.run()

        with open(results_log_file_path) as f:
            probe_data = json.load(f)

        self.assertTrue(probe_data["random_count"] == 0)
        self.assertTrue(len(probe_data["probe_set"]) == 6)

    def test_run_probe(self):
        # does a probe correctly instantiate parameter space?
        rfp = self.run_probe.make_result_file_path({
                "shallow_parameter": 1,
                "deep_parameter/parameter": 4,
                "deep_complex_parameter/parameter": 7,
                "deep_complex_parameter/parameter2": 10
            })

        self.run_probe.run()

        model_file_path = os.path.join(rfp, 'raw', 'model.h5')
        stats_file_path = os.path.join(rfp, 'raw', 'stats.json')

        # does probing lead to stored results?

        self.assertTrue(os.path.isfile(model_file_path))
        self.assertTrue(os.path.isfile(stats_file_path))

        self.assertTrue(self.param_trace[0]["shallow_parameter"] == 1)
        self.assertTrue(self.param_trace[0]["deep_parameter"]["parameter"] == 4)
        self.assertTrue(self.param_trace[0]["deep_complex_parameter"]["parameter"] == 7)
        self.assertTrue(self.param_trace[0]["deep_complex_parameter"]["parameter2"] == 10)

    def test_make_result_file_path(self):
        rfp = self.run_probe.make_result_file_path({
                "shallow_parameter": 1,
                "deep_parameter/parameter": "b",
                "deep_complex_parameter/parameter": "e",
                "deep_complex_parameter/parameter2": 2
            })

        self.assertTrue(rfp == os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources/runner/test_parameter_space/_deep_complex_parameter_parameter_e_deep_complex_parameter_parameter2_2_deep_parameter_parameter_b_shallow_parameter_1'))

    def test_generate_result(self):

        rfp = self.run_probe.make_result_file_path({
                "shallow_parameter": 1,
                "deep_parameter/parameter": 4,
                "deep_complex_parameter/parameter": 7,
                "deep_complex_parameter/parameter2": 10
            })


        # is a result stored correclty?
        self.run_probe.generate_result(self.trace_params, {
                "shallow_parameter": 1,
                "deep_parameter/parameter": 4,
                "deep_complex_parameter/parameter": 7,
                "deep_complex_parameter/parameter2": 10
            })

        model_file_path = os.path.join(rfp, 'raw', 'model.h5')
        stats_file_path = os.path.join(rfp, 'raw', 'stats.json')

        self.assertTrue(os.path.isfile(model_file_path))
        self.assertTrue(os.path.isfile(stats_file_path))

        self.assertTrue(self.param_trace[0]["shallow_parameter"] == 1)
        self.assertTrue(self.param_trace[0]["deep_parameter"]["parameter"] == 4)
        self.assertTrue(self.param_trace[0]["deep_complex_parameter"]["parameter"] == 7)
        self.assertTrue(self.param_trace[0]["deep_complex_parameter"]["parameter2"] == 10)

