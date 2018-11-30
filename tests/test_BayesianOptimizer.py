import unittest
import os
import json
import numpy as np
import scipy
from BayesianOptimizer import BayesianOptimizer



class test_BayesianOptimizer(unittest.TestCase):

    def setUp(self):

        self.test_probe_set = [
            {
                "result": 3,
                "parameters": {
                    "continuous": 3,
                    "discrete": 7,
                    "nominal": "c"
                }
            },
            {
                "result": 1,
                "parameters": {
                    "continuous": 3,
                    "discrete": 7,
                    "nominal": "c"
                }
            }
        ]

        bound_specification = {
            "continuous": [3, None, 9],
            "discrete": [4, 7, 8],
            "nominal": ["a", "c", "b"]
        }

        self.probe_trace = []

        def trace_probe(params):
            self.probe_trace.append(params)
            return params["continuous"]

        self.config = {
            "target_function": trace_probe,
            "random_count": 3,
            "bound_specification": bound_specification,
            "state_file_path": os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources/bayesian_optimizer/test_state.json'),
            "guess_metric": "ei",
            "exploration_rate": 1.0,
            "guess_count": 15,
        }

        self.tearDown()
        self.bo = BayesianOptimizer(self.config)

    def tearDown(self):
        if os.path.isfile(self.config["state_file_path"]):
            os.remove(self.config["state_file_path"])

        self.bo = None

    def test_probe(self):
        params = {
            "continuous": 8,
            "discrete": 8,
            "nominal": "a"
        }

        self.bo.probe(params)

        self.assertTrue(self.bo.probe_set == [{"result": 8, "parameters": params}])

        with open(self.bo.state_file_path, 'r') as f:
            state = json.load(f)

        self.assertTrue(self.bo.probe_set == state["probe_set"])

        params2 = {
            "continuous": 7,
            "discrete": 7,
            "nominal": "b"
        }

        self.bo.probe(params2)

        self.assertTrue(self.bo.probe_set == [{"result": 8, "parameters": params}, {"result": 7, "parameters": params2}])

        with open(self.bo.state_file_path, 'r') as f:
            state = json.load(f)

        self.assertTrue(self.bo.probe_set == state["probe_set"])

        self.assertTrue(self.probe_trace == [params, params2])

    def test_save_restore_state(self):

        self.assertTrue(not os.path.isfile(self.bo.state_file_path))

        self.assertTrue(self.bo.probe_set == [])
        self.assertTrue(self.bo.random_count == 3)

        self.bo.probe_set = self.test_probe_set
        self.bo.save_state()
        self.assertTrue(os.path.isfile(self.bo.state_file_path))

        self.bo.random_count = 2
        self.bo.probe_set = []
        self.bo.restore_state()
        self.assertTrue(self.bo.probe_set == self.test_probe_set)
        self.assertTrue(self.bo.random_count == 3)

    def test_search(self):
        # strongly negative exploration to speed things up
        self.bo.exploration_rate = -10
        best_params = self.bo.search_best_parameters(2)
        self.assertTrue(self.bo.random_count == 1)
        best_params = self.bo.search_best_parameters(3)
        self.assertTrue(self.bo.random_count == 0)
        best_params = self.bo.search_best_parameters(4)
        self.assertTrue(self.bo.random_count == 0)
        best_params = self.bo.search_best_parameters(20)


        self.assertTrue(np.abs(best_params["continuous"] - 9.0) < .1)

    def test_guess_best_parameters(self):
        self.bo.probe_set = self.mock_quadratic_probe_results_centered_halfway()

        # heavily punish exploration to make sure it goes for the optimum
        self.bo.exploration_rate = -100

        best_params = self.bo.guess_best_parameters(self.bo.update_metric())

        as_real = self.bo.codec.to_real(best_params)

        self.assertTrue(np.all(np.abs(as_real[[0, 1]] - .5)) < .001)

    def mock_quadratic_probe_results_centered_halfway(self):

        probe_set = []

        real_nominal_list = [
            self.bo.codec.parameter_codecs["nominal"].to_real("a"),
            self.bo.codec.parameter_codecs["nominal"].to_real("b"),
            self.bo.codec.parameter_codecs["nominal"].to_real("c")]

        real_continuous_list = [i / 100 for i in range(101) ]
        real_discrete_list = [0, .5, 1]

        for x in real_continuous_list:
            for y in real_discrete_list:
                for z in real_nominal_list:

                    reals = np.concatenate([np.array([x]), np.array([y]), z])

                    probe_set.append({
                        "result": -(np.square(x - .5) + np.square(y - .5)),
                        "parameters": self.bo.codec.to_param(reals)
                    })

        return probe_set

    def test_expected_improvement(self):

        mean = 6
        standard_deviation = 2
        best_known = 9
        exploration_rate = 1

        expected_improvement = self.bo.expected_improvement(mean, standard_deviation, best_known, exploration_rate)

        sample_size = 100000
        random_points_in_distribution = np.random.normal(mean, standard_deviation, size=sample_size)

        true_mean_improvement = np.sum(
            random_points_in_distribution[random_points_in_distribution > (best_known + exploration_rate)] - (best_known + exploration_rate)) / sample_size

        self.assertTrue(np.abs(true_mean_improvement - expected_improvement) < .01)

        real_params_centroid = np.array([.5, .75, 0, 0])

        self.bo.probe_set = self.mock_normally_distributed_results_for_real_params(real_params_centroid, 1000)
        self.bo.guess_metric = 'ei'

        ei_metric = self.bo.update_metric()

        ei_val = ei_metric(real_params_centroid)

        m, s = self.bo.gaussian_process.predict(np.reshape(real_params_centroid, [1, -1]), return_std=True)

        correct_metric_score = (self.bo.expected_improvement(m, s, self.bo.get_best_known()["result"], self.bo.exploration_rate))

        # note that the dataset has a standard deviation that is close to zero so no ei is to be expected
        self.assertTrue(np.abs(ei_val - correct_metric_score) < 0.1)

    def test_probability_of_improvement(self):

        mean = 6
        standard_deviation = 2
        best_known = 9
        exploration_rate = -1

        poi = self.bo.probability_of_improvement(mean, standard_deviation, best_known, exploration_rate)

        sample_size = 100000
        random_points_in_distribution = np.random.normal(mean, standard_deviation, size=sample_size)

        percent_better_than_er_corrected_best_known = np.sum(random_points_in_distribution >= (best_known + exploration_rate)) / sample_size

        self.assertTrue(np.abs(percent_better_than_er_corrected_best_known - poi) < .01)

        real_params_centroid = np.array([.5, .75, 0, 0])

        self.bo.probe_set = self.mock_normally_distributed_results_for_real_params(real_params_centroid, 1000)
        self.bo.guess_metric = 'poi'

        poi_metric = self.bo.update_metric()

        poi_val = poi_metric(real_params_centroid)

        m, s = self.bo.gaussian_process.predict(np.reshape(real_params_centroid, [1, -1]), return_std=True)

        correct_metric_score = (self.bo.probability_of_improvement(m, s, self.bo.get_best_known()["result"], self.bo.exploration_rate))

        # note that the dataset has a standard deviation that is close to zero so no poi is to be expected
        self.assertTrue(np.abs(poi_val - correct_metric_score) < 0.1)


    def test_upper_confidence_bound(self):
        ucb = self.bo.upper_confidence_bound(1, 2, 3)
        self.assertTrue(ucb == 7)

        real_params_centroid = np.array([.5, .75, 0, 0])

        self.bo.probe_set = self.mock_normally_distributed_results_for_real_params(real_params_centroid, 1000)
        self.bo.guess_metric = 'ucb'

        ucb_metric = self.bo.update_metric()

        ucb_val = ucb_metric(real_params_centroid)

        m, s = self.bo.gaussian_process.predict(np.reshape(real_params_centroid, [1, -1]), return_std=True)

        correct_metric_score = (self.bo.upper_confidence_bound(m, s, self.bo.exploration_rate))

        self.assertTrue(np.abs(ucb_val - correct_metric_score) < 0.1)
        raise(Exception("For some reason the gaussian process does not predict a mean of 0 with std of 1, even though I fed it these values. " +
                        "Why? GP code matches the repo that seemed to most thoroughly tackle bayesian optimization" +
                        "Other repos also exhibit this issue"))

    def mock_normally_distributed_results_for_real_params(self, real_params, sample_size):
        params = self.bo.codec.to_param(real_params)

        probe_set = [
            {
                "result": np.random.normal(),
                "parameters": {
                    **params,
                    "continuous": np.clip(params["continuous"] + np.random.normal() / 1000.0, 3.0, 9.0)
                }
            } for i in range(sample_size)]

        return probe_set

    def test__real_probe_set(self):
        self.bo.probe_set = self.test_probe_set

        (X, y) = self.bo._real_probe_set()

        self.assertTrue(X.shape == (2, 4))
        self.assertTrue(y.shape == (2,))

    def test_explore_random_parameters(self):
        self.bo.explore_random_parameters(50)

        self.assertTrue(len(self.bo.probe_set) == 50)

        result_y = list(map(lambda v: v["result"], self.bo.probe_set))
        result_y.sort()
        self.assertTrue(scipy.stats.pearsonr(result_y, range(len(result_y)))[1] < .05)

    def test_retrain_gaussian_process_model(self):
        self.bo.retrain_gaussian_process_model()
        self.assertTrue(True)  # can only test if it does not crash
        self.bo.probe_set = self.test_probe_set
        self.bo.retrain_gaussian_process_model()
        self.assertTrue(True)  # can only test if it does not crash

    def test_get_best_known(self):
        self.bo.probe_set = self.test_probe_set

        best_known = self.bo.get_best_known()

        self.assertTrue(best_known == {
                "result": 3,
                "parameters": {
                    "continuous": 3,
                    "discrete": 7,
                    "nominal": "c"
                }
            })