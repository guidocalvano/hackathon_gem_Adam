from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from ParameterCodec import ParameterCodec
from scipy.stats import norm as normal_distribution
from scipy.optimize import minimize
import os
import json


class BayesianOptimizer:

    def __init__(self, config):

        self.target_function = config["target_function"]
        self.bound_specification = config["bound_specification"]
        self.codec = ParameterCodec(self.bound_specification)
        self.probe_set = []
        self.guess_metric = config["guess_metric"] if "guess_metric" in config else "ei"

        guess_metric_default_explorations = {
            "ei": 0.0,
            "poi": 0.0,
            "ucb": 2.576
        }
        self.exploration_rate = config["exploration_rate"] \
            if "exploration_rate" in config \
            else guess_metric_default_explorations[self.guess_metric]

        self.state_file_path = config["state_file_path"]
        self.random_count = config["random_count"] if "random_count" in config else 5

        self.guess_count = config["guess_count"] if "guess_count" in config else 250

        self.restore_state()

    def restore_state(self):
        if os.path.isfile(self.state_file_path):
            with open(self.state_file_path, 'r') as f:
                state = json.load(f)
            self.random_count = state["random_count"]
            self.probe_set = state["probe_set"]

    def save_state(self):
        os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)

        with open(self.state_file_path, 'w') as f:
            json.dump({
                "probe_set": self.probe_set,
                "random_count": self.random_count
            }, f)

    def retrain_gaussian_process_model(self):

        self.gaussian_process = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25
        )

        if len(self.probe_set) == 0:
            return

        real_probe_set = self._real_probe_set()

        self.gaussian_process.fit(real_probe_set[0], real_probe_set[1])

    def probe(self, parameters):
        probed_value = self.target_function(parameters)

        self.probe_set.append({
            "result": probed_value,
            "parameters": parameters
        })

        self.save_state()

        return probed_value

    # OPTIMIZER FINDS THE ***OPTIMUM*** SO THE HIGHEST VALUE
    def search_best_parameters(self, probe_limit):

        if(len(self.probe_set) == 0):
            self.random_probe()
            if self.random_count < 0: self.random_count = 0

        while self.random_count > 0 and len(self.probe_set) < probe_limit:
            self.random_probe()

        while len(self.probe_set) < probe_limit:
            metric = self.update_metric()
            best_guess_parameters = self.guess_best_parameters(metric)

            self.probe(best_guess_parameters)

        return self.get_best_known()["parameters"]

    def random_probe(self):
        self.random_count = self.random_count - 1
        self.probe(self.codec.random_param())

    def get_best_known(self):
        best_known = {
            "result": -float("inf"),
            "parameters": None
        }

        for i in range(len(self.probe_set)):
            if self.probe_set[i]["result"] >= best_known["result"]:
                best_known = self.probe_set[i]

        return best_known

    def update_metric(self):
        self.retrain_gaussian_process_model()

        best_known = self.get_best_known()["result"]

        if self.guess_metric == "ei":

            def ei(real_params):
                mean, standard_deviation = self.gaussian_process.predict(np.reshape(real_params, [1, -1]), return_std=True)

                return self.expected_improvement(mean, standard_deviation, best_known, self.exploration_rate)

            return ei

        if self.guess_metric == "ucb":
            def ucb(real_params):
                mean, standard_deviation = self.gaussian_process.predict(np.reshape(real_params, [1, -1]), return_std=True)

                # NOTE THAT UPPER CONFIDENCE BOUND HAS ONE LESS ARGUMENT
                return self.upper_confidence_bound(mean, standard_deviation, self.exploration_rate)

            return ucb

        if self.guess_metric == "poi":
            def poi(real_params):
                mean, standard_deviation = self.gaussian_process.predict(np.reshape(real_params, [1, -1]), return_std=True)

                return self.probability_of_improvement(mean, standard_deviation, best_known, self.exploration_rate)

            return poi

    def guess_best_parameters(self, real_metric):

        guess_starting_parameters = [self.codec.random_param() for i in range(self.guess_count)]

        best_parameters = None
        best_guess = -float("inf")
        for start_parameter in guess_starting_parameters:
            result = minimize(lambda vals: -real_metric(vals), self.codec.to_real(start_parameter), bounds=self.codec.real_bounds(), method="L-BFGS-B")
            if not result.success:
                continue

            if -result.fun[0] >= best_guess:
                best_guess = -result.fun[0]
                best_parameters = self.codec.to_param(result.x)

        return best_parameters

    def probability_of_improvement(self, mean, standard_deviation, best_known, exploration_rate_poi):

        # The exploration rate punishes rating with lower standard deviations more
        # than ratings with high standard deviations, thus it creates
        # a preference for higher standard deviations
        z = (mean - best_known - exploration_rate_poi) / standard_deviation

        return normal_distribution.cdf(z)

    def upper_confidence_bound(self, mean, standard_deviation, exploration_rate_ucb):

        return mean + exploration_rate_ucb * standard_deviation

    def expected_improvement(self, mean, standard_deviation, best_known, exploration_rate_ei):

        z = (mean - best_known - exploration_rate_ei) / standard_deviation

        return standard_deviation * (z * normal_distribution.cdf(z) + normal_distribution.pdf(z))

    def explore_random_parameters(self, count):
        for i in range(count):
            self.probe(self.codec.random_param())

    def _real_probe_set(self):
        real_y = np.array(list(map(lambda v: v["result"], self.probe_set)))
        real_X_list = np.vstack(map( lambda v: self.codec.to_real(v["parameters"]), self.probe_set))

        return (real_X_list, real_y)

