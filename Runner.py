from ParameterSpace import ParameterSpace
from Analysis import Analysis
import os
import os.path
import sys
from BayesianOptimizer import BayesianOptimizer
from bayes_opt.util import load_logs
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
import json


class Runner:

    def __init__(self, argv, learning_dictionary):

        self.argv = argv

        self.parameter_space_file_path = self.argv[1]
        self.learning_dictionary = learning_dictionary

        self.parameter_space = ParameterSpace.load(self.parameter_space_file_path)

        self.parameter_space_path = os.path.splitext(self.parameter_space_file_path)[0]

        self.results_log_file_path = os.path.join(self.parameter_space_path, 'result_log.json')

        os.makedirs(self.parameter_space_path, exist_ok=True)

        self.flattened_parameter_space = self.parameter_space.flatten_parameters()

        target_function_key = self.parameter_space.config["target_function"]

        this = self
        def target_function(params):
            return this.generate_result(this.learning_dictionary[target_function_key], params)

        self.bayesian_optimizer = BayesianOptimizer({
                "target_function": target_function,
                "bound_specification": self.flattened_parameter_space,
                "state_file_path": self.results_log_file_path
            }
        )

    def make_result_file_path(self, params):

        return os.path.join(self.parameter_space_path, self.parameter_space.configuration_name(params))

    def make_tensorboard_path(self, params):

        return os.path.join(self.parameter_space_path, 'tensorboard', self.parameter_space.configuration_name(params))

    def generate_result(self, target_function, params):

        result_path = self.make_result_file_path(params)
        tensorboard_path = self.make_tensorboard_path(params)

        os.makedirs(result_path, exist_ok=True)

        config = self.parameter_space.apply_instantiation(params)

        performance, raw_analysis = target_function({**config, "result_path": result_path, "tensorboard_path": tensorboard_path})

        Analysis.store_raw_result(result_path, raw_analysis)

        return performance

    def arguments_to_flattened_params(self, args):
        assert(len(args) % 2 == 0)
        param_count = int(len(args) / 2)

        flattened_params = {}

        for i in range(param_count):
            offset = i * 2

            key = args[offset][2:]
            value = args[offset + 1]

            flattened_params[key] = int(value)

        return flattened_params

    def run_probe(self):
        params = self.arguments_to_flattened_params(self.argv[2:])

        self.bayesian_optimizer.probe(params)

    def run_search_params(self):
        self.bayesian_optimizer.search_best_parameters(int(self.argv[2]))

    def run(self):

        if len(self.argv) > 3:  # this is a probe command

            self.run_probe()
            return

        self.run_search_params()
