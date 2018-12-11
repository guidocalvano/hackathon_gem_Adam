import unittest
from Learning import Learning
import numpy as np
import keras
from keras.layers import Input
from keras import backend as K
from scipy.stats import pearsonr
import datetime
import matplotlib.pyplot as plt
import gc
import os
import shutil
import json
from ParameterSpace import ParameterSpace
from run_data_parameter_import import run as run_data_import
from DataImporter import DataImporter


class test_Learning(unittest.TestCase):
    def setUp(self):
        self.parameter_space_file_path = './resources/learning/parameter_space.json'
        self.parameter_space_float_file_path = './resources/learning/parameter_space_float.json'
        self.parameter_space_bool_file_path = './resources/learning/parameter_space_bool.json'

        self.tensorboard_path = './resources/learning/tensorboard'

    def tearDown(self):

        if os.path.isdir(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)

    def test_resnet_convolution(self):
        # is the output shape correct?
        # are arguments passed?

        input = Input(shape=[5, 6, 7], tensor=K.variable(np.random.random([1, 5, 6, 7])))
        weight_regularizer = keras.regularizers.l1_l2(l1=.5, l2=.75)

        layer = Learning.resnet_convolution(input, 2, 2, regularizer=weight_regularizer)

        self.assertTrue(layer.shape[1:].as_list() == [3, 3, 14])

        layer_output = layer.eval(session=K.get_session())

        self.assertTrue(layer_output.shape == (1, 3, 3, 14))

        # self.assertTrue(cost == (3 * 3 * np.prod(layer.shape[1:].as_list()) * input.shape[-1].value ))

        # does the cost correlate with measured times?
        #
        # estimates = [Learning.resnet_convolution(Input(shape=[30, 30, 30], tensor=K.variable(np.random.random([200, 30, 30, 30]))), np.random.randint(1, 4), np.random.randint(1, 4), regularizer=None) for i in range(10) ]
        #
        # self.assert_correct_execution_cost(estimates)

    def test_initial_convolution(self):
        # is the output shape correct?
        # are arguments passed?

        input = Input(shape=[5, 6, 7], tensor=K.variable(np.random.random([1, 5, 6, 7])))
        weight_regularizer = keras.regularizers.l1_l2(l1=.5, l2=.75)

        layer = Learning.initial_convolution(input, 2, 40, 7, regularizer=weight_regularizer)

        self.assertTrue(layer.shape[1:].as_list() == [3, 3, 40])

        layer_output = layer.eval(session=K.get_session())

        self.assertTrue(layer_output.shape == (1, 3, 3, 40))

    def test_resnet_skip_block(self):
        # is the output shape correct?
        # are arguments passed? (test through last layer)

        input = Input(shape=[5, 6, 7], tensor=K.variable(np.random.random([1, 5, 6, 7])))
        weight_regularizer = keras.regularizers.l1_l2(l1=.5, l2=.75)


        layer = Learning.resnet_skip_block(input,
                                                    2,  # stride
                                                    2,  # filter increase
                                                    3,  # skip distance
                                                    weight_regularizer)

        self.assertTrue(layer.shape.as_list() == [1, 3, 3, 14])

        layer_output = layer.eval(session=K.get_session())

        self.assertTrue(layer_output.shape == (1, 3, 3, 14))

    def test_resnet_output(self):
        # are arguments passed correctly?
        input = Input(shape=[5, 6, 7], tensor=K.variable(np.random.random([1, 5, 6, 7])))
        weight_regularizer = keras.regularizers.l1_l2(l1=.5, l2=.75)

        layer = Learning.resnet_output(input, 10, weight_regularizer)

        self.assertTrue(layer.shape.as_list()[1:] == [10])

        layer_output = layer.eval(session=K.get_session())

        self.assertTrue(layer_output.shape == (1, 10))

        # self.assertTrue(cost == 7 * 10 + 5 * 6 * 7)

        # # does the cost correlate with measured times?
        #
        # estimates = []
        #
        # for i in range(10):
        #     width = np.random.randint(1, 2)
        #     height = np.random.randint(1, 2)
        #     filter_count = width = np.random.randint(20, 1380)
        #     output_count = np.random.randint(1,2)
        #
        #     estimates.append(Learning.resnet_output(Input(shape=[width, height, filter_count], tensor=K.variable(np.random.random([100, width, height, filter_count]))), output_count, regularizer=None))
        #
        # self.assert_correct_execution_cost(estimates)

    # def test_dense(self):
    #
    #     estimates = [Learning.dense(Input(shape=[30], tensor=K.variable(np.random.random([200, 30]))), np.random.randint(1, 60000)) for i in range(10) ]
    #
    #     self.assert_correct_execution_cost(estimates)
    #
    # def test_avg_pooling(self):
    #
    #     estimates = []
    #
    #     for i in range(10):
    #
    #         m = 80
    #         w = np.random.randint(30, m)
    #         h = np.random.randint(30, m)
    #         estimates.append(Learning.avg_pool(Input(shape=[30, w, h, 30], tensor=K.variable(np.random.random([200, w, h, 30])))))
    #
    #     self.assert_correct_execution_cost(estimates)

    def test_build_linear_auto_architecture(self):
        input = Input(shape=[64, 64, 3], tensor=K.variable(np.random.random([1, 64, 64, 3])))

        depth = 6
        skip_distance = 3
        blocks_per_rescale = 2
        stride_difference_per_rescale = 1.5
        filter_increase_factor_per_rescale = 1.2

        regularizer = keras.regularizers.l1_l2(l1=.5, l2=.75)

        layer = Learning.build_linear_auto_architecture(
            input,
            depth,
            blocks_per_rescale,
            skip_distance,
            stride_difference_per_rescale,
            filter_increase_factor_per_rescale,
            regularizer)

        layer_output = layer.eval(session=K.get_session())

        self.assertTrue(layer.shape.as_list()[1:] == [32, 32, 6])

        self.assertTrue(layer_output.shape == (1, 32, 32, 6))

        stride_difference_per_rescale = 2

        layer = Learning.build_linear_auto_architecture(
            input,
            depth,
            blocks_per_rescale,
            skip_distance,
            stride_difference_per_rescale,
            filter_increase_factor_per_rescale,
            regularizer)
        layer_output = layer.eval(session=K.get_session())

        self.assertTrue(layer.shape.as_list()[1:] == [8, 8, 6])

        self.assertTrue(layer_output.shape == (1, 8, 8, 6))

        stride_difference_per_rescale = 1.67
        filter_increase_factor_per_rescale = 3

        layer = Learning.build_linear_auto_architecture(
            input,
            depth,
            blocks_per_rescale,
            skip_distance,
            stride_difference_per_rescale,
            filter_increase_factor_per_rescale,
            regularizer)
        layer_output = layer.eval(session=K.get_session())

        self.assertTrue(layer.shape.as_list()[1:] == [16, 16, 81])

        self.assertTrue(layer_output.shape == (1, 16, 16, 81))

    def test_build_model(self):
        input = Input(shape=[64, 64, 3])

        block_depth = 2
        output_type = 'categorical_int'

        skip_distance = 3
        blocks_per_rescale = 2
        stride_difference_per_rescale = 1.5
        filter_increase_factor_per_rescale = 1.2

        initial_stride = 2
        initial_filter_count = 64
        initial_kernel_size = 7
        l1_factor = .1
        l2_factor = .1

        model = Learning.build_model(input,
                                 4,
                                 block_depth,
                                 output_type,
                                 blocks_per_rescale,
                                 stride_difference_per_rescale,
                                 filter_increase_factor_per_rescale,
                                 skip_distance,
                                 initial_stride,
                                 initial_filter_count,
                                 initial_kernel_size,
                                 l1_factor,
                                 l2_factor)

        # does compiled model actually produce values?

        model.compile(optimizer='adam', loss='mean_squared_error')

        result = model.predict(np.random.random([30] + input.shape.as_list()[1:]))

        self.assertTrue(result.shape == (30, 4))

        self.assertTrue(np.all(np.isfinite(result)))


    def test_get_data_sets(self):
        # is data loaded?
        # is it in the right directory
        run_data_import([None, self.parameter_space_file_path])

        config = ParameterSpace.load(self.parameter_space_file_path).get_configuration_grid()[0]

        ds = Learning.get_data_sets(config)

        self.assertTrue(ds['training'][0].shape == tuple([8] + config['data_import']['output_image_dimensions'] + [3]))
        self.assertTrue(ds['validation'][0].shape == tuple([4] + config['data_import']['output_image_dimensions'] + [3]))
        self.assertTrue(ds['test'][0].shape == tuple([4] + config['data_import']['output_image_dimensions'] + [3]))

        self.assertTrue(np.all(ds['from_network'](ds['to_network'](ds['training'][1])) == ds['training'][1]))

        self.assertTrue(ds['output_type'] == config['learning']['target']['output_type'])

        self.assertTrue(ds['value_names'] == {0: 'w', 2: 'y', 3: 'z', 1: 'x'})

        config = ParameterSpace.load(self.parameter_space_float_file_path).get_configuration_grid()[0]

        ds = Learning.get_data_sets(config)

        self.assertTrue(np.all(ds['from_network'](ds['to_network'](ds['training'][1])) == ds['training'][1]))

        self.assertTrue(ds['output_type'] == config['learning']['target']['output_type'])

        self.assertTrue(ds['value_names'] == {3: 'D', 2: 'C', 0: 'A', 1: 'B'})


        config = ParameterSpace.load(self.parameter_space_bool_file_path).get_configuration_grid()[0]

        ds = Learning.get_data_sets(config)

        self.assertTrue(np.all(ds['from_network'](ds['to_network'](ds['training'][1])) == ds['training'][1]))

        self.assertTrue(ds['output_type'] == config['learning']['target']['output_type'])

        self.assertTrue(ds['value_names'] == ['dirty', 'clean'])

        pass


    def test_train(self):
        # does the code execute?
        # does performance improve for very simply learned input

        batch_size = 8
        epochs = 200
        output_type = 'bool'

        d = keras.layers.Dense(32, input_shape=(784,))
        d2 = keras.layers.Dense(1)

        model = keras.models.Sequential([
            d,
            keras.layers.Activation('relu'),
            d2,
            keras.layers.Activation('sigmoid'),
        ])

        w = d.get_weights()
        w2 = d2.get_weights()

        input = np.random.random([100, 784])
        output = np.sum(input, axis=1, keepdims=True) > .7

        training_set = (
            input,
            output
        )

        validation_set = (
            input,
            output
        )

        results = Learning.train(model, output_type, training_set, validation_set, epochs, batch_size, self.tensorboard_path)

        self.assertTrue(os.path.isdir(self.tensorboard_path))

        tw = d.get_weights()
        tw2 = d2.get_weights()

        self.assertTrue(np.sum(w[0] == tw[0]) < 5000)

        self.assertTrue(np.sum(w2[0] == tw2[0]) < 10)
        self.assertTrue(model == results.model)
        self.assertTrue('acc' in results.history)
        self.assertTrue('lr' in results.history)
        self.assertTrue(len(results.epoch) < 190)

        results = Learning.train(model, 'float', training_set, validation_set, epochs, batch_size, self.tensorboard_path)

        self.assertTrue('mean_squared_error' in results.history)
        self.assertTrue('acc' not in results.history)


        d = keras.layers.Dense(32, input_shape=(784,))
        d2 = keras.layers.Dense(3)

        model = keras.models.Sequential([
            d,
            keras.layers.Activation('relu'),
            d2,
            keras.layers.Activation('sigmoid'),
        ])

        input = np.random.random([100, 784])

        t = np.random.random([784, 3])

        output = np.matmul(input, t)
        output = (output == np.max(output, axis=1, keepdims=True)).astype('int')


        training_set = (
            input,
            output
        )

        validation_set = (
            input,
            output
        )

        results = Learning.train(model, 'categorical_int', training_set, validation_set, epochs, batch_size, self.tensorboard_path)

        self.assertTrue('mean_squared_error' not in results.history)
        self.assertTrue('acc' in results.history)

    def test_run_experiment_categorical_int(self):
        # is the experiment actually run?
        # are results stored in the appropriate directory?
        # is data loaded from the appropriate directory?
        # are parameters assigned correctly?

        run_data_import([None, self.parameter_space_file_path])

        config = ParameterSpace.load(self.parameter_space_file_path).get_configuration_grid()[0]

        raw_performance, stats = Learning.run_experiment({**config, "tensorboard_path": self.tensorboard_path})

        self.assertTrue(stats["stats"]["meta"] == {'type': 'categorical_int', 'labels': {1: 'x', 3: 'z', 2: 'y', 0: 'w'}})
        self.assertTrue(isinstance(stats["model"], keras.models.Model))

    def test_run_experiment_bool(self):

        run_data_import([None, self.parameter_space_bool_file_path])

        config = ParameterSpace.load(self.parameter_space_bool_file_path).get_configuration_grid()[0]

        raw_performance, stats = Learning.run_experiment({**config, "tensorboard_path": self.tensorboard_path})

        self.assertTrue(stats["stats"]["meta"] == {'type': 'bool', 'labels': ["dirty", "clean"]})
        self.assertTrue(isinstance(stats["model"], keras.models.Model))

    def test_run_experiment_float(self):

        run_data_import([None, self.parameter_space_float_file_path])

        config = ParameterSpace.load(self.parameter_space_float_file_path).get_configuration_grid()[0]

        raw_performance, stats = Learning.run_experiment({**config, "tensorboard_path": self.tensorboard_path})
        self.assertTrue(stats["stats"]["meta"] == {'type': 'float', 'labels': {0: "A", 1: "B", 2: "C", 3: "D"}})
        self.assertTrue(isinstance(stats["model"], keras.models.Model))

    def assert_correct_execution_cost(self, estimates):

        execution_times = np.array([self.time_execution(est[0]) for est in estimates])
        predicted_costs = np.array([est[1] for est in estimates])

        filtered_indices = execution_times != np.max(execution_times)

        execution_times = execution_times[filtered_indices]
        predicted_costs = predicted_costs[filtered_indices]

        r, p = pearsonr(execution_times, predicted_costs)

        # debugging visualization
        indices = np.argsort(predicted_costs)
        sorted_predicted_costs = predicted_costs[indices]
        sorted_execution_times = execution_times[indices]
        plt.plot(sorted_predicted_costs, sorted_execution_times)
        plt.xlabel('predicted')
        plt.ylabel('actual')
        plt.show()
        # assert null hypothesis can be rejected
        self.assertTrue(p < .05)

    def time_execution(self, layer):
        gc.collect()
        gc.disable()
        a = datetime.datetime.now()

        layer.eval(session=K.get_session())

        b = datetime.datetime.now()
        time_cost = b - a
        gc.enable()
        gc.collect()


        return time_cost.microseconds