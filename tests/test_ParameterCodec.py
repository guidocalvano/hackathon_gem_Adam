import unittest
import numpy as np
from ParameterCodec import ParameterCodec
import scipy.stats
from collections import Counter


class test_ParameterCodec(unittest.TestCase):

    def setUp(self):
        self.parameterCodec = ParameterCodec({
            "continuous": [3, None, 9],
            "discrete": [4,7,8],
            "nominal": ["a", "c", "b"]
        })

    def tearDown(self):
        self.parameterCodec = None

    def test_to_real_to_params(self):
        lower_bound_params = {
            "continuous": 3,
            "discrete": 4,
            "nominal": "a"
        }

        lower_bound_real = self.parameterCodec.to_real(lower_bound_params)
        self.assertTrue(np.all(lower_bound_real == 0))
        self.assertTrue(self.parameterCodec.to_param(lower_bound_real) == lower_bound_params)

        upper_bound_params = {
            "continuous": 9,
            "discrete": 8,
            "nominal": "b"
        }

        upper_bound_real = self.parameterCodec.to_real(upper_bound_params)
        self.assertTrue(np.all(upper_bound_real == np.array([1, 1, 1, 0])))  # last values of 1 0 due to nominal mapping
        self.assertTrue(self.parameterCodec.to_param(upper_bound_real) == upper_bound_params)

        middle_params = {
            "continuous": 6,
            "discrete": 7,
            "nominal": "c"
        }

        middle_real = self.parameterCodec.to_real(middle_params)
        self.assertTrue(np.all(np.abs(middle_real - np.array([.5, .5, .5, 0.866])) < .001))  # last values of .5 .866 due to nominal mapping
        self.assertTrue(self.parameterCodec.to_param(middle_real) == middle_params)

    def test_random_params(self):
        sample_size = 1000
        many_random_params = [self.parameterCodec.random_param() for i in range(sample_size)]

        continous_vals = np.array(list(map(lambda p: p["continuous"], many_random_params)))
        continous_vals.sort()
        continuous_is_not_random = scipy.stats.pearsonr(continous_vals, np.arange(sample_size))[1]
        nominal_frequencies = np.array(list(Counter(map(lambda p: p["nominal"], many_random_params)).values()))
        discrete_frequencies = np.array(list(Counter(map(lambda p: p["discrete"], many_random_params)).values()))

        self.assertTrue(continuous_is_not_random < .05)  # refute the null hypothesis ^^
        self.assertTrue(np.all(nominal_frequencies > sample_size / 4))
        self.assertTrue(np.all(discrete_frequencies > sample_size / 4))

        many_random_params_from_reals = [self.parameterCodec.to_param(self.parameterCodec.random_real()) for i in range(sample_size)]

        continous_vals = np.array(list(map(lambda p: p["continuous"], many_random_params_from_reals)))
        continous_vals.sort()
        continuous_is_not_random = scipy.stats.pearsonr(continous_vals, np.arange(sample_size))[1]
        nominal_frequencies = np.array(list(Counter(map(lambda p: p["nominal"], many_random_params_from_reals)).values()))
        discrete_frequencies = np.array(list(Counter(map(lambda p: p["discrete"], many_random_params_from_reals)).values()))

        self.assertTrue(continuous_is_not_random < .05)  # refute the null hypothesis ^^
        self.assertTrue(np.all(nominal_frequencies > sample_size / 4))
        self.assertTrue(np.all(discrete_frequencies > sample_size / 4))



    def test_real_size(self):
        self.assertTrue(self.parameterCodec.real_size() == 4)

    def test_real_bounds(self):
        self.assertTrue(np.all(self.parameterCodec.real_bounds() == np.array([[0, 1], [0, 1], [0, 1], [0, 1]])))