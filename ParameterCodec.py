from Math import Math
import numpy as np


class ParameterCodec:

    def __init__(self, parameter_bounds):
        self.parameter_order = list(sorted(parameter_bounds.keys()))

        def build_codec(bounds_specs):
            if bounds_specs[1] is None:
                return RealParamCodec(bounds_specs)

            if isinstance(bounds_specs[0], str) or isinstance(bounds_specs[0], list):
                return NominalParamCodec(bounds_specs)

            return NumberParamCodec(bounds_specs)

        self.parameter_codecs =dict(zip(self.parameter_order, map(lambda name: build_codec(parameter_bounds[name]), self.parameter_order)))

    def to_real(self, params):

        real_list = []

        for name in self.parameter_order:
            real_list.append(self.parameter_codecs[name].to_real(params[name]))

        return np.concatenate(real_list)

    def to_param(self, reals):
        remaining_reals = reals
        as_param = {}
        for name in self.parameter_order:
            codec = self.parameter_codecs[name]
            target_reals = remaining_reals[:codec.real_size()]
            as_param[name] = codec.to_param(target_reals)

            remaining_reals = remaining_reals[codec.real_size():]

        return as_param

    def random_param(self):
        as_param = {}
        for name in self.parameter_order:
            codec = self.parameter_codecs[name]
            as_param[name] = codec.random_param()

        return as_param

    def random_real(self):
        reals = []
        for name in self.parameter_order:
            codec = self.parameter_codecs[name]
            reals.append(codec.random_real())

        return np.concatenate(reals)

    def real_size(self):
        known_size = 0
        for name in self.parameter_order:
            codec = self.parameter_codecs[name]

            known_size += codec.real_size()

        return known_size

    def real_bounds(self):
        return np.concatenate([np.zeros([self.real_size(), 1]), np.ones([self.real_size(), 1])], axis=1)

class RealParamCodec:
    def __init__(self, bounds):
        self.min = bounds[0]
        self.max = bounds[2]

        self.range = self.max - self.min

    def to_param(self, r):
        return self.min + r[0] * self.range

    def to_real(self, p):
        return np.array([(p - self.min) / self.range])

    def random_param(self):
        return self.min + self.random_real()[0] * self.range

    def random_real(self):
        return np.random.random([1])

    def real_size(self):
        return 1

class NumberParamCodec:
    def __init__(self, bounds):
        self.accepted_values = np.array(sorted(bounds))

    def to_param(self, r):
        target_index = np.round(r * (self.accepted_values.shape[0] - 1))

        return float(self.accepted_values[int(target_index.item())])

    def to_real(self, p):
        target_index = np.argmin(np.abs(self.accepted_values - p))

        return np.array([target_index / (self.accepted_values.shape[0] - 1)])

    def random_param(self):
        return float(np.random.choice(self.accepted_values).item())

    def random_real(self):
        return self.to_real(self.random_param())

    def real_size(self):
        return 1

class NominalParamCodec:
    def __init__(self, bounds):
        self.count = len(bounds)

        sorted_names = sorted(bounds)
        assigned_name_indices = list(range(self.count))
        self.name_to_vertex_id_map = dict(zip(sorted_names, assigned_name_indices))
        self.vertex_id_to_name_map = sorted_names

        self.vertices_for_values = Math.k_simplex(self.count)

    def to_param(self, vec):
        distance_vec = self.vertices_for_values - vec
        target_index = np.argmin(np.sum(distance_vec * distance_vec, axis=1), axis=0)

        return self.vertex_id_to_name_map[target_index]

    def to_real(self, name):
        return self.vertices_for_values[self.name_to_vertex_id_map[name], :]

    def random_real(self):
        return self.vertices_for_values[np.random.choice(self.count), :]

    def random_param(self):
        return self.vertex_id_to_name_map[np.random.choice(self.count)]

    def real_size(self):
        return self.vertices_for_values.shape[1]