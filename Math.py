import numpy as np


class Math:

    @staticmethod
    def k_simplex(k):  # k dimensional tetrahedron
        if k == 1: return np.array([]).reshape([1, 0])

        k_minus_1_simplex = Math.k_simplex(k - 1)

        centroid = np.sum(k_minus_1_simplex, axis=0, keepdims=True) / k_minus_1_simplex.shape[0]

        # a2 + b2 = c2
        # a2 = c2 - b2
        # a = sqrt(c2 - b2)

        # b2 = distance(centroid)2
        # c2 = 1
        new_dimension_value = np.sqrt(1 - np.sum(centroid * centroid))

        new_vertex = np.concatenate([centroid, np.array([new_dimension_value]).reshape([1, 1])], axis=1)

        k_minus_1_simplex_plus_dim = np.concatenate([k_minus_1_simplex, np.zeros([k_minus_1_simplex.shape[0], 1])], axis=1)

        k_simplex_val = np.concatenate([k_minus_1_simplex_plus_dim, new_vertex], axis = 0)

        return k_simplex_val
