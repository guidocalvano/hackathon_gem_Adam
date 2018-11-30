from Math import Math
import unittest
from scipy.spatial.distance import pdist
import numpy as np


class test_Math(unittest.TestCase):
    def test_KSimplex(self):
        kSimplex1000 = Math.k_simplex(500)

        # assert that all distances between the points in the simplex are almost exactly 1
        self.assertTrue(np.all(np.abs(pdist(kSimplex1000) - 1) < .000001))
