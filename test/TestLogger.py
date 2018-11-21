import unittest
import datetime
import os
import pandas as pd
import numpy as np
from Logger import Logger


class TestLogger(unittest.TestCase):

    def setUp(self):
        self.tearDown()

        self.two_rows = pd.DataFrame(
            {
                'datetime':
                    [str(datetime.datetime.now()), str(datetime.datetime.now())],
                'level': ['x', 'y'],
                'type': ['x', 'y'],
                'description': ['x', 'y'],
                'meta': ['x', 'y']
            })

        base_path = os.path.dirname(os.path.realpath(__file__))


        self.two_rows_file_path = os.path.join(base_path, 'resources', 'logger', 'two_rows.log')
        self.two_rows.to_csv(self.two_rows_file_path, header=True, index=False)

        self.no_file_path = os.path.join(base_path, 'resources', 'logger', 'no_file.log')

    def tearDown(self):
        if hasattr(self, 'two_rows_file_path'): os.remove(self.two_rows_file_path )

        if hasattr(self, 'no_file_path') and os.path.isfile(self.no_file_path):
            os.remove(self.no_file_path)

    def test_log_to_new_file(self):
        before = str(datetime.datetime.now())
        l = Logger(self.no_file_path)

        l.l('a', 'b', 'c')
        l.l('e', 'f', 'g')

        after = str(datetime.datetime.now())

        result = pd.read_csv(self.no_file_path)

        self.assertTrue(result.datetime[0] < result.datetime[1], 'lines must have increasing datetimes')
        self.assertTrue(before < result.datetime[0] and result.datetime[1] < after, 'datetimes must match the time the line was written')

        self.assertTrue(np.all(result.level == 'l'), 'logging levels must be written correctly')
        self.assertTrue(np.all(result.type == ['a', 'e']), 'logging types must be written correctly')
        self.assertTrue(np.all(result.description == ['b', 'f']), 'logging types must be written correctly')
        self.assertTrue(np.all(result.meta == ['c', 'g']), 'logging types must be written correctly')


    def test_log_to_existing_file(self):
        before = str(datetime.datetime.now())
        l = Logger(self.two_rows_file_path)

        l.l('a', 'b', 'c')
        l.i('e', 'f', 'g')

        after = str(datetime.datetime.now())

        result = pd.read_csv(self.two_rows_file_path)

        self.assertTrue(result.datetime[2] < result.datetime[3], 'lines must have increasing datetimes')
        self.assertTrue(before < result.datetime[2] and result.datetime[3] < after, 'datetimes must match the time the line was written')

        self.assertTrue(np.all(result.level == ['x', 'y', 'l', 'i']), 'logging levels must be written correctly')
        self.assertTrue(np.all(result.type == ['x', 'y', 'a', 'e']), 'logging types must be written correctly')
        self.assertTrue(np.all(result.description == ['x', 'y', 'b', 'f']), 'logging types must be written correctly')
        self.assertTrue(np.all(result.meta == ['x', 'y', 'c', 'g']), 'logging types must be written correctly')
