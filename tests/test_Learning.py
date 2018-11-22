import unittest
from Learning import Learning


class test_Learning(unittest.TestCase):

    def setUp(self):
        self.unstandardized_photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'learning', 'unstandardized_photos')
        self.photos_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'learning', 'photos')

        self.dataImporter.convert_to_standard_resolution(
            self.unstandardized_photos_path,
            self.photos_path,
            [152, 114],
            {
                "multi_core_count": 2,
                "timeout_secs": 100,
                "chunk_size": 2
            }
        )

    def tearDown(self):
        pass

    def simple_binary_classification(self):

        data_set = self.dataImporter.import_all_data(self.test_photos_csv_file_path, self.photos_path)

        results = Learning.simple_binary_classification(data_set)

        pass  # for breakpoints