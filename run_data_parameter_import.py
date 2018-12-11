
from ParameterSpace import ParameterSpace
from DataImporter import DataImporter
import os
import time
import sys


def run(argv):

    parameter_space_file_path = argv[1]

    parameter_space = ParameterSpace(ParameterSpace.load(parameter_space_file_path).config["data_import"])

    configurations = parameter_space.get_configuration_grid()

    for config in configurations:
        di = DataImporter()
        start = time.time()

        sink_path = os.path.join(config["standardized_photos"], config["configuration_name"])

        di.convert_to_standard_resolution(
            config["unstandardized_photos"],
            sink_path,
            config["output_image_dimensions"],
            config["loading_config"]
        )
        after_convert_t = time.time()
        print('standardizing ')
        print(after_convert_t - start)


if __name__ == '__main__':
    run(sys.argv)
