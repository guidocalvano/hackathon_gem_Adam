from Configuration import Configuration
from DataImporter import DataImporter
import time


config = Configuration.load()

description = config.parseArguments()

config = config.config["data_import"]

di = DataImporter()
start = time.time()
di.convert_to_standard_resolution(
    config["unstandardized_photos"],
    config["standardized_photos"],
    config["output_image_dimensions"],
    config["loading_config"]
)
after_convert_t = time.time()
print('standardizing ')
print(after_convert_t - start)
# DataImporter.load_from_cache(
#     config["cache_file_path"],
#     config["data_description_file_path"],
#     config["standardized_photos"]
# )
after_load_t = time.time()
print('normalizing ')
print(after_load_t - after_convert_t)