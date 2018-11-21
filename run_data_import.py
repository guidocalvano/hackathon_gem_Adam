from Configuration import Configuration
from DataImporter import DataImporter

config = Configuration.load()

description = config.parse_arguments()

di = DataImporter()
di.convert_to_standard_resolution(
    config["data_importer"]["unstandardized_photos"],
    config["data_importer"]["standardized_photos"],
    config["data_importer"]["output_image_dimensions"],
    config["data_importer"]["loading_config"]
)

DataImporter.load_from_cache(
    config["data_importer"]["cache_file_path"],
    config["data_description_file_path"],
    config["standardized_photos"]
)
