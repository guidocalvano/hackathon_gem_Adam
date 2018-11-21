import DataImporter from DataImporter

config = Configuration.get()

description = config.parse_arguments()

DataImporter.importData(**config["data_importer"])
