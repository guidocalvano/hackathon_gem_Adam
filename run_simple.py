from Configuration import Configuration
from DataImporter import DataImporter
from Learning import Learning



config = Configuration.load()

description = config.parseArguments()

print("ARGUMENTS PARSED: " + description)

config = config.config

di = DataImporter()

data_set_dictionary = di.dataImporter.import_all_data(config["data_import"]["data_description_file_path"], config["data_import"]["standardized_photos"])

print("DATA IMPORTED")

binary_results = Learning.simple_binary_classification(data_set_dictionary, epochs=config["learning"]["epochs"], batch_size=config["learning"]["batch_size"])
print("BINARY")
print("binary_results")
print(binary_results)

type_results = Learning.simple_categorical_classification(data_set_dictionary, epochs=config["learning"]["epochs"], batch_size=config["learning"]["batch_size"])
print("TYPE")
print("type_results")
print(type_results)

crow_results = Learning.simple_crow_score_regression(data_set_dictionary, epochs=config["learning"]["epochs"], batch_size=config["learning"]["batch_size"])
print("CROW")
print("crow_results")
print(crow_results)

print("ALL")
print({
    "binary": binary_results,
    "type": type_results,
    "crow": crow_results
})