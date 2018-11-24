from Configuration import Configuration
from DataImporter import DataImporter
from Learning import Learning
from Analysis import Analysis
import os
import os.path


config = Configuration.load()

description = config.parseArguments()

print("ARGUMENTS PARSED: " + description)

config = config.config

di = DataImporter()

data_set_dictionary = di.import_all_data(config["data_import"]["data_description_file_path"], config["data_import"]["standardized_photos"])

print("DATA IMPORTED")

binary_results = Learning.simple_binary_classification(data_set_dictionary, epochs=config["learning"]["epochs"], batch_size=config["learning"]["batch_size"])
print("BINARY")
print("binary_results")
print(binary_results)
binary_result_path = os.path.join(config["analysis"]["result_base_path"], "binary")
Analysis.store_raw_result(binary_result_path, binary_results)
Analysis.process_result(binary_result_path)

type_results = Learning.simple_categorical_classification(data_set_dictionary, epochs=config["learning"]["epochs"], batch_size=config["learning"]["batch_size"])
print("TYPE")
print("type_results")
print(type_results)
type_result_path = os.path.join(config["analysis"]["result_base_path"], "type")
Analysis.store_raw_result(type_result_path, type_results)
Analysis.process_result(type_result_path)


crow_results = Learning.simple_crow_score_regression(data_set_dictionary, epochs=config["learning"]["epochs"], batch_size=config["learning"]["batch_size"])
print("CROW")
print("crow_results")
print(crow_results)
crow_result_path = os.path.join(config["analysis"]["result_base_path"], "crow")
Analysis.store_raw_result(crow_result_path, crow_results)
Analysis.process_result(crow_result_path)


print("ALL")
print({
    "binary": binary_results,
    "type": type_results,
    "crow": crow_results
})