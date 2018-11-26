from Configuration import Configuration
from Analysis import Analysis
import os.path

config = Configuration.load()

description = config.parseArguments()

print("ARGUMENTS PARSED: " + description)

config = config.config

binary_result_path = os.path.join(config["analysis"]["result_base_path"], "binary")
Analysis.process_result(binary_result_path)

type_result_path = os.path.join(config["analysis"]["result_base_path"], "type")
Analysis.process_result(type_result_path)

crow_result_path = os.path.join(config["analysis"]["result_base_path"], "crow")
Analysis.process_result(crow_result_path)

print("processed results")
