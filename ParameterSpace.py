import json
import argparse
import sys

class ParameterSpace:

    Singleton = None

    @staticmethod
    def load(filePath='./defaults.json'):
        return ParameterSpace(json.load(open(filePath, 'r')))

    def __init__(self, config):

        self.config = config
        self.pathSeparator = '/'

        self.changesToDefault = {}

        self.config_name = 'default'
        self.config_name_arg_separator = '#'

    def copy(self):

        return ParameterSpace(json.loads(json.dumps(self.config)))

    def parseArguments(self, args=None):
        copy = self.copy()
        args = args if args is not None else sys.argv[2:]

        pathToType = copy.getPathToTypeDictionary()

        parser = argparse.ArgumentParser(add_help=False)

        # parser.add_argument('--help', '-h', action='store_true')

        for path in pathToType.keys():
            parser.add_argument('--' + path, type=pathToType[path])

        parsed = parser.parse_args(args)

        copy.apply_arguments(vars(parsed))

        copy.try_collapse_parameter_arrays()

        return copy.config

    def getPathToTypeDictionary(self, config=None, pathOffset='', pathToTypeDictionary=None):
        config = config if config is not None else self.config
        pathToTypeDictionary = pathToTypeDictionary if pathToTypeDictionary is not None else {}

        if not isinstance(config, dict):
            pathToTypeDictionary[pathOffset] = type(config)
            return

        for key in config.keys():
            if len(pathOffset) > 0:
                pathOffset = pathOffset + self.pathSeparator

            self.getPathToTypeDictionary(config[key], pathOffset + key, pathToTypeDictionary)

        return pathToTypeDictionary

    def try_collapse_paremeter_arrays(self):

        if not isinstance(self.config, dict): return

        task_stack = [{"config": self.config, "path": []}]

        while len(task_stack) > 0:
            next_task = task_stack.pop()

            for key, value in list(next_task["config"].items()):
                next_path = next_task["path"].copy()
                next_path.append(key)
                if isinstance(next_task["config"][key], dict):
                    task_stack.append({
                        "config": next_task["config"][key],
                        "path": next_path
                    })
                    continue

                if not isinstance(next_task["config"][key], list):
                    continue

                if len(next_task["config"][key]) != 1:
                    print("Cannot collapse parameter arrays. Parameter is not defined: " + json.dumps(next_path))
                    print("Aborting")
                    sys.exit()

                next_task["config"][key] = next_task["config"][key][0]

    def flatten_parameters(self):

        if not isinstance(self.config, dict): return {}

        task_stack = [{"config": self.config, "path": []}]
        flattened_parameters = {}

        while len(task_stack) > 0:
            next_task = task_stack.pop()

            for key, value in list(next_task["config"].items()):
                next_path = next_task["path"].copy()
                next_path.append(key)
                if isinstance(next_task["config"][key], dict):
                    task_stack.append({
                        "config": next_task["config"][key],
                        "path": next_path
                    })
                    continue

                if not isinstance(next_task["config"][key], list):
                    continue

                if len(next_task["config"][key]) == 1:
                    continue

                flattened_parameters[self.pathSeparator.join(next_path)] = next_task["config"][key]

        return flattened_parameters

    def apply_instantiation(self, flattened_instantiation):
        copy = self.copy()
        copy.apply_arguments(flattened_instantiation)

        copy.try_collapse_paremeter_arrays()

        return copy.config

    def setPath(self, path, value):

        cursor = self.config
        changesCursor = self.changesToDefault

        pathNodes = path.split(self.pathSeparator)

        for i in range(len(pathNodes) - 1):
            nextNode = pathNodes[i]

            if nextNode not in changesCursor:
                changesCursor[nextNode] = {}

            cursor = cursor[nextNode]
            changesCursor = changesCursor[nextNode]

        if not isinstance(cursor[pathNodes[-1]], list) or len(cursor[pathNodes[-1]]) < 2 or (value not in cursor[pathNodes[-1]]):
            print("Attempting to assign invalid value to parameter in this parameter space: \n" +
                  str(value) + " at path " + json.dumps(path))

            print("You are advised to create a new parameter space that allows assignment of the value.")
            print("Aborting execution")
            sys.exit()

        cursor[pathNodes[-1]] = value
        changesCursor[pathNodes[-1]] = value

    def apply_arguments(self, parsed):

        for path in parsed.keys():
            if parsed[path] is not None:
                self.setPath(path, parsed[path])
