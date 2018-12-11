import json
import argparse
import sys

class ParameterSpace:

    Singleton = None

    @staticmethod
    def load(filePath='./defaults.json'):
        with open(filePath, 'r') as f:
            return ParameterSpace(json.load(f))

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

            if isinstance(config, list) and len(config) >= 1:
                # if config is length 1 then it is a constant and the type should be the single element
                # if config is larger than length 1, then it is not a constant but a list of parameter options
                # and the type should be the type of the first element (which should be the same as all subsequent elements).
                pathToTypeDictionary[pathOffset] = type(config[0])
                return

            pathToTypeDictionary[pathOffset] = list

            return

        if len(pathOffset) > 0:
            pathOffset = pathOffset + self.pathSeparator

        for key in config.keys():

            self.getPathToTypeDictionary(config[key], pathOffset + key, pathToTypeDictionary)

        return pathToTypeDictionary

    def try_collapse_parameter_arrays(self):

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

        copy.try_collapse_parameter_arrays()

        name = self.configuration_name(flattened_instantiation)
        copy.config["configuration_name"] = name

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

        try:
            if not isinstance(cursor[pathNodes[-1]], list): raise Exception("Trying to instantiate parameter, but path does not specify parameter.")
            if len(cursor[pathNodes[-1]]) < 2: raise Exception("not a property")
            if value not in cursor[pathNodes[-1]]: raise Exception("invalid option")

            if isinstance(value, list): value = [value]

            cursor[pathNodes[-1]] = value
            changesCursor[pathNodes[-1]] = value
        except Exception as e:
            print(e)
            print("Attempting to assign invalid value to parameter in this parameter space: \n" +
                  str(value) + " at path " + json.dumps(path))

            print("You are advised to create a new parameter space that allows assignment of the value.")
            print("Aborting execution")
            sys.exit()

    def getPath(self, path):

        cursor = self.config

        pathNodes = path.split(self.pathSeparator)

        for i in range(len(pathNodes)):
            nextNode = pathNodes[i]

            cursor = cursor[nextNode]

        return cursor

    def apply_arguments(self, parsed):

        for path in parsed.keys():
            if parsed[path] is not None:
                value = parsed[path]

                self.setPath(path, value)

    def get_configuration_grid(self):
        # To create the configuration grid without resorting to
        # complicated recursive functions I number all
        # possible configurations with an integer,
        # and then convert that integer into a different number
        # system, where each digit choses one parameter value.
        # The most practical number system is multi radix, i.e. it does not use a single
        # 'radix' (2 is the radix of a binary number system, 3 of a ternary
        # and 10 of a decimal number system). Instead it uses multiple radixes.
        # A good example of a multi radix system is time expressed in years, days,
        # hours, minutes and seconds.
        # Applied to this case: If you have 3 options for param 1, 5 for param 2 and 2 for param 3
        # then the first digit ranges from 0 to 2, the second from 0 to 4 and the last
        # is either 0 or 1. Whenever a digit increases beyond its range, it resets to 0,
        # and the next digit increases by 1, until it too increases beyond its range, and
        # the next digit increases by 1, and so one recursively.
        # To determine the value of the highest digit we must know how large a single unit is.
        # To compute the value of the highest digit, divide by the size of a single unit of
        # the highest digit, and discard the fraction. To compute the next digit, subtract
        # the highest digit, and divide the remaining value by the size of a unit of the next
        # digit, and continue for all digits recursively.

        flattened_space = self.flatten_parameters()

        flattened_configuration_list = self.list_all_flattened_parameter_configurations(flattened_space)

        full_configuration_list = list(map(lambda fc: self.apply_instantiation(fc), flattened_configuration_list))

        return full_configuration_list

    def list_all_flattened_parameter_configurations(self, flattened_space):

        mixed_radix_unit_size, space_size = self._flattened_space_unit_sizes(flattened_space)

        flattened_index_list = self._flattened_space_unit_size_to_index_lists(mixed_radix_unit_size, space_size)

        flattened_configuration_list = list(map(lambda flattened_index: self._flat_space_index_to_configuration(flattened_space, flattened_index), flattened_index_list))

        return flattened_configuration_list

    def _flattened_space_unit_sizes(self, flattened_space):

        mixed_radix_unit_size = {}
        product = 1

        for parameter, options in reversed(sorted(flattened_space.items())):
            mixed_radix_unit_size[parameter] = product

            product *= len(options)

        space_size = product

        return mixed_radix_unit_size, space_size

    def _flattened_space_unit_size_to_index_lists(self, mixed_radix_unit_size, space_size):
        index_list = []
        for i in range(space_size):
            next_configuration = self._configuration_index_to_parameter_indices(mixed_radix_unit_size, i)
            index_list.append(next_configuration)

        return index_list

    def _configuration_index_to_parameter_indices(self, mixed_radix_unit_size, still_unbroken_index):
        next_configuration = {}

        for parameter, options in sorted(mixed_radix_unit_size.items()):
            scale = mixed_radix_unit_size[parameter]
            next_digit = int(still_unbroken_index / scale)
            still_unbroken_index -= next_digit * scale  # note that cast to int rounded down
            next_configuration[parameter] = next_digit  # i.e. pick the parameter

        return next_configuration

    def _flat_space_index_to_configuration(self, flattened_space, index_map):

        configuration = {}

        for key, values in flattened_space.items():
            configuration[key] = values[index_map[key]]

        return configuration

    def configuration_name(self, flattened_parameters):
        parameter_configuration_name = ""

        for param, value in list(sorted(flattened_parameters.items())):
            if parameter_configuration_name != "":
                parameter_configuration_name = parameter_configuration_name + "/"

            parameter_configuration_name = parameter_configuration_name + param.replace('/', '@') + '@' + str(
                value)

        return parameter_configuration_name