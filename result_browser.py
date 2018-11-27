import os
import json
from pathlib import Path

from Configuration import Configuration
config = Configuration.load()
config = config.config


from flask import Flask, request, send_from_directory, safe_join
app = Flask(__name__, static_url_path='', static_folder='./result_browser')

@app.route('/')
def index():
    print("index.html")
    return app.send_static_file('index.html')

@app.route('/public/<path:file_path>')
def public_files(file_path):
    return send_from_directory('result_browser', file_path)

def recursive_remove_empty_dicts(root_dict):
    keys = list(root_dict.keys())

    for key in keys:
        value = root_dict[key]
        if not isinstance(value, dict):
            return
        else:
            recursive_remove_empty_dicts(value)

        if len(value.keys()) == 0:
            del root_dict[key]

@app.route('/data_index')
def data_index():
    part_count_base_path = len(Path(config["analysis"]["result_base_path"]).parts)

    index_json = {}

    for path_to_files, dirs, files in os.walk(config["analysis"]["result_base_path"]):

        dir_cursor = index_json
        for dir in Path(path_to_files).parts[part_count_base_path:]:
            if not dir in dir_cursor:
                dir_cursor[dir] = {}

            dir_cursor = dir_cursor[dir]

        for file in files:
            if file in config["analysis"]["result_browser_whitelist"]:
                dir_cursor[file] = os.path.join('data', *Path(path_to_files).parts[part_count_base_path:], file)

    recursive_remove_empty_dicts(index_json)

    return json.dumps(index_json)

@app.route('/data/<path:file_path>')
def data(file_path):

    path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)

    if file_name not in config["analysis"]["result_browser_whitelist"]:
        return "Cannot deliver file, it is not whitelisted"

    return send_from_directory(safe_join(config["analysis"]["result_base_path"], path), file_name)

if __name__ == "__main__":
    app.run()