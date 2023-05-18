import csv
import json
from operator import itemgetter
from pathlib import Path

import pickle
import pandas as pd


def get_settings():
    with open('settings.json') as set_file:
        settings = json.loads(set_file.read())

    return settings


def get_paths():
    with open('paths.json') as set_file:
        paths = json.loads(set_file.read())

    for key in paths:
        paths[key] = Path(paths[key])

    return paths


def read_data(path):
    return pd.read_csv(path)


def read_processed_data(path):
    return pd.read_csv(path, compression='gzip', index_col=['srch_id', 'prop_id'])


def read_train():
    path = get_paths()['train']
    return read_data(path)


def read_test():
    path = get_paths()['test']
    return read_data(path)


def read_processed_train():
    path = get_paths()['processed_data'] / Path('train.csv.gz')
    return read_processed_data(path)


def read_processed_val():
    path = get_paths()['processed_data'] / Path('val.csv.gz')
    return read_processed_data(path)


def read_processed_test():
    path = get_paths()['processed_data'] / Path('test.csv.gz')
    return read_processed_data(path)


def save_compressed_data(data, path):
    data.to_csv(path, chunksize=100000, compression='gzip')


def save_processed_train(data):
    path = get_paths()['processed_data'] / Path('train.csv.gz')
    save_compressed_data(data, path)


def save_processed_val(data):
    path = get_paths()['processed_data'] / Path('val.csv.gz')
    save_compressed_data(data, path)


def save_processed_test(data):
    path = get_paths()['processed_data'] / Path('test.csv.gz')
    save_compressed_data(data, path)


def save_model(model):
    out_path = get_paths()['models_folder'] / Path(get_settings()['model_name'])
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)


def load_model():
    in_path = get_paths()['models_folder'] / Path(get_settings()['model_name'])
    with open(in_path, 'rb') as f:
        model = pickle.load(f)
    return model


def write_solution(solution):
    """ with rank """
    solution.to_csv('solution.csv')


def write_submission(solution):
    """ without ranks, sorted """
    submission_path = get_paths()['submission']
    sorted_solution = solution.reset_index().sort_values(by=['srch_id', 'ranks'], ascending=[True, False])
    sorted_solution = sorted_solution.drop(['ranks'], axis=1)

    sorted_solution.to_csv(submission_path, index=False)
