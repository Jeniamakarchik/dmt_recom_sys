from datetime import datetime
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
    return pd.read_csv(path, compression='gzip')

def read_processed_knn_train():
    path = get_paths()['processed_data'] / Path('train.csv.gz')
    return pd.read_csv(path, compression='gzip')

def read_processed_knn_val():
    path = get_paths()['processed_data'] / Path('val.csv.gz')
    return pd.read_csv(path, compression='gzip')

def read_processed_knn_test():
    path = get_paths()['processed_data'] / Path('test.csv.gz')
    return pd.read_csv(path, compression='gzip')

def read_train():
    path = get_paths()['train']
    return read_data(path)


def read_test():
    path = get_paths()['test']
    return read_data(path)


def read_processed_train():
    path = get_paths()['processed_data'] / Path('train.csv.gz')
    return read_processed_data(path)


def read_processed_val(num=None):
    if num:
        path = get_paths()['processed_data'] / Path(f'val_{num}.csv.gz')
    else:
        path = get_paths()['processed_data'] / Path('val.csv.gz')
    return read_processed_data(path)


def read_processed_test():
    path = get_paths()['processed_data'] / Path('test.csv.gz')
    return read_processed_data(path)


def save_compressed_data(data, path):
    data.to_csv(path, chunksize=100000, compression='gzip', index=False)


def save_processed_train(data):
    path = get_paths()['processed_data'] / Path('train.csv.gz')
    save_compressed_data(data, path)


def save_processed_val(data, num=None):
    if num:
        path = get_paths()['processed_data'] / Path(f'val_{num}.csv.gz')
    else:
        path = get_paths()['processed_data'] / Path('val.csv.gz')
    save_compressed_data(data, path)


def save_processed_test(data):
    path = get_paths()['processed_data'] / Path('test.csv.gz')
    save_compressed_data(data, path)


def save_model(model):
    out_path = get_paths()['models_folder'] / Path(f'{get_settings()["model_name"]}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)


def load_model():
    in_path = get_paths()['models_folder'] / Path(f'{get_settings()["model_name"]}.pkl')
    with open(in_path, 'rb') as f:
        model = pickle.load(f)
    return model


def write_solution(solution, folder_path):
    """
    Function to save solution with scores.

    :param solution: solution for the search with 'srch_id', 'prop_id' and 'ranks'
    :return:
    """
    folder_path.mkdir(parents=True, exist_ok=True)
    solution.to_csv(folder_path / Path('solution.csv'))


def write_submission(solution, folder_path):
    """
    Function to create and save the submission.

    :param solution: solution for the search with 'srch_id', 'prop_id'
    :return:
    """
    folder_path.mkdir(parents=True, exist_ok=True)
    sorted_solution = solution.reset_index().sort_values(by=['srch_id', 'ranks'], ascending=[True, False])
    sorted_solution = sorted_solution.drop(['ranks'], axis=1)
    sorted_solution.to_csv(folder_path / Path('submission.csv'), index=False)
