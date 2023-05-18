from datetime import datetime
from pathlib import Path

import pandas as pd
from data import load_model, read_processed_test, read_processed_val, write_submission, write_solution, get_paths, get_settings


def predict(is_test=True):
    """
    Function to predict ranking.
    :param is_test: flag to detect which dataset we are using now
    :return:
    """
    folder_path = (
            get_paths()['results'] /
            Path(get_settings()['model_name']) /
            Path("{:%Y%m%d_%H%M%S}/".format(datetime.now()))
    )

    # Data preparation
    print('Loading the data...')
    if is_test:
        data = read_processed_test()
    else:
        data = read_processed_val()

    data = data.set_index(['srch_id', 'prop_id'], drop=True)
    print('Data is loaded.')

    feature_names = list(data.columns)
    for col in ['date_time', 'target']:
        if col in feature_names:
            feature_names.remove(col)

    features = data[feature_names].values
    model = load_model()
    predictions = model.predict(features)
    solution = pd.Series(predictions, index=data.index, name='ranks')

    print('Saving results...')
    write_solution(solution, folder_path)
    write_submission(solution, folder_path)
    print(f'Results are saved to {folder_path}.')


if __name__ == '__main__':
    predict(is_test=True)
