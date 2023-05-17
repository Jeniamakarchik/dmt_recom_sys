import pandas as pd
from data import load_model, read_processed_test, read_processed_val, write_submission, write_solution


def predict(is_test=True):
    """
    Function to predict ranking.
    :param is_test: flag to detect which dataset we are using now
    :return:
    """
    # Data preparation
    if is_test:
        data = read_processed_test()
    else:
        data = read_processed_val()

    data = data.set_index(['srch_id', 'prop_id'], drop=True)

    feature_names = list(data.columns)
    for col in ['date_time', 'target']:
        if col in feature_names:
            feature_names.remove(col)

    features = data[feature_names].values
    model = load_model()
    predictions = model.predict(features)
    solution = pd.Series(predictions, index=data.index, name='ranks')
    write_solution(solution)
    write_submission(solution)


if __name__ == '__main__':
    predict(is_test=True)
