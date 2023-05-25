import numpy as np
import random
import argparse

from data import read_train, read_test, get_settings, save_processed_train, save_processed_val, save_processed_test

# --------------------------------------------------------------------------- #
# Parse command line arguments

parser = argparse.ArgumentParser()

parser.add_argument(
    "-v2",
    "--version2",
    action="store_true",
    help="Create 2 validation sets instead of 1"
    )

parser.add_argument(
    '-s',
    '--splits',
    default=[0.7, 0.2, 0.1],
    nargs=3,
    metavar=('train', 'val1', 'val2'),
    type=float,
    help='specify the train, val1 and val2 splits'
)

args = parser.parse_args()

# --------------------------------------------------------------------------- #

def calculate_proportion_of_left_data(data, threshold=0.8):
    """
    Function to calculate proportion of the data which will remain in the dataset based on the threshold for NAs.

    :param data: dataset
    :param threshold: NAs amount that is critical to remove the column

    :return: (proportion of data left, columns to drop based on threshold)
    """
    cols_to_drop = data.columns[data.isna().sum() / len(data) > threshold].values
    upd_data = data.drop(cols_to_drop, axis=1)
    print(f'{data.shape} -> {upd_data.shape}')

    return (~upd_data.isna()).sum().sum() / (~data.isna()).sum().sum(), cols_to_drop


def create_target(data, check_randomness=False):
    """
    Function to create rank for every property.

    :param check_randomness: parameter that show do we need to apply position bias with respect to randomness
    :param data: dataframe
    :return: composed target values
    """

    data['target'] = 0
    data.loc[data['click_bool'] == 1, 'target'] = 1
    data.loc[data['booking_bool'] == 1, 'target'] = 5

    if check_randomness:
        mask = data.random_bool == 0
        data.loc[mask, 'target'] = data[mask, 'target'] * get_settings()['click_weight'] / np.log(data['position'] + 1)
    else:
        data['target'] = data['target'] * get_settings()['click_weight']/np.log(data['position'] + 1)

    return data


def split_search_ids(search_ids, train_frac=0.7, val_frac=0.2):
    """
    Function to split dataset into train-val-test based on search_ids.

    :param search_ids: set of unique search_ids
    :param train_frac: ratio of data in trainset
    :param valid_frac: ration of data in valset
    :return: dictionary with mapping btw search_id and set
    """
    index = np.arange(len(search_ids))
    np.random.shuffle(index)

    train_end = int(train_frac * len(search_ids))
    valid_end = int((train_frac + val_frac) * len(search_ids))

    train = index[:train_end]
    valid = index[train_end:valid_end]
    test = index[valid_end:]

    search_ids_split = dict()
    for i in train:
        search_ids_split[search_ids[i]] = "train"
    for i in valid:
        search_ids_split[search_ids[i]] = "val"
    for i in test:
        search_ids_split[search_ids[i]] = "test"

    return search_ids_split


def create_train_val_test_sets():
    """
    Function to create train val split of initial train data. Splitted data is saved to three separate compressed files.
    """
    data = read_train()
    test_data = read_test()

    # removing columns with na amount > threshold
    _, cols_to_drop = calculate_proportion_of_left_data(data, threshold=get_settings()['na_threshold'])
    data = data.drop(cols_to_drop, axis=1)
    test_data = test_data.drop(set(cols_to_drop).intersection(set(test_data.columns)), axis=1)
    print(f'Columns were deleted: {cols_to_drop}')

    # filling na with median value for every numeric column, others - 0
    print(f'Columns with NAs: {data.columns[data.isnull().any()]}')
    data = data.fillna(data.median(numeric_only=True))
    test_data = test_data.fillna(data.median(numeric_only=True))

    # creating simple target variable
    data = create_target(data)  # on this step 'gross_bookings_usd' is not in axis
    # data = data.drop(['position', 'click_bool', 'booking_bool'], axis=1)  # remove cols which are not in test set

    # splitting into train/val/test
    srch_id_div = split_search_ids(data.srch_id.unique(), train_frac=0.8, val_frac=0.2)
    data['split'] = data['srch_id'].map(srch_id_div)

    assert (set(data.columns) - set(test_data.columns)) == {'target', 'split', 'position', 'click_bool', 'booking_bool'}

    # save datasets
    print(f'Saving trainset - {data[data.split == "train"].shape}.')
    save_processed_train(data[data.split == 'train'].drop(['split'], axis=1))
    print(f'Saving valset - {data[data.split == "val"].shape}.')
    save_processed_val(data[data.split == 'val'].drop(['split'], axis=1))
    print(f'Saving testset - {test_data.shape}.')
    save_processed_test(test_data)

def create_balanced_train_val_test_sets(train_frac=0.7, val1_frac=0.2):
    # Read raw data
    data = read_train()
    test_data = read_test()

    # removing columns with na amount > threshold
    _, cols_to_drop = calculate_proportion_of_left_data(data, threshold=get_settings()['na_threshold'])
    data = data.drop(cols_to_drop, axis=1)
    test_data = test_data.drop(set(cols_to_drop).intersection(set(test_data.columns)), axis=1)
    print(f'Columns were deleted: {cols_to_drop}')

    # filling na with median value for every numeric column, others - 0
    print(f'Columns with NAs: {data.columns[data.isnull().any()]}')
    data = data.fillna(data.median(numeric_only=True))
    test_data = test_data.fillna(data.median(numeric_only=True))

    # creating simple target variable
    data = create_target(data)

    # Splitting into train/val/test
    srch_ids = data['srch_id'].unique()
    random.shuffle(srch_ids)

    # Train
    train_end = int(train_frac * len(srch_ids))
    train_ids = srch_ids[:train_end]
    train_df = data[data['srch_id'].isin(train_ids)]
    train_df = train_df.sort_values(by=['srch_id'])

    # Validation 1
    val1_end = int((train_frac + val1_frac) * len(srch_ids))
    val1_ids = srch_ids[train_end:val1_end]
    val1_df = data[data['srch_id'].isin(val1_ids)]
    val1_df = val1_df.sort_values(by=['srch_id'])

    # Validation 2
    val2_ids = srch_ids[val1_end:]
    val2_df = data[data['srch_id'].isin(val2_ids)]
    val2_df = val2_df.sort_values(by=['srch_id'])

    # Save datasets
    print(f'Saving trainset - {train_df.shape}.')
    save_processed_train(train_df)
    print(f'Saving valset 1 - {val1_df.shape}.')
    save_processed_val(val1_df, num=1)
    print(f'Saving valset 2 - {val2_df.shape}.')
    save_processed_val(val2_df, num=2)
    print(f'Saving testset - {test_data.shape}.')
    save_processed_test(test_data)


if __name__ == '__main__':
    if args.version2:
        if sum(x*10 for x in args.splits) != 10:
            raise ValueError('Sum of the splits must be equal to 1')
        else:
            print(f'Creating train/val1/val2 sets with splits {args.splits}')
            create_balanced_train_val_test_sets(train_frac=args.splits[0], val1_frac=args.splits[1])
    else:
        create_train_val_test_sets()
