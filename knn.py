import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from data import read_processed_knn_train, read_processed_knn_val, read_processed_knn_test, save_model
import numpy as np
import argparse


def train_knn(debug=False):
    
    print('Reading training data')
    test_data = read_processed_knn_test()
    train_data = read_processed_knn_train()
    val_data = read_processed_knn_val()
    print("Data loaded")

    # If in debug mode, use only a subset of the data.
    if debug == True:
        print('Debug mode')
        train_data = train_data.sample(n=10)
        test_data = test_data.sample(n=100)
        val_data = val_data.sample(n=10)
    
    # Divide the data into features and target
    train_features = train_data.drop(['srch_id', 'prop_id', 'target', 'date_time'], axis=1)
    train_target = train_data['target']
    val_features = val_data.drop(['srch_id', 'prop_id', 'target','date_time'], axis=1)
    val_target = val_data['target']
    test_features = test_data.drop(['srch_id', 'prop_id', 'date_time'], axis=1)

    # Train a kNN regressor
    model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    model.fit(train_features, train_target)

    print("Saving the model")
    save_model(model)

    print('Evaluate model')

    # Predict the target for the validation and test set
    val_pred = model.predict(val_features)
    test_pred = model.predict(test_features)

    # Add the predicted target to the validation set
    df_pred = val_features.copy()
    df_pred['srch_id'] = val_data.loc[val_features.index, 'srch_id']
    df_pred['prop_id'] = val_data.loc[val_features.index, 'prop_id']
    df_pred['predicted_target'] = val_pred
    df_pred['target'] = val_data.loc[val_features.index, 'target']

    # Add the predicted target to the test set
    df_test = test_features.copy()
    df_test['srch_id'] = test_data.loc[test_features.index, 'srch_id']
    df_test['prop_id'] = test_data.loc[test_features.index, 'prop_id']
    df_test['predicted_target'] = test_pred

    # Rank the items within each srch_id group 
    df_pred['rank'] = df_pred.groupby('srch_id')['predicted_target'].rank(ascending=False)
    df_test['rank'] = df_test.groupby('srch_id')['predicted_target'].rank(ascending=False)

    # Sort the items by srch_id and rank
    df_pred = df_pred.sort_values(['srch_id', 'rank'], ascending=[True, False])
    df_test = df_test.sort_values(['srch_id', 'rank'], ascending=[True, False])

    val_ndcg = calculate_ndcg(df_pred, 5)
    print(f'The validation NDCG@5 = {val_ndcg}')

    # Save the srch_id and prop_id of the ranked items to a CSV file
    df_pred[['srch_id', 'prop_id']].to_csv('ranking_knn/ranking_val.csv', index=False)
    df_test[['srch_id', 'prop_id']].to_csv('ranking_knn/ranking_test.csv', index=False)


def dcg_at_k(r, k=5):
    """Discounted Cumulative Gain (DCG)"""
    r = np.asfarray(r)[:k]
    return r[0] + np.sum(r[1:] / np.log2(np.arange(3, r.size + 2)))

def ndcg_at_k(r, k=5):
    """Normalized Discounted Cumulative Gain (NDCG)"""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def calculate_ndcg(df, k=5):
    ndcg_scores = []
    for srch_id, group in df.groupby('srch_id'):
        true_ranking = group.sort_values('target', ascending=False)['predicted_target'].tolist()
        ndcg_score = ndcg_at_k(true_ranking, k)
        ndcg_scores.append(ndcg_score)

    return sum(ndcg_scores) / len(ndcg_scores)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()

    #train_knn(debug=args.debug)
    train_knn()


# k = 5