from data import read_processed_knn_val
from knn import calculate_ndcg
import pandas as pd
import numpy as np

# Load and shuffle data
val_data = read_processed_knn_val()
val_data = val_data.sample(frac=1).reset_index(drop=True)

# Add random targets
val_data['predicted_target'] = np.random.uniform(0, 5, val_data.shape[0])

# Calculate NDCG@5 for random ranking
ndcg = calculate_ndcg(val_data, 5)
print(f'The NDCG@% score for random ranking is {ndcg}')


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

